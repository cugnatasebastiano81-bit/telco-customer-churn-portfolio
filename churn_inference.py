"""
churn_inference.py — Modulo di inferenza per il modello Customer Churn.

Qui vive l'unica implementazione del preprocessing del progetto. Il notebook la
importa in Fase 2 e la riusa fino alla serializzazione finale: non esistono due
versioni del preprocessing che possono divergere nel tempo.

Contenuto:
- valida_dataset_sorgente(): controllo anticipato di righe e schema del file
  IBM usato dal notebook, prima dell'analisi esplorativa.
- ChurnFeatureEngineer: pulizia e feature engineering (nessun parametro appreso
  dai dati, quindi applicabile prima o dopo lo split senza rischio di leakage).
- crea_preprocessore(): il preprocessore completo, cioè feature engineering +
  scaling delle numeriche + encoding delle categoriche. Usa OneHotEncoder con
  handle_unknown='ignore', che ha un comportamento documentato e deterministico
  sui valori categorici mai visti in addestramento.
- predici_churn(): inferenza su uno o più clienti (probabilità, decisione alla
  soglia scelta, decile di rischio, opzionalmente i motivi principali).
- contributi_locali() / motivi_principali(): spiegazione locale della singola
  predizione, cioè quali caratteristiche del cliente spingono la sua probabilità
  verso l'alto o verso il basso.
- categorie_non_viste(): diagnostica sui valori categorici assenti dal training.

La pipeline serializzata attesa contiene due step:
    ('prep', preprocessore) → ('clf', LogisticRegression(...))

Colonne raw richieste in input (18):
    Senior Citizen, Partner, Dependents, Tenure Months, Phone Service,
    Multiple Lines, Internet Service, Online Security, Online Backup,
    Device Protection, Tech Support, Streaming TV, Streaming Movies,
    Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Costanti di schema
# ---------------------------------------------------------------------------

RIGHE_DATASET_SORGENTE = 7043

COLONNE_DATASET_SORGENTE = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Gender", "Senior Citizen",
    "Partner", "Dependents", "Tenure Months", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method", "Monthly Charges",
    "Total Charges", "Churn Label", "Churn Value", "Churn Score", "CLTV",
    "Churn Reason",
]

COLONNE_RAW_INPUT = [
    "Senior Citizen", "Partner", "Dependents",
    "Tenure Months", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges",
]

# Colonne che vengono collassate da 3 stati a 2 (il terzo → "No")
COLONNE_DA_COLLASSARE = {
    "Multiple Lines": "No phone service",
    "Online Security": "No internet service",
    "Online Backup": "No internet service",
    "Device Protection": "No internet service",
    "Tech Support": "No internet service",
    "Streaming TV": "No internet service",
    "Streaming Movies": "No internet service",
}

# Servizi opzionali su cui contare i "Yes" per Num Services Optional
SERVIZI_OPZIONALI = [
    "Multiple Lines",
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies",
]

# Colonne binarie Yes/No → 1/0
COLONNE_BINARIE = [
    "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines",
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies",
    "Paperless Billing",
]

# Colonne multi-categoria da codificare in one-hot
COLONNE_ONE_HOT = ["Internet Service", "Contract", "Payment Method"]

# Tutte le categoriche consumate dal modello devono avere un valore esplicito.
# I valori nuovi ma non vuoti restano ammessi e vengono segnalati separatamente.
COLONNE_CATEGORICHE = COLONNE_BINARIE + COLONNE_ONE_HOT

# Colonne numeriche da scalare (due sono create dal feature engineering)
COLONNE_NUMERICHE = [
    "Tenure Months", "Monthly Charges", "Total Charges",
    "Num Services Optional", "Charge Per Tenure",
]

# Feature già 0/1 costruita dal feature engineering: passa senza trasformazioni
COLONNE_FLAG = ["Profilo Rischio"]

# Parametri operativi selezionati sulle predizioni out-of-fold del training.
# Restano come fallback per artefatti precedenti; i nuovi modelli li incorporano.
SOGLIA_RIFERIMENTO = 0.4265

# I 9 cut-point interni dei decili di rischio, calcolati UNA VOLTA sulla
# distribuzione OOF del training. Servono per assegnare
# il decile a QUALSIASI input — anche a un singolo cliente — rispetto alla
# popolazione di riferimento, invece che rispetto ai soli clienti della chiamata.
# Ordine crescente: soglie tra decile 10 (meno a rischio) e decile 1 (più a rischio).
DECILI_RIFERIMENTO = np.array([
    0.02891497, 0.06935953, 0.13629173, 0.22970155, 0.36283307,
    0.49818547, 0.63843384, 0.75546632, 0.85102610,
])

# Etichette leggibili per l'output tecnico delle spiegazioni locali.
# Sono nomi di caratteristiche, non frasi: la direzione ("sotto la media", "no")
# viene aggiunta da motivi_principali in base al valore del singolo cliente.
ETICHETTE_LEGGIBILI = {
    "Tenure Months": "anzianità",
    "Monthly Charges": "spesa mensile",
    "Total Charges": "spesa totale accumulata",
    "Num Services Optional": "servizi opzionali attivi",
    "Charge Per Tenure": "spesa mensile rapportata all'anzianità",
    "Senior Citizen": "cliente senior",
    "Partner": "partner",
    "Dependents": "familiari a carico",
    "Phone Service": "servizio telefonico",
    "Multiple Lines": "linee multiple",
    "Online Security": "servizio Online Security",
    "Online Backup": "servizio Online Backup",
    "Device Protection": "servizio Device Protection",
    "Tech Support": "supporto tecnico",
    "Streaming TV": "streaming TV",
    "Streaming Movies": "streaming film",
    "Paperless Billing": "fattura elettronica",
    "Internet Service_DSL": "connessione DSL",
    "Internet Service_Fiber optic": "connessione in fibra",
    "Internet Service_No": "nessuna connessione internet",
    "Contract_Month-to-month": "contratto mensile",
    "Contract_One year": "contratto annuale",
    "Contract_Two year": "contratto biennale",
    "Payment Method_Bank transfer (automatic)": "pagamento con bonifico automatico",
    "Payment Method_Credit card (automatic)": "pagamento con carta automatica",
    "Payment Method_Electronic check": "pagamento con assegno elettronico",
    "Payment Method_Mailed check": "pagamento con assegno cartaceo",
    "Profilo Rischio": "profilo mensile + fibra + assegno elettronico",
}


def valida_dataset_sorgente(dati, intestazione_originale=None):
    """Rifiuta una variante IBM con righe o intestazione non canoniche.

    ``intestazione_originale`` permette al notebook di passare i nomi letti
    direttamente dalla prima riga Excel, prima che pandas renda univoci in
    automatico eventuali nomi duplicati.
    """
    if not isinstance(dati, pd.DataFrame):
        raise TypeError(
            f"Tipo dataset non gestito: {type(dati)}. Atteso un DataFrame."
        )

    intestazione = (
        list(dati.columns)
        if intestazione_originale is None
        else list(intestazione_originale)
    )
    problemi = []

    if dati.shape[0] != RIGHE_DATASET_SORGENTE:
        problemi.append(
            f"righe attese: {RIGHE_DATASET_SORGENTE:,}; "
            f"righe trovate: {dati.shape[0]:,}"
        )
    if dati.shape[1] != len(COLONNE_DATASET_SORGENTE):
        problemi.append(
            f"colonne attese: {len(COLONNE_DATASET_SORGENTE)}; "
            f"colonne trovate: {dati.shape[1]}"
        )
    if len(intestazione) != dati.shape[1]:
        problemi.append(
            "numero di celle nell'intestazione diverso dal numero di colonne "
            f"lette: {len(intestazione)} vs {dati.shape[1]}"
        )

    viste = set()
    duplicate = []
    for nome in intestazione:
        if nome in viste and nome not in duplicate:
            duplicate.append(nome)
        viste.add(nome)

    mancanti = [
        nome for nome in COLONNE_DATASET_SORGENTE if nome not in intestazione
    ]
    inattese = [
        nome for nome in intestazione if nome not in COLONNE_DATASET_SORGENTE
    ]

    if duplicate:
        problemi.append(f"colonne duplicate: {duplicate}")
    if mancanti:
        problemi.append(f"colonne mancanti: {mancanti}")
    if inattese:
        problemi.append(f"colonne inattese: {inattese}")
    if not (duplicate or mancanti or inattese) and (
        intestazione != COLONNE_DATASET_SORGENTE
    ):
        problemi.append("ordine delle colonne diverso dal catalogo IBM atteso")

    if problemi:
        dettaglio = "; ".join(problemi).replace("7,043", "7.043")
        raise ValueError(f"Dataset sorgente Telco non conforme: {dettaglio}.")

    return dati


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Pulizia e feature engineering, applicati riga per riga:

    1) valida le tre colonne numeriche; Total Charges può mancare soltanto dove
       Tenure Months = 0 (cliente che non ha ancora ricevuto la prima fattura)
    2) "No internet service" / "No phone service" → "No"
    3) tre feature nuove: Num Services Optional, Charge Per Tenure, Profilo Rischio

    Nessuno di questi passaggi apprende parametri dai dati: il risultato per un
    cliente dipende solo da quel cliente. Per questo il trasformatore può essere
    applicato indifferentemente prima o dopo lo split senza creare leakage.
    """

    def fit(self, X, y=None):
        self._valida_input(X)
        return self

    def transform(self, X):
        X = self._valida_input(X).copy()

        # 1) Numeriche: gli errori di estrazione non devono diventare zeri
        # silenziosi. L'unica eccezione di dominio è Total Charges assente per
        # un cliente con anzianità zero, che non ha ancora ricevuto una fattura.
        tenure = self._numerica_obbligatoria(X, "Tenure Months")
        mensile = self._numerica_obbligatoria(X, "Monthly Charges")
        self._rifiuta_booleani(X, "Total Charges")
        totale = pd.to_numeric(X["Total Charges"], errors="coerce")

        if pd.api.types.is_integer_dtype(tenure.dtype):
            tenure_fuori_int64 = tenure > np.iinfo(np.int64).max
        else:
            # Per i float, 2**63 è il primo valore non rappresentabile come
            # int64. Il massimo int64 arrotonda proprio a 2**63 in float64,
            # quindi il confronto deve usare il limite esclusivo.
            tenure_fuori_int64 = tenure >= float(2**63)
        if tenure_fuori_int64.any():
            indici = tenure_fuori_int64[tenure_fuori_int64].index.tolist()[:5]
            raise ValueError(
                "Tenure Months supera il massimo rappresentabile come int64; "
                f"righe non valide: {indici}."
            )
        if (tenure < 0).any() or not np.isclose(tenure % 1, 0).all():
            raise ValueError("Tenure Months deve contenere interi maggiori o uguali a zero.")
        if (mensile < 0).any():
            raise ValueError("Monthly Charges deve contenere valori maggiori o uguali a zero.")

        totale_non_finito = totale.notna() & ~np.isfinite(totale)
        if totale_non_finito.any():
            raise ValueError("Total Charges contiene valori non finiti.")
        totale_mancante_non_nuovo = totale.isna() & tenure.ne(0)
        if totale_mancante_non_nuovo.any():
            indici = totale_mancante_non_nuovo[totale_mancante_non_nuovo].index.tolist()[:5]
            raise ValueError(
                "Total Charges può mancare soltanto quando Tenure Months è 0; "
                f"righe non valide: {indici}."
            )
        if (totale.dropna() < 0).any():
            raise ValueError("Total Charges deve contenere valori maggiori o uguali a zero.")

        X["Tenure Months"] = tenure.astype(np.int64)
        X["Monthly Charges"] = mensile.astype(float)
        X["Total Charges"] = totale.astype(float)
        X.loc[tenure.eq(0), "Total Charges"] = 0.0

        # 2) Il terzo stato ridondante equivale a "No"
        for colonna, valore in COLONNE_DA_COLLASSARE.items():
            X[colonna] = X[colonna].replace(valore, "No")

        # 3) Feature engineering
        X["Num Services Optional"] = (X[SERVIZI_OPZIONALI] == "Yes").sum(axis=1)
        # Il denominatore passa a float prima del +1 per evitare overflow al
        # confine superiore di int64, che altrimenti diventerebbe negativo.
        denominatore_tenure = X["Tenure Months"].astype(float) + 1.0
        X["Charge Per Tenure"] = X["Monthly Charges"] / denominatore_tenure
        X["Profilo Rischio"] = (
            (X["Contract"] == "Month-to-month")
            & (X["Internet Service"] == "Fiber optic")
            & (X["Payment Method"] == "Electronic check")
        ).astype(int)

        return X

    @staticmethod
    def _numerica_obbligatoria(X, colonna):
        """Converte una colonna numerica e rifiuta valori mancanti o non finiti."""
        ChurnFeatureEngineer._rifiuta_booleani(X, colonna)
        valori = pd.to_numeric(X[colonna], errors="coerce")
        non_validi = valori.isna() | ~np.isfinite(valori)
        if non_validi.any():
            indici = non_validi[non_validi].index.tolist()[:5]
            raise ValueError(
                f"{colonna} contiene valori mancanti, testuali o non finiti; "
                f"righe non valide: {indici}."
            )
        return valori

    @staticmethod
    def _rifiuta_booleani(X, colonna):
        """Impedisce che True e False vengano interpretati come 1 e 0."""
        booleani = X[colonna].map(
            lambda valore: isinstance(valore, (bool, np.bool_))
        )
        if booleani.any():
            indici = booleani[booleani].index.tolist()[:5]
            raise ValueError(
                f"{colonna} non accetta valori booleani; "
                f"righe non valide: {indici}."
            )

    def get_feature_names_out(self, input_features=None):
        colonne = list(input_features) if input_features is not None else list(COLONNE_RAW_INPUT)
        nuove = ["Num Services Optional", "Charge Per Tenure"] + COLONNE_FLAG
        return np.array(colonne + [c for c in nuove if c not in colonne], dtype=object)

    @staticmethod
    def _valida_input(X):
        """
        Accetta un DataFrame, oppure un singolo cliente come dict o Series,
        oppure una lista di dict. Verifica che ci siano tutte le colonne raw e
        che le categoriche non siano mancanti, vuote o composte solo da spazi.
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = pd.DataFrame([X.to_dict()])
        elif isinstance(X, list):
            X = pd.DataFrame(X)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"Tipo di input non gestito: {type(X)}. "
                f"Attesi: DataFrame, dict, Series, list of dict."
            )

        colonne_mancanti = [c for c in COLONNE_RAW_INPUT if c not in X.columns]
        if colonne_mancanti:
            raise ValueError(
                f"Colonne raw mancanti: {colonne_mancanti}. "
                f"Sono richieste tutte le {len(COLONNE_RAW_INPUT)} colonne "
                f"del catalogo Telco (vedi COLONNE_RAW_INPUT)."
            )

        categoriche_non_valide = {}
        for colonna in COLONNE_CATEGORICHE:
            valori = X[colonna]
            vuoti = valori.isna() | valori.map(
                lambda valore: isinstance(valore, str) and not valore.strip()
            )
            if vuoti.any():
                categoriche_non_valide[colonna] = (
                    vuoti[vuoti].index.tolist()[:5]
                )

        if categoriche_non_valide:
            raise ValueError(
                "Le colonne categoriche non possono contenere valori "
                "mancanti, vuoti o composti soltanto da spazi; "
                "righe non valide per colonna: "
                f"{categoriche_non_valide}."
            )
        return X


# ---------------------------------------------------------------------------
# Preprocessore completo
# ---------------------------------------------------------------------------

def _nome_binaria(feature, category):
    """
    Nome della colonna in uscita per una binaria.

    Con `drop='first'` di ogni binaria sopravvive una sola colonna, quindi il
    nome originale resta univoco e leggibile ("Partner" invece di "Partner_Yes").
    Deve essere una funzione del modulo e non una lambda: `joblib` serializza lo
    stato dell'oggetto per riferimento al nome, e una lambda non ha un nome
    importabile.
    """
    return feature


def crea_preprocessore() -> Pipeline:
    """
    Costruisce il preprocessore completo: feature engineering, poi scaling e
    encoding gestiti da un ColumnTransformer.

    Tre scelte che contano nell'inferenza su dati nuovi:

    - `OneHotEncoder(handle_unknown='ignore')`: se in input arriva un metodo di
      pagamento o un tipo di contratto mai visto in addestramento, tutte le
      colonne di quel blocco valgono 0. È un comportamento documentato e identico
      a ogni chiamata, e resta rilevabile con `categorie_non_viste()`.
    - Sulle 12 binarie, `OneHotEncoder(categories=[["No", "Yes"]], drop="first")`
      fissa la codifica (No → 0, Yes → 1) invece di dedurla dai dati presenti al
      momento della chiamata; un valore mai visto ricade su 0, cioè su "No".
    - Lo scaler viene addestrato solo sui dati passati al `fit`, quindi dentro
      una cross validation impara le statistiche del solo fold di training.

    L'output è un DataFrame con nomi di colonna leggibili (28 feature).
    """
    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLONNE_NUMERICHE),
            ("bin", OneHotEncoder(
                categories=[["No", "Yes"]] * len(COLONNE_BINARIE),
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.int64,
                feature_name_combiner=_nome_binaria,
            ), COLONNE_BINARIE),
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.int64,
            ), COLONNE_ONE_HOT),
            ("flag", "passthrough", COLONNE_FLAG),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline([
        ("fe", ChurnFeatureEngineer()),
        ("enc", encoder),
    ])


def nomi_feature(pipeline) -> list:
    """Nomi delle 28 feature in uscita dal preprocessore, nell'ordine del modello."""
    return list(_preprocessore(pipeline).named_steps["enc"].get_feature_names_out())


# ---------------------------------------------------------------------------
# Diagnostica sui valori categorici mai visti
# ---------------------------------------------------------------------------

def categorie_non_viste(pipeline, clienti) -> dict:
    """
    Elenca, colonna per colonna, i valori categorici presenti nei clienti passati
    ma assenti dal training.

    L'encoder li gestisce in modo deterministico (blocco one-hot tutto a zero,
    binaria trattata come "No"), ma resta un'informazione da monitorare: se in
    produzione compare un contratto o un metodo di pagamento nuovo, il modello lo
    sta ignorando e va ri-addestrato.
    """
    prep = _preprocessore(pipeline)

    # Il confronto va fatto dopo il feature engineering: valori legittimi come
    # "No internet service" a quel punto sono già stati collassati su "No", ed
    # è quella la forma che l'encoder ha visto durante il fit.
    clienti = prep.named_steps["fe"].transform(clienti)
    encoder = prep.named_steps["enc"]

    categorie_viste = {}
    for nome_blocco, colonne in (("bin", COLONNE_BINARIE), ("cat", COLONNE_ONE_HOT)):
        trasformatore = encoder.named_transformers_[nome_blocco]
        for colonna, categorie in zip(colonne, trasformatore.categories_):
            categorie_viste[colonna] = set(categorie)

    ignote = {}
    for colonna, viste in categorie_viste.items():
        presenti = set(clienti[colonna].dropna().unique())
        nuove = sorted(presenti - viste, key=str)
        if nuove:
            ignote[colonna] = nuove
    return ignote


# ---------------------------------------------------------------------------
# Spiegazione locale della singola predizione
# ---------------------------------------------------------------------------

def registra_baseline(pipeline, X_train):
    """
    Salva nella pipeline la media delle feature sul training set.

    Serve alle spiegazioni locali: il contributo di una feature si misura
    rispetto al cliente medio, quindi la media va congelata insieme al modello.
    """
    trasformato = _preprocessore(pipeline).transform(X_train)
    pipeline.medie_train_ = np.asarray(trasformato.mean(axis=0), dtype=float)
    pipeline.nomi_feature_ = list(trasformato.columns)
    return pipeline


def _valida_cut_decili(cut_decili):
    """Restituisce nove cut-point finiti, crescenti e interni a (0,1)."""
    try:
        cut = np.asarray(cut_decili, dtype=float)
    except (TypeError, ValueError) as errore:
        raise ValueError(
            "I cut-point dei decili devono essere 9 numeri finiti."
        ) from errore

    if cut.shape != (9,) or not np.all(np.isfinite(cut)):
        raise ValueError("I cut-point dei decili devono essere 9 numeri finiti.")
    if not np.all(np.diff(cut) > 0) or cut[0] <= 0 or cut[-1] >= 1:
        raise ValueError(
            "I cut-point dei decili devono essere strettamente crescenti e in (0,1)."
        )
    return cut


def registra_parametri_operativi(pipeline, X_train, soglia, cut_decili):
    """Congela baseline, soglia e decili selezionati prima del test finale."""
    if not (0.0 < float(soglia) < 1.0):
        raise ValueError(f"Soglia fuori range (0,1): {soglia}")

    cut = _valida_cut_decili(cut_decili)

    registra_baseline(pipeline, X_train)
    pipeline.soglia_decisione_ = float(soglia)
    pipeline.decili_riferimento_ = cut.copy()
    return pipeline


def contributi_locali(pipeline, clienti) -> pd.DataFrame:
    """
    Contributo di ogni feature alla predizione di ogni cliente, in unità di
    log-odds: `coefficiente * (valore del cliente − media del training)`.

    Per un modello lineare questa quantità coincide esattamente con il valore
    SHAP della feature, quindi la spiegazione locale in inferenza non richiede
    la libreria shap. Contributo positivo = spinge verso l'abbandono.
    """
    if not hasattr(pipeline, "medie_train_"):
        raise AttributeError(
            "Baseline mancante: chiamare registra_baseline(pipeline, X_train) "
            "una volta dopo il fit, prima di serializzare la pipeline."
        )

    X_proc = _preprocessore(pipeline).transform(clienti)
    medie = pd.Series(
        np.asarray(pipeline.medie_train_, dtype=float),
        index=getattr(pipeline, "nomi_feature_", X_proc.columns),
    ).reindex(X_proc.columns)
    coefficienti = pd.Series(
        pipeline.named_steps["clf"].coef_[0], index=X_proc.columns
    )
    return (X_proc - medie) * coefficienti


def motivi_principali(pipeline, clienti, n: int = 3) -> pd.Series:
    """
    Per ogni cliente, i primi `n` contributi che ne alzano il punteggio di churn,
    espressi in una frase leggibile durante la revisione umana.

    Al nome della caratteristica viene aggiunta la direzione: "anzianità: sotto
    la media" descrive la posizione rispetto alla baseline del modello meglio di
    "anzianità" da sola. Non identifica una causa né un intervento efficace.
    """
    contributi = contributi_locali(pipeline, clienti)
    valori = _preprocessore(pipeline).transform(clienti)

    def descrivi(feature, contributo):
        etichetta = ETICHETTE_LEGGIBILI.get(feature, feature)
        valore = valori.loc[contributo.name, feature]
        if feature in COLONNE_NUMERICHE:
            direzione = "sopra la media" if valore > 0 else "sotto la media"
        else:
            direzione = "sì" if valore == 1 else "no"
        return f"{etichetta}: {direzione}"

    def riga_a_testo(riga):
        spinte = riga[riga > 0].sort_values(ascending=False).head(n)
        if spinte.empty:
            return "nessun fattore di rischio prevalente"
        return "; ".join(
            f"{descrivi(feature, riga)} (+{valore:.2f})"
            for feature, valore in spinte.items()
        )

    return contributi.apply(riga_a_testo, axis=1).rename("motivi")


# ---------------------------------------------------------------------------
# Funzione di inferenza
# ---------------------------------------------------------------------------

def predici_churn(
    pipeline,
    clienti,
    soglia: float | None = None,
    cut_decili=None,
    con_motivazioni: bool = False,
    avvisa_categorie_ignote: bool = True,
) -> pd.DataFrame:
    """
    Applica la pipeline serializzata a uno o più clienti e restituisce
    un DataFrame con probabilità, decisione binaria alla soglia scelta
    e decile di rischio.

    Parametri
    ---------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline addestrata, con step ('prep', preprocessore) e
        ('clf', LogisticRegression(...)).
    clienti : DataFrame | dict | Series | list of dict
        Uno o più clienti con lo schema di COLONNE_RAW_INPUT.
    soglia : float, opzionale
        Se omessa usa `pipeline.soglia_decisione_`; per artefatti precedenti
        usa `SOGLIA_RIFERIMENTO`. Passare un valore per simulare
        assunzioni economiche diverse senza modificare il modello.
    cut_decili : array-like di 9 soglie, opzionale
        Cut-point numerici finiti, strettamente crescenti e interni a (0,1).
        Se None (default) si usano quelli incorporati nella pipeline oppure
        DECILI_RIFERIMENTO. Passarne di nuovi se il modello viene ri-addestrato.
    con_motivazioni : bool, default False
        Se True aggiunge la colonna `motivi` con i tre fattori che più alzano
        la probabilità del singolo cliente.
    avvisa_categorie_ignote : bool, default True
        Se True segnala con un warning i valori categorici mai visti in
        addestramento, che l'encoder tratta come tutti-zero.

    Ritorna
    -------
    pd.DataFrame con colonne: proba, decisione (0/1), decile (1 = più a rischio)
    e, se richiesto, motivi. Ha lo stesso indice del DataFrame di input.

    Nota
    ----
    Il decile è assegnato rispetto a una distribuzione di riferimento FISSA, non
    rispetto ai soli clienti passati nella chiamata. Così un singolo cliente ad
    alto rischio ottiene correttamente il decile 1, e i decili sono confrontabili
    tra chiamate diverse (un cliente non "cambia decile" a seconda di chi gli sta
    accanto nel batch).
    """
    if soglia is None:
        soglia = getattr(pipeline, "soglia_decisione_", SOGLIA_RIFERIMENTO)
    if not (0.0 < soglia < 1.0):
        raise ValueError(f"Soglia fuori range (0,1): {soglia}")

    if cut_decili is None:
        cut = getattr(pipeline, "decili_riferimento_", DECILI_RIFERIMENTO)
    else:
        cut = cut_decili
    cut = _valida_cut_decili(cut)

    clienti = ChurnFeatureEngineer._valida_input(clienti)

    if avvisa_categorie_ignote:
        ignote = categorie_non_viste(pipeline, clienti)
        if ignote:
            warnings.warn(
                "Valori categorici mai visti in addestramento, trattati come "
                f"assenti dall'encoder: {ignote}. Il modello li sta ignorando: "
                "verificare l'estrazione dati e valutare un ri-addestramento.",
                stacklevel=2,
            )

    proba = pipeline.predict_proba(clienti)[:, 1]
    decisione = (proba >= soglia).astype(int)

    # Decili di rischio rispetto a una distribuzione di riferimento fissa.
    # np.digitize conta quanti cut-point stanno sotto la proba del cliente:
    # proba alta → molti cut sotto → decile basso (1 = più a rischio).
    decile = (10 - np.digitize(proba, cut)).clip(1, 10).astype(int)

    esiti = pd.DataFrame(
        {"proba": proba, "decisione": decisione, "decile": decile},
        index=clienti.index,
    )

    if con_motivazioni:
        esiti["motivi"] = motivi_principali(pipeline, clienti)

    return esiti


# ---------------------------------------------------------------------------
# Utilità interne
# ---------------------------------------------------------------------------

def _preprocessore(pipeline):
    """Estrae lo step di preprocessing, accettando anche il preprocessore da solo."""
    if hasattr(pipeline, "named_steps") and "prep" in pipeline.named_steps:
        return pipeline.named_steps["prep"]
    return pipeline
