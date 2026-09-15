"""
Test automatici del modulo di inferenza.

Coprono i punti in cui un preprocessing sbagliato produrrebbe un errore
silenzioso: categorie mai viste, dipendenza dal contenuto del batch, schema
di input incompleto, coerenza tra spiegazione locale e predizione.

Esecuzione dalla radice del repository:  python -m pytest tests/test_churn_inference.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_inference import (
    COLONNE_DATASET_SORGENTE,
    COLONNE_NUMERICHE,
    COLONNE_RAW_INPUT,
    RIGHE_DATASET_SORGENTE,
    ChurnFeatureEngineer,
    categorie_non_viste,
    contributi_locali,
    crea_preprocessore,
    motivi_principali,
    nomi_feature,
    predici_churn,
    registra_baseline,
    registra_parametri_operativi,
    valida_dataset_sorgente,
)

CONTRATTI = ["Month-to-month", "One year", "Two year"]
CONNESSIONI = ["DSL", "Fiber optic", "No"]
PAGAMENTI = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]


def clienti_finti(n=200, seed=0):
    """Dataset sintetico che contiene tutte le categorie previste dallo schema."""
    rng = np.random.default_rng(seed)
    dati = {
        "Tenure Months": rng.integers(0, 73, n),
        "Monthly Charges": rng.uniform(18.0, 120.0, n).round(2),
        "Contract": rng.choice(CONTRATTI, n),
        "Internet Service": rng.choice(CONNESSIONI, n),
        "Payment Method": rng.choice(PAGAMENTI, n),
    }
    for colonna in COLONNE_RAW_INPUT:
        if colonna not in dati and colonna != "Total Charges":
            dati[colonna] = rng.choice(["Yes", "No"], n)

    df = pd.DataFrame(dati)
    df["Total Charges"] = (df["Monthly Charges"] * df["Tenure Months"]).round(2)
    # I clienti a tenure 0 arrivano dal gestionale senza spesa totale
    df.loc[df["Tenure Months"] == 0, "Total Charges"] = np.nan
    return df[COLONNE_RAW_INPUT]


def dataset_sorgente_finto():
    """Replica soltanto forma e intestazione del file IBM canonico."""
    return pd.DataFrame(
        index=range(RIGHE_DATASET_SORGENTE),
        columns=COLONNE_DATASET_SORGENTE,
    )


@pytest.fixture(scope="module")
def dati():
    X = clienti_finti()
    # Target correlato al contratto, così i coefficienti non sono degeneri
    rischio = (X["Contract"] == "Month-to-month").astype(int)
    y = pd.Series((rischio + np.random.default_rng(1).random(len(X)) > 1.0).astype(int))
    return X, y


@pytest.fixture(scope="module")
def pipeline(dati):
    X, y = dati
    modello = Pipeline([
        ("prep", crea_preprocessore()),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                   max_iter=1000, random_state=42)),
    ])
    modello.fit(X, y)
    return registra_baseline(modello, X)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_dataset_sorgente_canonico_viene_accettato():
    dati = dataset_sorgente_finto()
    assert valida_dataset_sorgente(dati) is dati


@pytest.mark.parametrize(
    "caso,messaggio",
    [
        ("riga_mancante", "righe attese: 7.043"),
        ("colonna_mancante", "colonne mancanti"),
        ("colonna_rinominata", "colonne mancanti"),
        ("intestazione_duplicata", "colonne duplicate"),
    ],
)
def test_dataset_sorgente_non_canonico_viene_rifiutato(caso, messaggio):
    dati = dataset_sorgente_finto()
    intestazione = list(COLONNE_DATASET_SORGENTE)

    if caso == "riga_mancante":
        dati = dati.iloc[:-1]
    elif caso == "colonna_mancante":
        dati = dati.drop(columns=["Churn Reason"])
        intestazione = list(dati.columns)
    elif caso == "colonna_rinominata":
        dati = dati.rename(columns={"Churn Reason": "Churn Cause"})
        intestazione = list(dati.columns)
    elif caso == "intestazione_duplicata":
        intestazione[-1] = intestazione[0]

    with pytest.raises(ValueError, match=messaggio):
        valida_dataset_sorgente(dati, intestazione_originale=intestazione)

def test_total_charges_azzerata_per_i_clienti_nuovi():
    X = clienti_finti()
    nuovi = X["Tenure Months"] == 0
    assert nuovi.any(), "il dataset sintetico deve contenere clienti a tenure 0"

    risultato = ChurnFeatureEngineer().fit_transform(X)

    assert risultato["Total Charges"].isna().sum() == 0
    assert (risultato.loc[nuovi, "Total Charges"] == 0.0).all()


def test_terzo_stato_collassato_su_no():
    X = clienti_finti(n=10)
    X.loc[X.index[:5], "Online Security"] = "No internet service"
    X.loc[X.index[:3], "Multiple Lines"] = "No phone service"

    risultato = ChurnFeatureEngineer().fit_transform(X)

    assert set(risultato["Online Security"].unique()) <= {"No", "Yes"}
    assert set(risultato["Multiple Lines"].unique()) <= {"No", "Yes"}


def test_feature_costruite():
    X = clienti_finti(n=1)
    X.loc[X.index[0], list(X.columns)] = X.iloc[0]
    X.iloc[0, X.columns.get_loc("Tenure Months")] = 3
    X.iloc[0, X.columns.get_loc("Monthly Charges")] = 80.0
    X.iloc[0, X.columns.get_loc("Contract")] = "Month-to-month"
    X.iloc[0, X.columns.get_loc("Internet Service")] = "Fiber optic"
    X.iloc[0, X.columns.get_loc("Payment Method")] = "Electronic check"
    for servizio in ["Multiple Lines", "Online Security", "Online Backup",
                     "Device Protection", "Tech Support", "Streaming TV",
                     "Streaming Movies"]:
        X.iloc[0, X.columns.get_loc(servizio)] = "No"
    X.iloc[0, X.columns.get_loc("Online Security")] = "Yes"
    X.iloc[0, X.columns.get_loc("Tech Support")] = "Yes"

    risultato = ChurnFeatureEngineer().fit_transform(X).iloc[0]

    assert risultato["Num Services Optional"] == 2
    assert risultato["Charge Per Tenure"] == pytest.approx(80.0 / 4)
    assert risultato["Profilo Rischio"] == 1


def test_input_incompleto_solleva_errore():
    X = clienti_finti(n=5).drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        ChurnFeatureEngineer().fit_transform(X)


@pytest.mark.parametrize("colonna", ["Senior Citizen", "Contract"])
@pytest.mark.parametrize(
    "valore",
    [None, np.nan, pd.NA, "", "   "],
    ids=["none", "nan", "pd-na", "vuota", "soli-spazi"],
)
def test_categoriche_mancanti_o_vuote_solleva_errore(
    dati, pipeline, colonna, valore
):
    X, _ = dati
    cliente = X.iloc[[0]].copy()
    cliente[colonna] = cliente[colonna].astype(object)
    cliente.loc[cliente.index[0], colonna] = valore

    with pytest.raises(ValueError, match=colonna):
        predici_churn(pipeline, cliente)


@pytest.mark.parametrize(
    "colonna,valore,messaggio",
    [
        ("Tenure Months", "dodici", "Tenure Months"),
        ("Tenure Months", -1, "Tenure Months"),
        ("Tenure Months", 1.5, "Tenure Months"),
        ("Monthly Charges", "non disponibile", "Monthly Charges"),
        ("Monthly Charges", -10, "Monthly Charges"),
        ("Total Charges", -1, "Total Charges"),
    ],
)
def test_valori_numerici_non_validi_solleva_errore(colonna, valore, messaggio):
    X = clienti_finti(n=5)
    X[colonna] = X[colonna].astype(object)
    X.loc[X.index[0], colonna] = valore

    with pytest.raises(ValueError, match=messaggio):
        ChurnFeatureEngineer().fit_transform(X)


@pytest.mark.parametrize("valore", [1e20, np.iinfo(np.int64).max + 1])
def test_tenure_fuori_range_int64_viene_rifiutata(dati, pipeline, valore):
    X, _ = dati
    cliente = X.iloc[[0]].copy()
    cliente["Tenure Months"] = cliente["Tenure Months"].astype(object)
    cliente.loc[cliente.index[0], "Tenure Months"] = valore

    with pytest.raises(ValueError, match="Tenure Months"):
        predici_churn(pipeline, cliente)


def test_tenure_massima_int64_non_trabocca_nella_feature_derivata():
    cliente = clienti_finti(n=1)
    cliente["Tenure Months"] = cliente["Tenure Months"].astype(object)
    cliente.loc[cliente.index[0], "Tenure Months"] = np.iinfo(np.int64).max

    trasformato = ChurnFeatureEngineer().fit_transform(cliente)

    assert trasformato["Tenure Months"].iloc[0] == np.iinfo(np.int64).max
    assert np.isfinite(trasformato["Charge Per Tenure"].iloc[0])
    assert trasformato["Charge Per Tenure"].iloc[0] >= 0


@pytest.mark.parametrize("colonna", ["Tenure Months", "Monthly Charges", "Total Charges"])
@pytest.mark.parametrize(
    "valore",
    [True, False, np.bool_(True), np.bool_(False)],
    ids=["true", "false", "numpy-true", "numpy-false"],
)
def test_booleani_non_sono_accettati_come_numeri(
    dati, pipeline, colonna, valore
):
    X, _ = dati
    cliente = X.iloc[[0]].copy()
    cliente[colonna] = cliente[colonna].astype(object)
    cliente.loc[cliente.index[0], colonna] = valore

    with pytest.raises(ValueError, match=colonna):
        predici_churn(pipeline, cliente)


def test_total_charges_mancante_con_tenure_positiva_solleva_errore():
    X = clienti_finti(n=5)
    X.loc[X.index[0], "Tenure Months"] = 12
    X["Total Charges"] = X["Total Charges"].astype(object)
    X.loc[X.index[0], "Total Charges"] = " "

    with pytest.raises(ValueError, match="Total Charges"):
        ChurnFeatureEngineer().fit_transform(X)


# ---------------------------------------------------------------------------
# Preprocessore
# ---------------------------------------------------------------------------

def test_schema_di_uscita(pipeline):
    nomi = nomi_feature(pipeline)
    assert len(nomi) == 28
    assert len(set(nomi)) == 28, "nomi di feature duplicati"
    assert "Contract_Two year" in nomi
    assert "Payment Method_Electronic check" in nomi
    assert "Profilo Rischio" in nomi


def test_scaling_appreso_solo_sul_training(dati, pipeline):
    X, _ = dati
    trasformato = pipeline.named_steps["prep"].transform(X)
    medie = trasformato[COLONNE_NUMERICHE].mean().abs()
    assert (medie < 1e-9).all(), "sul training la media delle numeriche deve essere 0"


def test_categoria_mai_vista_azzera_il_blocco(dati, pipeline):
    X, _ = dati
    cliente = X.iloc[[0]].copy()
    cliente["Payment Method"] = "Crypto wallet"

    trasformato = pipeline.named_steps["prep"].transform(cliente)
    blocco = [c for c in trasformato.columns if c.startswith("Payment Method_")]

    assert (trasformato[blocco].to_numpy() == 0).all()
    assert categorie_non_viste(pipeline, cliente) == {"Payment Method": ["Crypto wallet"]}


def test_nessun_falso_allarme_sui_dati_di_training(dati, pipeline):
    X, _ = dati
    assert categorie_non_viste(pipeline, X) == {}


def test_nessun_falso_allarme_sul_terzo_stato(dati, pipeline):
    """
    "No internet service" è un valore legittimo del catalogo: il feature
    engineering lo collassa su "No" prima dell'encoder, quindi non va segnalato
    come categoria sconosciuta.
    """
    X, _ = dati
    clienti = X.iloc[:5].copy()
    clienti["Online Security"] = "No internet service"
    clienti["Multiple Lines"] = "No phone service"

    assert categorie_non_viste(pipeline, clienti) == {}


def test_predizione_indipendente_dal_resto_del_batch(dati, pipeline):
    """Il difetto che pd.get_dummies() introdurrebbe: colonne dedotte dal batch."""
    X, _ = dati
    in_batch = pipeline.predict_proba(X)[:, 1]
    da_soli = np.array([pipeline.predict_proba(X.iloc[[i]])[:, 1][0] for i in range(20)])
    assert np.abs(da_soli - in_batch[:20]).max() < 1e-12


def test_batch_di_soli_contratti_mensili(dati, pipeline):
    """Un batch che non contiene tutte le categorie non deve cambiare lo schema."""
    X, _ = dati
    mensili = X[X["Contract"] == "Month-to-month"]
    assert len(mensili) > 0

    trasformato = pipeline.named_steps["prep"].transform(mensili)
    assert list(trasformato.columns) == nomi_feature(pipeline)
    assert (trasformato["Contract_Two year"] == 0).all()


# ---------------------------------------------------------------------------
# Inferenza
# ---------------------------------------------------------------------------

def test_formati_di_input_equivalenti(dati, pipeline):
    X, _ = dati
    riga = X.iloc[0]

    da_dataframe = predici_churn(pipeline, X.iloc[[0]])
    da_dict = predici_churn(pipeline, riga.to_dict())
    da_series = predici_churn(pipeline, riga)
    da_lista = predici_churn(pipeline, [riga.to_dict()])

    for esito in (da_dict, da_series, da_lista):
        assert esito["proba"].iloc[0] == pytest.approx(da_dataframe["proba"].iloc[0])


def test_indice_preservato(dati, pipeline):
    X, _ = dati
    campione = X.iloc[[3, 7, 11]]
    esiti = predici_churn(pipeline, campione)
    assert list(esiti.index) == list(campione.index)


def test_decile_coerente_con_la_probabilita(dati, pipeline):
    X, _ = dati
    esiti = predici_churn(pipeline, X)
    ordinati = esiti.sort_values("proba", ascending=False)
    # Probabilità decrescente implica decile non decrescente (1 = più a rischio)
    assert ordinati["decile"].is_monotonic_increasing
    assert esiti["decile"].between(1, 10).all()


def test_decisione_segue_la_soglia(dati, pipeline):
    X, _ = dati
    esiti = predici_churn(pipeline, X, soglia=0.4)
    assert (esiti["decisione"] == (esiti["proba"] >= 0.4).astype(int)).all()


def test_soglia_fuori_range(dati, pipeline):
    X, _ = dati
    with pytest.raises(ValueError, match="Soglia"):
        predici_churn(pipeline, X.iloc[[0]], soglia=1.5)


@pytest.mark.parametrize(
    "cut",
    [
        np.linspace(0.1, 0.8, 8),
        np.array([0.1, 0.2, 0.3, 0.4, np.nan, 0.6, 0.7, 0.8, 0.9]),
        np.linspace(0.0, 1.0, 9),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0.9]),
    ],
    ids=["otto-cut", "nan", "estremi-zero-uno", "non-crescenti"],
)
def test_override_cut_decili_non_validi_viene_rifiutato(dati, pipeline, cut):
    X, _ = dati

    with pytest.raises(ValueError, match="cut-point"):
        predici_churn(pipeline, X.iloc[[0]], cut_decili=cut)


def test_override_cut_decili_viene_validato_prima_della_predizione(
    dati, pipeline, monkeypatch
):
    X, _ = dati

    def predizione_non_consentita(*args, **kwargs):
        raise AssertionError("predict_proba non deve essere chiamato")

    monkeypatch.setattr(pipeline, "predict_proba", predizione_non_consentita)

    with pytest.raises(ValueError, match="cut-point"):
        predici_churn(
            pipeline,
            X.iloc[[0]],
            cut_decili=np.linspace(0.1, 0.8, 8),
        )


def test_avviso_su_categoria_mai_vista(dati, pipeline):
    X, _ = dati
    cliente = X.iloc[[0]].copy()
    cliente["Contract"] = "Contratto sperimentale"

    with pytest.warns(UserWarning, match="mai visti"):
        predici_churn(pipeline, cliente)


# ---------------------------------------------------------------------------
# Spiegazione locale
# ---------------------------------------------------------------------------

def test_contributi_sommano_alla_predizione(dati, pipeline):
    X, _ = dati
    campione = X.iloc[:15]

    contributi = contributi_locali(pipeline, campione)
    clf = pipeline.named_steps["clf"]
    base = clf.intercept_[0] + float(np.dot(clf.coef_[0], pipeline.medie_train_))

    logit = base + contributi.sum(axis=1).to_numpy()
    atteso = pipeline.decision_function(campione)
    assert np.abs(logit - atteso).max() < 1e-9


def test_motivi_presenti_e_leggibili(dati, pipeline):
    X, _ = dati
    motivi = motivi_principali(pipeline, X.iloc[:5], n=3)
    assert len(motivi) == 5
    assert all(isinstance(m, str) and m for m in motivi)


def test_spiegazione_richiede_la_baseline(dati):
    X, y = dati
    senza_baseline = Pipeline([
        ("prep", crea_preprocessore()),
        ("clf", LogisticRegression(solver="liblinear", random_state=42)),
    ]).fit(X, y)

    with pytest.raises(AttributeError, match="Baseline"):
        contributi_locali(senza_baseline, X.iloc[[0]])


def test_parametri_operativi_congelati_nella_pipeline(dati, pipeline):
    X, _ = dati
    cut = np.linspace(0.1, 0.9, 9)
    registra_parametri_operativi(pipeline, X, soglia=0.42, cut_decili=cut)

    assert pipeline.soglia_decisione_ == pytest.approx(0.42)
    np.testing.assert_allclose(pipeline.decili_riferimento_, cut)
    assert isinstance(pipeline.medie_train_, np.ndarray)
    assert pipeline.nomi_feature_ == nomi_feature(pipeline)

    esiti_default = predici_churn(pipeline, X.iloc[:10])
    esiti_espliciti = predici_churn(
        pipeline, X.iloc[:10], soglia=0.42, cut_decili=cut
    )
    pd.testing.assert_frame_equal(esiti_default, esiti_espliciti)


@pytest.mark.parametrize(
    "soglia,cut",
    [
        (0.0, np.linspace(0.1, 0.9, 9)),
        (0.4, [0.1] * 9),
        (0.4, np.linspace(0.1, 0.8, 8)),
    ],
)
def test_parametri_operativi_non_validi_solleva_errore(dati, pipeline, soglia, cut):
    X, _ = dati
    with pytest.raises(ValueError):
        registra_parametri_operativi(pipeline, X, soglia=soglia, cut_decili=cut)
