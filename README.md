# Telco Customer Churn: dalla selezione del modello all'analisi dei punteggi

**Italiano** | [English](README_EN.md)

Progetto end-to-end di machine learning per stimare il churn dei clienti e
trasformare le probabilità in una tabella ordinata di punteggi destinata alla
revisione tecnica. L'analisi non si limita all'accuratezza: confronta le
strategie per lo sbilanciamento delle classi, seleziona la soglia decisionale
con pesi ipotetici degli errori, definisce i decili di rischio sulla
distribuzione di training e mantiene intatto il set di test finale fino al
blocco di tutte le decisioni di modellazione.

Il notebook è scritto in italiano. La versione inglese di questa pagina è
disponibile in [README_EN.md](README_EN.md).

Questo progetto di portfolio è stato sviluppato durante il mio percorso di
formazione come Machine Learning/AI Engineer. Dimostra un prototipo
riproducibile, non un sistema in produzione né un incarico professionale per un
cliente.

## Che cosa dimostra il progetto

- Un'unica pipeline scikit-learn per feature engineering, scaling, encoding e
  classificazione, con gestione deterministica delle categorie mai viste.
- Un confronto numerico tra nessuna correzione, pesi di classe e SMOTE su
  Regressione Logistica, Random Forest e XGBoost.
- Predizioni out-of-fold a cinque fold per confrontare i modelli, scegliere la
  soglia e definire i decili di rischio di riferimento.
- Una valutazione finale sul set di test eseguita una sola volta, senza
  ulteriore ottimizzazione.
- Interpretabilità globale e locale tramite coefficienti della regressione
  logistica, SHAP per XGBoost e motivazioni associate a ogni previsione.
- Un modello serializzato che incorpora soglia, limiti dei decili e baseline
  esplicativa nello stesso artefatto.
- Validazione rigorosa dei campi numerici e dei valori categorici mancanti o
  vuoti, più un test end-to-end che copre input Excel, caricamento reale del
  modello, inferenza e output CSV in un nuovo processo Python.

## Risultati finali sul set di test

La Regressione Logistica selezionata usa una soglia di `0.4265`, scelta dalle
predizioni out-of-fold del training con un rapporto di costo tra falsi negativi
e falsi positivi pari a 5:1.

| Modello | Soglia | ROC-AUC | PR-AUC | Precision | Recall | F1 | FN | FP | Unità di costo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Regressione Logistica, selezionata | 0.4265 | 0.854 | 0.675 | 0.490 | 0.837 | 0.618 | 61 | 326 | 631 |
| XGBoost, benchmark bloccato | 0.3775 | 0.854 | 0.670 | 0.481 | 0.877 | 0.621 | 46 | 354 | 584 |

XGBoost ha ottenuto il costo più basso con il rapporto ipotetico scelto. La
Regressione Logistica è stata mantenuta perché la sua ROC-AUC in validazione
incrociata (`0.8617 ± 0.0124`) rientrava nell'intervallo di un errore standard
di XGBoost (`0.8641 ± 0.0098`) e offriva un modello più compatto e direttamente
interpretabile. È una scelta di modellazione, non l'affermazione che la
regressione logistica sia superiore in ogni scenario commerciale. Il notebook
riporta l'intera analisi di sensitività.

## Contenuto del repository

```text
.
├── README.md                            # pagina principale in italiano
├── README_EN.md                         # versione inglese
├── telco_customer_churn_analysis.ipynb  # analisi end-to-end eseguita
├── churn_inference.py                   # preprocessing e contratto di inferenza
├── pipeline_churn.joblib                # pipeline di Regressione Logistica addestrata
├── predict_churn.py                     # esempio da Excel a CSV da riga di comando
├── MODEL_CARD.md                        # uso previsto, metriche e limiti
├── DATASET.md                           # fonte e contratto dei dati
├── examples/
│   └── customers_sample.xlsx            # esempio sintetico per la sola inferenza
└── tests/
    ├── test_churn_inference.py
    └── test_end_to_end.py
```

## Dataset

Il dataset di training non viene redistribuito in questo repository. Scaricare
`Telco_customer_churn.xlsx` dal
[dataset IBM Telco Customer Churn su Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
e collocarlo accanto al notebook. In alternativa, impostare
`TELCO_CHURN_DATA` con il percorso del file. Su Kaggle il notebook cerca anche
tra i dataset collegati come input.

Per lo schema previsto e le note sulla fonte, consultare
[DATASET.md](DATASET.md).

## Eseguire l'inferenza

Usare CPython 3.12. Il modello è serializzato con scikit-learn 1.9.0, quindi
l'ambiente con versioni bloccate fa parte del contratto dell'artefatto.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python predict_churn.py
```

Il comando predefinito legge l'esempio Excel sintetico e scrive
`outputs/churn_scores_above_threshold.csv`. È possibile specificare percorsi
diversi:

```powershell
.venv\Scripts\python predict_churn.py `
  --input path\to\customers.xlsx `
  --model pipeline_churn.joblib `
  --output outputs\churn_scores_above_threshold.csv
```

Non caricare mai un file `.joblib` non attendibile: la serializzazione dei
modelli Python può eseguire codice durante il caricamento.

## Riprodurre l'analisi

```powershell
.venv\Scripts\python -m pip install -r requirements-notebook.txt
.venv\Scripts\python -m ipykernel install --user --name telco-churn
```

Aprire `telco_customer_churn_analysis.ipynb`, selezionare il kernel
`telco-churn` ed eseguire tutte le celle. Il notebook usa un singolo processo
per mantenere deterministica l'esecuzione anche in ambienti con risorse
limitate. Su Kaggle è invece possibile collegare il dataset ed eseguire
direttamente il notebook.

## Eseguire i test

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

L'ambiente completo con versioni bloccate usato dall'integrazione continua è
disponibile in `requirements-lock.txt`.

## Perimetro

Questo è un prototipo di supporto decisionale da portfolio, costruito su un
dataset dimostrativo IBM. Non è un sistema di retention distribuito e non
dimostra relazioni causali. Prima di un impiego operativo sarebbero necessari
dati aziendali aggiornati, un modello dei costi validato, una valutazione delle
prestazioni per sottogruppo, monitoraggio e una politica di riaddestramento.
