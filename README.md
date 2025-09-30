# E-E91 — Simulazione, analisi e rilevamento di intercettazioni per il protocollo E91

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-yellow.svg)]()
[![Qiskit](https://img.shields.io/badge/qiskit-v1.2-lightgrey)]()

## Indice
- [E-E91 — Simulazione, analisi e rilevamento di intercettazioni per il protocollo E91](#e-e91--simulazione-analisi-e-rilevamento-di-intercettazioni-per-il-protocollo-e91)
  - [Indice](#indice)
- [Descrizione](#descrizione)
- [Caratteristiche principali](#caratteristiche-principali)
- [Struttura del repository](#struttura-del-repository)
- [Requisiti e installazione](#requisiti-e-installazione)
- [Esempi d'uso](#esempi-duso)
- [Features](#features)
- [Licenza](#licenza)
- [Riferimenti essenziali](#riferimenti-essenziali)

---

# Descrizione
Repository per la simulazione e l'analisi del protocollo **E-E91** (Enhanced-E91) con:

- Simulazioni in Qiskit.
- Implementazione di attacchi/intercettazioni configurabili,
- Pipeline di estrazione feature da statistiche CHSH e distribuzioni di anticorrelazione.
- Modelli per il rilevamento di `Eavesdropper`.

Obiettivo: fornire codice riproducibile per testare ipotesi e costruire un classificatore che distingua canali puliti da canali compromessi.

---

# Caratteristiche principali
- Simulazioni E91 (stati di Bell, misure, basi casuali).
- Simulazione di rumore e attacchi: in particolare, ci si è soffermati su un attacco tipico, ovvero *intercept-resend*.
- Calcolo statistico del valore CHSH su sample (distribuzioni, medie, varianze).
- Estrazione automatica di feature dalle colonne `CHSH samples statistics` e `Anticorrelation distribution`.
- Pipeline di training e valutazione per rilevamento `Eavesdropper` (scikit-learn/PyTorch).
- Notebook utilizzato nel corso del progetto.
---

# Struttura del repository
```
E-E91/
├─ README.md
├─ run.ps1              # per avviare main.py tramite ambiente virtuale
├─ run.sh
├─ bibliografia.bib     # bibliografia del progetto di tesi
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ package/
│  ├─ __init__.py
│  ├─ classifiers.py    # implementazioni dei wrapper 
│  ├─ E91.py            # codice per la simulazione dell'E91
│  ├─ EE91.py           # implementazione della componente intelligente E-E91
│  ├─ experiments.py    # esecuzione degli esperimenti
│  ├─ extracts.py       # processazione dei dataset
│  ├─ noise.py          # funzioni utili per la simulazione di rumore di canale
│  └─ utils.py          # funzioni utilitarie
├─ main.py              # script principale dove avviene la simulazione
├─ Eavesdropping detection in canali quantistici rumorosi tramite l'uso del machine learning.ipynb
├─ imbq_backends_properties/    # contiene le proprietà dei backend IBM Quantum
│  ├─ ibm_brisbane_properties.json
│  └─ ibm_sherbrooke_properties.json
└─ data/
   ├─ experiments/     
   │  ├─ E-E91_tests_fiber_noise_Ensemble.csv
   │  ├─ E-E91_tests_fiber_noise_Ensemble.json # per gestire il checkpoint del LHS
   │  ...
   │  └─ ...
   ├─ lhs/      
   │  ├─ lhs_parameters_seed_41
   │  └─ ...
   ├─ models/     
   │  ├─ imputer_SimpleImputer.pkl
   │  ├─ ...
   │  ├─ model_RandomForestClassifier.pkl
   │  └─ ...
   └─ training/
      ├─ test_set.csv
      ├─ test_set.json
      ├─ train_set.csv
      └─ train_set.json
```

---

# Requisiti e installazione
Si consiglia l'uso di un ambiente virtuale, per esempio:

```bash
# crea env (virtualenv)
python -m venv venv
source venv/bin/activate       # Linux
# venv\Scripts\Activate.ps1    # Windows

# installa dipendenze
pip install -r requirements.txt
```

Il file `requirements.txt` contiene le seguenti dipendenze:

```
numpy
scipy
pandas
matplotlib
seaborn
pylatexenc
scikit-learn
itertools
pathlib
joblib
torch
qiskit >=1.3, <2
qiskit_aer >=0.16
qiskit_ibm_runtime >=0.29 , <0.40
```

Per l'installazione delle reti di Hopfield moderne, è necessario eseguire:

```
set PYTHONUTF8=1 # Su Windows usare: $env:PYTHONUTF8=1
pip install "git+https://github.com/ml-jku/hopfield-layers"
```
---

# Esempi d'uso
Per la simulazione è sufficiente eseguire `main.py`.

# Features
I dati sono estratti per il mio lavoro provengono da due colonne chiave:  

- `CHSH samples statistics` (es. un dizionario con `mode`, `median`, `mean`, `variance`, `skewness`, `kurtosis`).
- `Anticorrelation distribution` (es. quantità di anti-correlazione tra i qubit di Alice e Bob rispetto a una determinata base del test di Bell).

Un esempio di CSV (riga singola):
```csv
Eavesdropper,"(X, W) anticorrelation","(X, V) anticorrelation","(Z, W) anticorrelation","(Z, V) anticorrelation",CHSH samples mode (mode),CHSH samples median (mode),CHSH samples mean (mode),CHSH samples skewness (mode),CHSH samples mode (median),CHSH samples median (median),CHSH samples mean (median),CHSH samples skewness (median),CHSH samples mode (mean),CHSH samples median (mean),CHSH samples mean (mean),CHSH samples skewness (mean)
1,0.559322033898305,0.3725490196078431,0.54,0.5517241379310345,-1.0,-1.0,-0.11864406779661017,-0.5272196186749782,-1.0,-1.0,-0.09172413793103448,0.18426352218274328,-0.5,-0.5,-0.011797595718591361,0.020070871347532444
```

---

# Licenza
Questo progetto è rilasciato sotto licenza **MIT**. Aggiungi un file `LICENSE` con il testo MIT se desideri questa licenza.

---

# Riferimenti essenziali

- **Ekert, A. K. (1991)** — *Quantum cryptography based on Bell’s theorem*. *Physical Review Letters* 67, 661–663.

- **Bell, J. S. (1964)** — *On the Einstein Podolsky Rosen paradox*. *Physics* 1, 195–200.  

- **Clauser, J. F., Horne, M. A., Shimony, A., & Holt, R. A. (1969)** — *Proposed experiment to test local hidden-variable theories*. *Physical Review Letters* 23, 880–884.

- **Nielsen, M. A. & Chuang, I. L. (2000)** — *Quantum Computation and Quantum Information*. Cambridge University Press.  
Testo di riferimento su circuiti quantistici, misure e formalismo.

- **Gisin, N., Ribordy, G., Tittel, W., & Zbinden, H. (2002)** — *Quantum cryptography*. *Reviews of Modern Physics* 74, 145–195.  
Revisione completa sui protocolli QKD, aspetti pratici e implementazioni sperimentali.

- **Qiskit Documentation (v1.2)** — *IBM / Qiskit docs*.  
Documentazione delle API e dei simulatori usati.

- **Pedregosa, F. et al. (2011)** — *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research* 12, 2825–2830.  

- **Paszke, A. et al. (2019)** — *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. *NeurIPS Workshop Proceedings*.

> **Nota:** la bibliografia completa è disponibile nel file `bibliografia.bib` del repository per chi volesse approfondire ulteriormente.

