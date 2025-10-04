# E-E91 — Simulation, analysis and eavesdropper detection for the E91 protocol

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-yellow.svg)]()
[![Qiskit](https://img.shields.io/badge/qiskit-v1.3+-lightgrey)]()

## Table of Contents
- [E-E91 — Simulation, analysis and eavesdropper detection for the E91 protocol](#e-e91--simulation-analysis-and-eavesdropper-detection-for-the-e91-protocol)
  - [Table of Contents](#table-of-contents)
- [Description](#description)
- [Main features](#main-features)
- [Repository structure](#repository-structure)
- [Requirements and installation](#requirements-and-installation)
- [Usage examples](#usage-examples)
- [Features (data)](#features-data)
- [License](#license)
- [Essential references](#essential-references)

---

# Description
Repository for the simulation and analysis of the **E-E91** (Enhanced-E91) protocol with:

- Simulations in Qiskit.
- Configurable implementation of attacks/eavesdropping.
- Pipeline for feature extraction from CHSH statistics and anticorrelation distributions.
- Models for `Eavesdropper` detection.

This project attempts to add an intelligent component to the E91 quantum key distribution (QKD) protocol, first introduced by Artur Ekert in 1991.
Specifically, a supervised approach was used, leveraging an ensemble of six different classifiers of different nature, making a naive assumption about the weight of each vote given by the individual models.
The enhanced protocol, called E-E91, has the ability to halt early, thus potentially saving computational resources.

---

# Main features
- E91 simulations (Bell states, measurements, random bases).
- Noise and attack simulation: in particular, I focused on a typical attack, i.e. *intercept-resend*.
- Statistical computation of the CHSH value on samples (distributions, means, variances).
- Automatic feature extraction from the `CHSH samples statistics` and `Anticorrelation distribution` fields.
- Training and evaluation pipeline for `Eavesdropper` detection (scikit-learn/PyTorch).
- Notebook used during the project.
---

# Repository structure
```
E-E91/
├─ README.md
├─ run.ps1              # helper to run main.py using the virtual environment
├─ run.sh
├─ bibliografia.bib     # bibliography used for the thesis
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ package/
│  ├─ __init__.py
│  ├─ classifiers.py    # wrapper implementations
│  ├─ E91.py            # E91 simulation code
│  ├─ EE91.py           # implementation of the E-E91 intelligent component
│  ├─ experiments.py    # experiment runner
│  ├─ extracts.py       # dataset processing
│  ├─ noise.py          # helper functions for channel noise simulation
│  └─ utils.py          # utility functions
├─ main.py              # main script where simulations are executed
├─ Eavesdropping detection in noisy quantum channels using machine learning.ipynb
├─ imbq_backends_properties/    # contains IBM Quantum backend properties
│  ├─ ibm_brisbane_properties.json
│  └─ ibm_sherbrooke_properties.json
└─ data/
   ├─ experiments/     
   │  ├─ E-E91_tests_fiber_noise_Ensemble.csv
   │  ├─ E-E91_tests_fiber_noise_Ensemble.json # checkpoint for LHS management
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

# Requirements and installation
A virtual environment is recommended, for example:

```bash
# create env (virtualenv)
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\Activate.ps1    # Windows PowerShell

# install dependencies
pip install -r requirements.txt
```

The `requirements.txt` file contains the following dependencies:

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

To install modern Hopfield networks, run:

```
set PYTHONUTF8=1 # On Windows use: $env:PYTHONUTF8=1
pip install "git+https://github.com/ml-jku/hopfield-layers"
```
---

# Usage examples
To run a simulation simply execute `main.py`.

# Features (data)
The data extracted for this work come from two key columns:

- `CHSH samples statistics` (e.g. a dictionary with `mode`, `median`, `mean`, `variance`, `skewness`, `kurtosis`).
- `Anticorrelation distribution` (e.g. the amount of anticorrelation between Alice's and Bob's qubits with respect to a given Bell test basis).

An example CSV (single row):
```csv
Eavesdropper,"(X, W) anticorrelation","(X, V) anticorrelation","(Z, W) anticorrelation","(Z, V) anticorrelation",CHSH samples mode (mode),CHSH samples median (mode),CHSH samples mean (mode),CHSH samples skewness (mode),CHSH samples mode (median),CHSH samples median (median),CHSH samples mean (median),CHSH samples skewness (median),CHSH samples mode (mean),CHSH samples median (mean),CHSH samples mean (mean),CHSH samples skewness (mean)
1,0.559322033898305,0.3725490196078431,0.54,0.5517241379310345,-1.0,-1.0,-0.11864406779661017,-0.5272196186749782,-1.0,-1.0,-0.09172413793103448,0.18426352218274328,-0.5,-0.5,-0.011797595718591361,0.020070871347532444
```

---

# License
This project is released under the **MIT** license. Add a `LICENSE` file with the MIT text if you choose this license.

---

# Essential references

- **Ekert, A. K. (1991)** — *Quantum cryptography based on Bell’s theorem*. *Physical Review Letters* 67, 661–663.

- **Bell, J. S. (1964)** — *On the Einstein Podolsky Rosen paradox*. *Physics* 1, 195–200.  

- **Clauser, J. F., Horne, M. A., Shimony, A., & Holt, R. A. (1969)** — *Proposed experiment to test local hidden-variable theories*. *Physical Review Letters* 23, 880–884.

- **Nielsen, M. A. & Chuang, I. L. (2000)** — *Quantum Computation and Quantum Information*. Cambridge University Press.  
Reference textbook on quantum circuits, measurements and formalism.

- **Gisin, N., Ribordy, G., Tittel, W., & Zbinden, H. (2002)** — *Quantum cryptography*. *Reviews of Modern Physics* 74, 145–195.  
Comprehensive review on QKD protocols, practical aspects and experimental implementations.

- **Qiskit Documentation (v1.2)** — *IBM / Qiskit docs*.  
Documentation for the APIs and simulators used.

- **Pedregosa, F. et al. (2011)** — *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research* 12, 2825–2830.  

- **Paszke, A. et al. (2019)** — *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. *NeurIPS Workshop Proceedings*.

> **Note:** the complete bibliography is available in the `bibliografia.bib` file in the repository for readers who want to go deeper.
