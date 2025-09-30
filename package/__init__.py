from .E91 import E91 # senza il punto "." si esegue un import assoluto, che cerca un modulo E91_Protocol a livello globale (non nel package): se E91_Protocol.py importa qualcosa da utils.py e utils.py importa E91_Protocol, si crea una dipendenza circolare
from .EE91 import EE91
from .experiments import *
from .classifiers import *
from .utils import *
from .extract import *
from .noise import *

import pandas as pd ; pd.set_option('future.no_silent_downcasting', True) # type: ignore # disabilita i suoi warning inutili

'''
N.B.: non eseguire mai direttamente i moduli all'interno di un package. Usare sempre un entrypoint esterno (come "example.py") per importare e utilizzare il package.

project_root/
├── example.py
└── package/
    ├── __init__.py
    ├── ...
    ├── utils.py
    ├── ...
    ├── E91.py
    └── ...
'''

__all__ = ["E91", "EE91", "experiments", "classifiers", "utils", "extract", "noise"]
