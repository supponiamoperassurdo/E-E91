# imports
import warnings
import os
import joblib
import numpy as np
import pandas as pd

# SkLearn
from sklearn import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, fbeta_score, f1_score, matthews_corrcoef, confusion_matrix

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Hopfield is All You Need
from hflayers import Hopfield

# package
from .utils import *

#################################################################################################################################################################################################################

warnings.filterwarnings(
    "ignore",
    message = "X has feature names, but",
    category = UserWarning
)

warnings.filterwarnings(
    "ignore",
    message = "optree is installed but the version is too old to support PyTorch Dynamo in C++ pytree",
    category = FutureWarning
)

#################################################################################################### SkLearn ####################################################################################################

class DataTransformer:

    '''
    Classe generica wrapper che instanzia un imputatore e uno scaler `sklearn` al fine di imputare i dati NaN e riscalare i dati con tecniche a proprio piacimento. \\
    In "Parameters" sono definiti gli elementi del costruttore. \\
    Utilizza la classe from `sklearn.pipeline.Pipeline`.

    :param imputer: Imputatore di input compatibile con `sklearn`.
    :param scaler: Scalatore di input compatibile con `sklearn`.
    '''

    def __init__(self, imputer: any = None, scaler: any = None) -> None:

        '''
        :param imputer: Imputatore di input compatibile con `sklearn`.
        :param scaler: Scalatore di input compatibile con `sklearn`.
        '''

        # se è un file path, caricalo
        if isinstance(imputer, str):
            self.imputer = joblib.load(imputer)
        else:
            self.imputer = imputer
        
        # se è un file path, caricalo
        if isinstance(scaler, str):
            self.scaler = joblib.load(scaler)
        else:
            self.scaler = scaler

    def __str__(self):
        return f"Imputer: {self.imputer.__class__.__name__}\tScaler: {self.scaler.__class__.__name__}"

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        '''
        Addestra e trasforma i dati sull'imputer e sullo scaler, li divide in train set e test set, e salva imputer e scaler su due file pickle per essere riutilizzati in seguito.

        :param X: Dataframe di input in formato numpy.

        :return None:
        '''

        # se esiste l'imputer per i dati NaN...
        if self.imputer is not None:

            # addestra e trasforma
            X = self.imputer.fit_transform(X)

        # se esiste lo scaler...
        if self.scaler is not None: # normalizzazione dei dati

            # addestra (solo sul train set) e trasforma entrambi i dati
            X = self.scaler.fit_transform(X)

        return X

    def fit(self, X: np.ndarray) -> None:

        '''
        Addestra l'imputer e lo scaler sui dati di input, e se salva quest'ultimi su due file pickle per essere riutilizzati in seguito.

        :param X: Dataframe di input in formato numpy.

        :return None:
        '''

        # se esiste l'imputer per i dati NaN...
        if self.imputer is not None:

            # addestra solamente
            self.imputer.fit(X)
            joblib.dump(self.imputer, f'imputer_{self.imputer.__class__.__name__}.pkl')
        
        # se esiste lo scaler...
        if self.scaler is not None:

            # addestra solamente 
            self.scaler.fit(X)
            joblib.dump(self.scaler, f'scaler_{self.scaler.__class__.__name__}.pkl')

    def transform(self, X: np.ndarray) -> np.ndarray:
        
        '''
        Trasforma i dati di input sfruttando l'imputer e lo scaler salvati su file pickle.

        :param X: Dataframe di input in formato .numpy

        :return None:

        Si può verificare che siano già stati addestrati tramite il metodo:

            hasattr(imputer, "statistics_")
            hasattr(scaler, "scale_")
            hasattr(clf, "coef_")
        '''
        
        # se esiste l'imputer per i dati NaN...
        if self.imputer is not None:

            # se è fittato, trasforma i dati
            if hasattr(self.imputer, "statistics_"):

                X = self.imputer.transform(X)
        
        # se esiste lo scaler per la normalizzazione dei dati...
        if self.scaler is not None:

            # se è fittato, trasforma i dati
            if hasattr(self.scaler, "scale_"):

                X = self.scaler.transform(X)

        return X

    def is_fitted(self) -> tuple[bool]:

        '''
        Restituisce un booleano per dire se l'imputer e lo scaler sono fittati o meno.
        
        :param None:

        :return bool, bool: Tupla in cui il primo bool è l'informazione sull'imputer, il secondo è l'informazione sullo scaler. Saranno "True" se sono fittati.
        '''

        return hasattr(self.imputer, "statistics_"), hasattr(self.scaler, "scale_")

    def save_model(self, dir: str) -> None:

        '''
        Salva imputatore e scaler addestrati su disco.

        :param dir: Directory in cui salvare i modelli.
        :return None:
        '''

        if not os.path.exists(dir):
            os.makedirs(dir)

        if self.imputer is not None:
            filename = f'imputer_{self.imputer.__class__.__name__}.pkl'
            filepath = os.path.join(dir, filename)
            joblib.dump(self.imputer, filepath)

        if self.scaler is not None:
            filename = f'scaler_{self.scaler.__class__.__name__}.pkl'
            filepath = os.path.join(dir, filename)
            joblib.dump(self.scaler, filepath)

    @classmethod
    def load_model(cls, filepath_imputer: str, filepath_scaler: str) -> None:

        '''
        Carica il modello addestrato da disco.

        :param filepath: Percorso del file da cui caricare il modello.
        :return None:
        '''

        with open(filepath_imputer, 'rb') as f:
            cls.imputer = joblib.load(f)

        with open(filepath_scaler, 'rb') as f:
            cls.scaler = joblib.load(f)
        
#################################################################################################### SkLearn ####################################################################################################

class Classifier:

    '''
    Classe generica wrapper che ingloba un classificatore binario `sklearn` con cui è possibile eseguire i due metodi principali:
    
    - `fit(X, y)`: addestra il classificatore su un un dataset X e un target y;
    - `predict(X)`: esegue la predizione sun un dataset non noto y;
    - `evaluate(X, y)`: valuta le performance del classificatore rispetto a un dataset.

    In "Parameters" sono definiti gli elementi del costruttore. \\
    Si può caricare il file pickle `.pkl` (inserendo il path di quest'ultimo) del corrispondente classificatore che si vuole usare per la predizione.
    '''

    def __init__(self, clf: any = None) -> None:

        '''
        :param clf: Classificatore o regressore di input compatibile con `sklearn`.
        '''

        if isinstance(clf, str): 
            self.clf = joblib.load(clf)
        else:
            self.clf = clf

        self.scores = {
            "mae": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "f2": 0,
            "mcc": 0,
            "confusion_matrix": None
        }

    def __str__(self):
        return f"{self.clf.__class__.__name__}"

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> dict:

        '''
        Funziona solo con modelli appartenenti al modulo `sklearn`. \\
        Addestra un classificatore `clf` per classificare un target `y`.

        :param X: Dataframe di input in formato numpy.
        :param y: Colonna del target che si vuole classificare in formato numpy.
        :param verbose: Abilità la visualizzazione a schermo degli scores del sotto-dataset di validazione.
        
        :return scores: Dizionario di score con le seguenti chiavi:

            - mae
            - accuracy
            - precision
            - recall
            - f1
            - f2
            - mcc
            - confusion_matrix
        '''

        if self.clf is None or y is None:
            raise ValueError("specificare un classificatore e una colonna target di input: parametri obbligatori.")

        # Step 1: si esegue la separazione in set di addestramento e di validazione
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 42)

        # Step 2: addestramento
        self.clf.fit(X_train, y_train)

        # Step 3: calcolo delle performance
        self.scores = self.evaluate(X = X_test, y = y_test, verbose = verbose)

        return self.scores

    def predict(self, X: np.ndarray, y: np.ndarray = None) -> list:

        '''
        Funziona solo con modelli appartenenti al modulo `sklearn`. \\
        Esegue la predizione dei dati di un dataset non noto.

        :param X: Dataset che si vuole predire (privo del target) in formato numpy.
        
        :return predict: Ritorna la lista delle predizioni eseguite.
        '''

        return self.clf.predict(X)
    
    def predict_proba(self, X: np.ndarray, y: np.ndarray = None) -> list:

        '''
        Funziona solo con modelli appartenenti al modulo `sklearn`. \\
        Esegue la predizione dei dati di un dataset non noto, ma ritorna la loro probabilità di predizione come [%CLASSE 0, %CLASSE 1, ...].

        :param X: Dataset che si vuole predire (privo del target) in formato numpy.
        
        :return predict: Ritorna la lista delle predizioni eseguite.
        '''

        return self.clf.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> dict:

        '''
        Valuta le performance del classificatore.

        :param X: Dataframe in formato numpy contenente i dati delle features da voler valutare.
        :param y: Colonna target su cui valutare la performance (si suppone di lavorare con un modello di apprendimento supervisionato).
        :param verbose: Abilità la visualizzazione a schermo degli scores.

        :return scores: Dizionario di score con le seguenti chiavi:

            - mae
            - accuracy
            - precision
            - recall
            - f1
            - f2
            - mcc
            - confusion_matrix
        '''

        # esegui la predizione e valutiamo la possibilità che clf possa essere un path o un oggetto
        predict = self.predict(X)

        # calcola metriche di valutazione
        self.scores["mae"] = mean_absolute_error(y, predict)
        self.scores["accuracy"] = accuracy_score(y, predict)
        self.scores["precision"] = precision_score(y, predict)
        self.scores["recall"] = recall_score(y, predict)
        self.scores["f1"] = f1_score(y, predict, average = "weighted", pos_label = None, zero_division = 0)
        self.scores["f2"] = fbeta_score(y, predict, beta = 2, zero_division = 0)
        self.scores["mcc"] = matthews_corrcoef(y, predict)
        self.scores["confusion_matrix"] = confusion_matrix(y, predict)

        if verbose:

            print("\n>>> Performance\n")
            print(f"Classifier:\t{self.__str__()}")
            print(f"Accuracy:  \t{self.scores['accuracy']:.4f}")
            print(f"Precision: \t{self.scores['precision']:.4f}")
            print(f"Recall:    \t{self.scores['recall']:.4f}")
            print(f"F1-Score:  \t{self.scores['f1']:.4f}")
            print(f"F2-Score:  \t{self.scores['f2']:.4f}")
            print(f"MCC:       \t{self.scores['mcc']:.4f}")

        return self.scores

    def clone(self) -> any:

        '''
        Ritorna una copia del classificatore con gli stessi parametri ma senza aver eseguito il fit su alcun dato.

        :param None:

        :return BaseEstimator: Copia del classificatre originale non addestrato. Si osservi che tale copia è già un oggetto [SkLearn](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone).
        '''

        return clone(self.clf)

    def is_fitted(self) -> bool:
        
        '''
        Verifica se il classificatore è stato fittato o meno.
        
        :param None:

        :return bool: Restituisce "True" se il classificatore è addestrato.
        '''
        
        return (hasattr(self.clf, "coef_") or hasattr(self.clf, "classes_"))

    def cross_val_score(self, X: np.ndarray, y: np.ndarray, cv: int = 5, params: list = None, verbose: bool = False) -> np.ndarray:

        '''
        Esegue la cross-validazione sul classificatore. \\
        Ovviamente, se è già addestrato, esegue l'operazione su un clone non fittato.

        :param X: Dataframe di input in formato numpy.
        :param y: Colonna del target che si vuole classificare in formato numpy.
        \\# :param scoring: NON UTILIZZATO Funzione di score che si vuole utilizzare, vedi su [SkLearn](https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers). Di default si userà *recall* visto che il problema di classificazione principale riguarda i falsi negativi.
        :param cv: Numero di cross-validazioni.
        :param params: Parametri da passare al classificatore in fase di fit, o altro.
        :param verbose: Se abilitato, mostra i risultati dello score.

        :return dict[np.ndarray]: Dizionario di score con 5 chiavi (accuracy, precision, recall, f1, roc_auc), con valori lungi *cv* o *len(cv)* (se si è passato un iteratore).
        '''

        # estraggo l'attributo "clf" che è un'oggetto SkLearn, necessario per la cross-validazione
        estimator = self.clone() 

        # inizializzo il dizionario di scores
        scores = {}

        # eseguo la cross-validazione
        scores['accuracy']  = cross_val_score(estimator = estimator, X = X, y = y, scoring = "accuracy", cv = cv, params = params)
        scores['precision'] = cross_val_score(estimator = estimator, X = X, y = y, scoring = "precision", cv = cv, params = params)
        scores['recall']    = cross_val_score(estimator = estimator, X = X, y = y, scoring = "recall", cv = cv, params = params)
        scores['f1']        = cross_val_score(estimator = estimator, X = X, y = y, scoring = "f1", cv = cv, params = params)
        scores['roc_auc']   = cross_val_score(estimator = estimator, X = X, y = y, scoring = "roc_auc", cv = cv, params = params)

        if verbose:

            print("\n>>> CV scores\n")
            print(f"Classifier:\t{estimator.__class__.__name__}")
            print(f"Accuracy:  \t{np.round(scores['accuracy'], 4)}")
            print(f"Precision: \t{np.round(scores['precision'], 4)}")
            print(f"Recall:    \t{np.round(scores['recall'], 4)}")
            print(f"F1-Score:  \t{np.round(scores['f1'], 4)}")
            print(f"ROC-AUC:   \t{np.round(scores['roc_auc'], 4)}")

        return scores

    def save_model(self, dir: str, filename: str = None) -> None:

        '''
        Salva il modello addestrato su disco.

        :param dir: Directory in cui salvare il modello.
        :param filename: Nome del file in cui salvare il modello.
        
        :return None: 
        '''

        if not os.path.exists(dir):
            os.makedirs(dir)

        if filename is None:
            filename = f'model_{self.clf.__class__.__name__}.pkl'
    
        filepath = os.path.join(dir, filename)
        joblib.dump(self.clf, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> None:

        '''
        Carica il modello addestrato da disco.

        :param filepath: Percorso del file da cui caricare il modello.
        :return None:
        '''

        with open(filepath, 'rb') as f:
            model = joblib.load(f)

        return cls(clf = model)

#################################################################################################### PyTorch ####################################################################################################

class EarlyStopping:

    def __init__(self, patience = 5, min_delta = 0.01):
        
        '''
        :param patience: Numero di epoche da attendere senza miglioramenti.
        :param min_delta: Miglioramento minimo richiesto per resettare il contatore.
        '''
        
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = None
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_loss): # in questo modo posso utilizzare direttamente l'istanza di classe come una funzione

        '''
        :param val_loss: Valore della loss da valutare.
        
        :return early_stop, best_loss: Tupla il cui primo valore è un boolean che afferma se c'è o meno stato un early stop, e il secondo valore è la migliore loss delle ulime "patience" epoche.
        '''

        if self.best_loss is None:    
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss # se supera il valore di delta, impostalo come valore migliore di loss
            self.counter = 0          # resetta il contatore se c'è un miglioramento
        
        else:
            
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop, self.best_loss

#################################################################################################### PyTorch ####################################################################################################

class NeuralNetwork(nn.Module):
    
    '''
    Classe che implementa una semplice rete neurale feed-forward.
    '''

    def __init__(self, layer_dimensions: int) -> None:

        '''
        :param layer_dimensions: Lista dei neuroni per ogni layer. Ci si assicuri che:
        
            - il numero di neuroni del primo layer rispecchi la dimensionalità delle colonne del dataset (es. con `X.columns.size`);
            - il numero di neuroni dell'ultimo layer rispecchi il problema che si vuole risolvere (es. 2 neuroni per un problema di classificazione binaria).
        
        :return None:
        '''

        super(NeuralNetwork, self).__init__()

        # necessario per il save-load model
        self.layer_dimensions = layer_dimensions

        # conservo l'insieme di hidden layer compresi tra l'input e l'output layer: il numero di strati e di neuroni si possono stabilire dal costruttore 
        layers = []
        for i in range(len(layer_dimensions) - 1):

            layers.append(nn.Linear(layer_dimensions[i], layer_dimensions[i + 1]))
            if i != len(layer_dimensions) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            *layers,     # spacchetto i layers generati
            nn.Sigmoid() # si mappa l'output con la sigmoide nell'intervallo [0, 1]
        )

    def __str__(self):
        elements = []
        for i, module in enumerate(self.layers):
            if isinstance(module, nn.Linear):
                activation = next(
                    (
                        m.__class__.__name__ for m in self.layers[i + 1:] if not isinstance(m, nn.Linear)
                    ),
                    "Sigmoid" if i == len(self.layers) - 2 else None
                )
                elements.append(f"[{module.in_features}, {activation}]")
        return f"NeuralNetwork({' → '.join(elements)})"

    def forward(self, x: nn.Sequential) -> any:    
        
        '''
        Il metodo `.forward(self, x)` è il metodo di classe principale ereditato del modulo `torch.nn`:
        
        - fondamentalmente, a seconda di come è definito, esegue sequenzialmente i layer definiti in `__init__` (vedi il caso dell'auto-encoder dopo);
        - in esso, si possono definire direttamente le funzioni di attivazione da applicare a fine layer;
        - diciamo che, l'approccio qui utilizzato è più [Keras-like](https://www.tensorflow.org/tutorials/quickstart/beginner?hl=it#build_a_machine_learning_model), che differisce da quello di [PyTorch](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network).
        
        Alla fine, "ritorna" il layer finale. \\
        Questa funzione viene utilizzata automaticamente.

        :param x: Un oggetto Sequential (una porzione di rete).

        :return layer: Attraversa l'intera rete e restituisce l'output di questa.
        '''

        return self.layers(x)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, batch_size: int = 32, epochs: int = 10, criterion: any = None, optimizer: torch.optim.Optimizer = None, earlystopping: EarlyStopping = None, test_size: float = 0.3, verbose: bool = True) -> None:

        '''
        Metodo di training della rete neurale. \\
        Per semplicità, si passeranno `X` e `y` in formato `np.ndarray`, che verranno poi appositamente trasformati in un `torch.utils.data.DataLoader`.
        
        :param X: Dataframe in formato numpy contenente i dati delle features da voler valutare.
        :param y: Colonna target su cui valutare la performance (si suppone di lavorare con un modello di apprendimento supervisionato).
        :param batch_size: Dimensione del batch di addestramento.
        :param epochs: Numero di epoche (aggiornamenti della rete).
        :param criterion: Funzione di loss che si vuole utilizzare.
        :param optimizer: Funzione di aggiornamento.
        :param test_size: Dimensione da dedicare al test set per valutare le performance del modello dopo l'addestramento.
        :param verbose: Se impostato a "True", mostra le epoche di addestramento e la loss.

        :return None:
        '''

        # suddivisione tra train set e test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state = 42)

        # tensorializzazione dei dati
        X_train    = torch.tensor(X_train, dtype = torch.float32) # si usa torch.float32 e...
        y_train    = torch.tensor(y_train, dtype = torch.long)    # torch.long per classificazione, se come loss utilzziamo CrossEntropyLoss
        train_data = TensorDataset(X_train, y_train)
        dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        
        # addestramento
        for epoch in range(epochs):

            for batch_index, (data, targets) in enumerate(dataloader):

                # forward pass
                outputs = self(data)
                loss    = criterion(outputs, targets)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if verbose:
                print(f'\r>>> Epoch {epoch + 1} / {epochs}\t\t\tLoss: {loss.item():.4f}', end = "", flush = True)
            
            # valutazione per l'early-stopping
            early_stop, best_loss = earlystopping(val_loss = loss)

            # salvataggio del modello con la best loss per consentire il ripristino in caso di early-stopping
            checkpoint = {
                'epoch': epoch + 1,                             # epoca corrente
                'model_state_dict':     self.state_dict(),      # salvataggio del modello
                'optimizer_state_dict': optimizer.state_dict(), # salvataggio dell'ottimizzatore
                'best_loss':            best_loss               # miglior loss ottenuta fino ad ora
            }
            torch.save(checkpoint, '__best_model_properties__.pth')

            # early-stopping
            if early_stop:

                print(f"\n\n>>> Early-stop epoch: {epoch + 1}\t\tBest loss: {best_loss:.4f}")
                break

        if early_stop:

            # caricamento dell'epoca con la loss migliore se c'è stat early-stopping: subito dopo, si cancella il file "temporaneo"
            checkpoint = torch.load('__best_model_properties__.pth') ; os.remove('__best_model_properties__.pth')
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f"\n>>> Model restored from epoch: {checkpoint['epoch']}\tBest loss: {checkpoint['best_loss']:.4f}.")

        # valutazione dello score modello sui dati di test
        self.evaluate(X = X_test, y = y_test, batch_size = batch_size, verbose = verbose)
        
    def _predict_proba_array(self, X: pd.DataFrame, y: pd.DataFrame = None, probability: bool = False) -> list[int]:

        '''
        Metodo di predizione della rete neurale.
        
        :param X: Dataframe in formato numpy contenente i dati delle features da voler valutare.
        :param probability: Se settato a True, restituisce la probabilità di classificazione associata alle classi.

        :return predict, predict_proba: Lista delle predizioni eseguite su X, lista delle probabilità di predizioni eseguite su X.
        '''
    
        self.eval()           # imposta il modello (di nn.Module, definito in __init__) in modalità "evaluation"
        with torch.no_grad(): # disabilita il calcolo dei gradienti
            
            '''
            Returns a new tensor with a dimension of size one inserted at the specified position.

                >>> x = torch.tensor([1, 2, 3, 4])
                >>> torch.unsqueeze(x, 0)
                    tensor([[ 1,  2,  3,  4]])
                >>> torch.unsqueeze(x, 1)
                    tensor([[ 1],
                            [ 2],
                            [ 3],
                            [ 4]])

            Con "dim = 0", esegue un flattening.
            '''

            if isinstance(X, pd.DataFrame):     # se è un dataframe, lo si trasforma in una lista numpy
                X = X.values

            if not isinstance(X, torch.Tensor): # tensorializza X nel caso in cui non lo sia
                input_tensor = torch.tensor(X, dtype = torch.float32)

            outputs = self(input_tensor)                # attraversa la rete neurale e gli output sono i logits
            probabilities = F.softmax(outputs, dim = 1) # applico la softmax lungo la dimensione delle classi: restituisce un tensore in cui ogni elemento rappresenta la probabilità associata alla rispettiva classe
  
        return probabilities
    
    def predict_proba(self, X):

        '''
        Return probabilities array (n_samples, n_classes).
        '''
        return self._predict_proba_array(X) # le probabilità associate alle classi sono proprio date dalla softmax

    def predict(self, X):

        '''
        Return label predictions (0/1/... depending on classes).
        '''
        probs = self.predict_proba(X)         
        predict = torch.argmax(probs, axis = 1) # calcola l'argmax delle probabilità, ritornando la classe predetta

        # Il metodo .item() funziona solo per tensori con un solo oggetto: per classificare interi dataset (che non possiedono un solo oggetto) è necessario ritornare tutti i dati; così, si converte da torch.Tensor a lista
        return predict.tolist()

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, verbose: bool = False) -> dict:

        '''
        Valuta le performance della rete neurale.

        :param X: Dataframe in formato numpy contenente i dati delle features da voler valutare.
        :param y: Colonna target su cui valutare la performance (si suppone di lavorare con un modello di apprendimento supervisionato).
        :param verbose: Se settato a "True", stampa gli scores.

        :return scores: Dizionario contenente le seguenti chiavi:

            - accuracy
            - precision
            - recall
            - f1
            - f2
            - mcc
            - confusion_matrix
        '''

        X          = torch.tensor(X, dtype = torch.float32)
        y          = torch.tensor(y, dtype = torch.long)
        dataset    = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

        all_predicts = []
        all_targets = []
        scores = {}

        # si imposta il modello in modalità valutazione e si eseguono le predizioni sul modello già addestrato (utilizzo della funzione max)
        self.eval()
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = self(data)
                _, preds = torch.max(outputs, 1)
                all_predicts.extend(preds.numpy())
                all_targets.extend(targets.numpy())
        
        scores['accuracy'] = accuracy_score(all_targets, all_predicts)
        scores['precision'] = precision_score(all_targets, all_predicts, average = 'weighted')
        scores['recall'] = recall_score(all_targets, all_predicts, average = 'weighted')
        scores['f1'] = f1_score(all_targets, all_predicts, average = 'weighted')
        scores['f2'] = fbeta_score(all_targets, all_predicts, beta = 2, zero_division = 0)
        scores['mcc'] = matthews_corrcoef(all_targets, all_predicts)
        scores['confusion_matrix'] = confusion_matrix(all_targets, all_predicts)

        if verbose:

            print("\n>>> Performance\n")
            print(f"Accuracy: \t{scores['accuracy']:.4f}")
            print(f"Precision:\t{scores['precision']:.4f}")
            print(f"Recall:   \t{scores['recall']:.4f}")
            print(f"F1-Score: \t{scores['f1']:.4f}")
            print(f"F2-Score: \t{scores['f2']:.4f}")
            print(f"MCC:      \t{scores['mcc']:.4f}")
            
        return scores
    
    def save_model(self, dir: str, filename: str = None) -> None:

        '''
        Salva il modello PyTorch con cui si sta lavorando. \\
        Il metodo di salvataggio utilizzato è quello [consigliato](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference),
        tramite l'uso di state_dict.

        :param filepath: Posizione del file in cui salvare il modello PyTorch.

        :return None:
        '''
        
        if not os.path.exists(dir):
            os.makedirs(dir)

        if filename is None:
            filename = 'model_NeuralNetwork.pth'
    
        filepath = os.path.join(dir, filename)

        torch.save({
            'config': { # in config ci andranno tutti gli elementi del costruttore: lo specifico così che si possa generalizzare in un'altra rete
                'layer_dimensions':   self.layer_dimensions
            },
            'state_dict':       self.state_dict()
        },
            filepath
        )

    @classmethod
    def load_model(cls, filepath: str) -> nn.Module:

        '''
        Carica il modello PyTorch con cui si sta lavorando. \\
        Dalla [documentazione](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference):

        "Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. \\
        Failing to do this will yield inconsistent inference results."

        :param path: Posizione del file contenente le informazioni sulla rete neurale.

        :return model: Rete neurale nn.Module già costuita con gli stessi parametri salvati.
        '''
    
        # carico lo state_dict
        properties = torch.load(filepath, weights_only = False)
        config = properties['config']

        # crea un'istanza del modello (cls è il corrispondente di "self" per i classmethod, quindi richiama la classe stessa)
        model = cls(**config)

        # creo il modello
        model.load_state_dict(properties['state_dict'])
        
        # imposta il modello in modalità evaluation, in base a quanto riportato sopra
        model.eval() 

        return model

#################################################################################################################################################################################################################

class ModernHopfieldNetwork(NeuralNetwork):

    '''
    Classe che implementa una rete di Hopfield moderna per l'apprendimento supervisionato.
    Il modello si basa su [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) di Demis Hassabis et al. (2020),
    e sull'implementazione in PyTorch di [Hopfield](https://github.com/borisdayma/hopfield).
    '''

    def __init__(self, input_dimension: int = None, hidden_dimension: int = 64, output_dimension: int = 2):

        '''
        :param input_dimension: Numero dei neuroni di input, deve rispecchiare la dimensionalità delle colonne del dataset (es. con `X.columns.size`).
        :param hidden_dimension: Numero dei neuroni del layer nascosto.
        :param output_dimension: Numero di neuroni dell'ultimo layer, che rispecchia il problema che si vuole risolvere (es. 2 neuroni per un problema di classificazione binaria).
        
        :return None:
        '''

        super().__init__([]) # <- vedere qui

        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension

        # 1) Proiezioni lineari per Q, K, V
        self.to_q = nn.Linear(input_dimension, hidden_dimension)
        self.to_k = nn.Linear(input_dimension, hidden_dimension)
        self.to_v = nn.Linear(input_dimension, hidden_dimension)
        
        # 2) Hopfield layer (1 head, scaling √d scaled)
        self.hopfield = Hopfield(
            input_size   = hidden_dimension, # dimensione di Q/K/V in ingresso
            hidden_size  = hidden_dimension, # head_dim
            output_size  = hidden_dimension, # out_dim (output projection)
            pattern_size = hidden_dimension, # pattern_dim
            num_heads    = 1,
            scaling      = hidden_dimension ** -0.5,
            update_steps_max = 1,
            batch_first  = True
        )

        self.output = nn.Linear(hidden_dimension, output_dimension)
        
        # 3) Testa finale per output (regressione/scalar)
        self.output = nn.Linear(hidden_dimension, output_dimension)

    def __str__(self):
        return f"ModernHopfieldNetwork(input_dim = {self.input_dimension}, hidden_dim = {self.hidden_dimension}, output_dim = {self.output_dimension})"

    def forward(self, x):

        Q = self.to_q(x).unsqueeze(1)  # l'input ha dimensione [N, 1, hidden_dimension]
        K = self.to_k(x).unsqueeze(1)  # stesse stored patterns
        V = self.to_v(x).unsqueeze(1)
        
        Z = self.hopfield((Q, K, V))  # [N, 1 hidden_dimension]
        Z = Z.squeeze(1)              # [N, output_dimension]
        return self.output(Z)         # [N, output_dimension]

    def save_model(self, dir: str, filename: str = None) -> None:

        '''
        Salva il modello PyTorch con cui si sta lavorando. \\
        Il metodo di salvataggio utilizzato è quello [consigliato](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference),
        tramite l'uso di state_dict.
        
        :param None:

        :return None:
        '''
                
        if not os.path.exists(dir):
            os.makedirs(dir)

        if filename is None:
            filename = 'model_ModernHopfieldNetwork.pth'
    
        filepath = os.path.join(dir, filename)

        torch.save({
            'config': { # in config ci andranno tutti gli elementi del costruttore: lo specifico così che si possa generalizzare in un'altra rete
                'input_dimension':  self.input_dimension,
                'hidden_dimension': self.hidden_dimension,
                'output_dimension': self.output_dimension
            },
            'state_dict':           self.state_dict()
        },
            filepath
        )

#################################################################################################################################################################################################################
