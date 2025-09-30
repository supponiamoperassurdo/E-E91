import ast
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, mode # type: ignore

################################################################################################################################################################################################################

def get_anticorrelation_distribution(chsh_samples: dict) -> dict:

    '''
    Calcola la percentuale di anti-correlazione per ogni base del test di Bell. \\
    Si osservi che la base dell'addendo negativo dovrebbe avere una anti-correlazione tendenzialmente bassa (idealmente 14.64%), essendo quello a correlazione positiva.
    
    :param chsh_samples: Campioni delle misure di Alice e Bob (già moltiplicati tra loro, per come è scritto il codice E91_Protocol()). Se è -1, Alice e Bob hanno basi anti-correlate.
    
    :return anticorrelation_values: Percentuale di anticorrelazione per ogni base di Bell. Ognuna di esse deve tendere idealmente al 70.71% per casi di anti-correlazione, e 14.64% per casi di correlazione.
    '''

    anticorrelation_distribution = {}
    
    for bases, samples in chsh_samples.items():
        if samples:
            counter = sum(1 for s in samples if s == -1)
            anticorrelation_distribution[bases] = counter / len(samples) # percentuale di anticorrelazione
    
    return anticorrelation_distribution

################################################################################################################################################################################################################

def __aggregate_chsh_samples_statistics_by(chsh_samples_statistics: dict[any, dict[str, float]], method: str = "mean") -> dict[str, float]:

    '''
    Funzione ausiliaria per aggregare i 6 dizionari restituiti dalla funzione `get_chsh_samples_statistics`:
    
    - salva, per ognuna delle 4 basi, il dizionario della corrispondente statistica salvata nella corispondente lista (es. "modes", "medians", etc);
    - la tecnica di aggregazione utilizzata è la media, ma si potrebbero, in futuro, utilizzare anche tecniche.

    :param chsh_samples_statistics: Dizionario contenente i 6 dizionari riguardanti le statistiche per ognuna delle 4 basi.

    :return dict: Dizionario di 6 elementi (le basi del test di Bell).
    '''

    # mappo i nomi dei metodi ai corrisponenti operatori numpy e scipy
    aggregate_functions: dict[str, callable[[list], float]] = {
        "mode":     lambda xs: float(mode(xs)[0]),     
        "median":   lambda xs: np.median(xs),
        "mean":     lambda xs: np.mean(xs),
        "variance": lambda xs: np.var(xs),   
        "skewness": lambda xs: skew(xs),
        "kurtosis": lambda xs: kurtosis(xs),
    }

    if method not in aggregate_functions:
        raise ValueError(
            f"Metodo di aggregazione '{method}' non riconosciuto:"
            f"scegli tra {list(aggregate_functions.keys())}."
        )

    # raccolgo le statistiche dalle quattro basi
    modes, medians, means, variances, skewnesses, kurtosises = [], [], [], [], [], []
    for key, value in chsh_samples_statistics.items():

        modes.append(value["mode"]) # le mode delle basi (X, W), (X, V), (Z, W), (Z, V)
        medians.append(value["median"])
        means.append(value["mean"])
        variances.append(value["variance"])
        skewnesses.append(value["skewness"])
        kurtosises.append(value["kurtosis"])
    
    # le aggrego sulla base del metodo scelto in input
    return {
        f"CHSH samples mode ({method})":     aggregate_functions[method](modes)      if modes else np.nan, # calcola la media delle mode delle basi (X, W), (X, V), (Z, W), (Z, V)
        f"CHSH samples median ({method})":   aggregate_functions[method](medians)    if medians else np.nan,
        f"CHSH samples mean ({method})":     aggregate_functions[method](means)      if means else np.nan,
        f"CHSH samples variance ({method})": aggregate_functions[method](variances)  if variances else np.nan,
        f"CHSH samples skewness ({method})": aggregate_functions[method](skewnesses) if skewnesses else np.nan,
        f"CHSH samples kurtosis ({method})": aggregate_functions[method](kurtosises) if kurtosises else np.nan,
    }

################################################################################################################################################################################################################

def get_chsh_samples_statistics(chsh_samples: dict) -> dict[dict]:

    '''
    Genera una serie di statistiche a partire dai risultati delle misure del protocollo E91 utilizzate per il test CHSH. \\
    Si osservi che questa funzione verrà spesso usata nei pd.Series, perciò si abbia l'accortezza di usarla come:

        df["CHSH samples"].apply(extract.get_chsh_samples_statistics(method =))
    
    :param chsh_samples: Dizionario dei campioni raccolti per il test di Bell.
    :param method: Metodo di aggregazione del dizionario delle statistiche delle basi del test di Bell. 
    
    :return chsh_samples_aggregated_statistics: 6 dizionari di 6 elementi (mode, median, mean, variance, skewness, kurtosis) dei campioni del test di Bell aggregati tra loro tramite 6 metodi statistici.
    '''
 
    # sanificazione dell'input nell'ipotesi in cui venga caricato da un file CSV: si può applicare anche a un dizionario diretto
    if isinstance(chsh_samples, str):
        chsh_samples = ast.literal_eval(chsh_samples)
    
    chsh_samples_statistics = {}
    if chsh_samples: # mi assicuro il dizionario non sia vuoto
        for key, value in chsh_samples.items(): # passo chiave e valore del dizionario
            chsh_samples_statistics[key] = {
                "mode":     float(mode(value)[0]), # estraggo solo le info sulla moda e lo "trasformo" in un datatype Python (su scipy sono spicy)
                "median":   np.median(value),
                "mean":     np.mean(value),
                "variance": np.var(value),
                "skewness": skew(value),
                "kurtosis": kurtosis(value),
            }
    else:
        raise ValueError("lista di campioni vuota.")

    # definisco i metodi di aggregazione:
    methods = ["mode", "median", "mean", "variance", "skewness", "kurtosis"]

    # salvo tutte le aggregazioni eseguite
    chsh_samples_aggregated_statistics = []
    for method in methods:
        chsh_samples_aggregated_statistics.append(
            __aggregate_chsh_samples_statistics_by(chsh_samples_statistics = chsh_samples_statistics, method = method)
        )

    return chsh_samples_aggregated_statistics

################################################################################################################################################################################################################

def extract_bloch_vector(df: pd.DataFrame, n_measurements: int = 1000) -> pd.DataFrame:
    
    '''
    Estrae e corrompe i vettori di Bloch con errori di tomografia realistici, reinserendoli correttamente nel dataframe originario.
    
    :param df: DataFrame contenente i vettori di Bloch "perfetti" (da simulazione).
    :param n_measurements: Numero di misure per componente Pauli (X, Y, Z). Default 1000 (~3.2% errore per componente).
    
    :return: DataFrame con vettori di Bloch "realistici" (media ± 1/√N).
    '''
    
    def _process_bloch_vector(bloch_vector: pd.Series, bloch_vector_name: str = "", n_measurements: int = 1000):

        '''
        Estrae e corrompe i vettori di Bloch con rumore di tomografia.

        :param bloch_vector: Vettori di Bloch da estrarre e corrompere.
        :param n_measurements: Di default, esegue un numero di misure per componente Pauli (X, Y, Z) pari a 1000 (~3.2% errore per componente). Impostando a "None" non esegue alcuna corruzione.
        '''

        # trasformazione necessarie: nel dizionario il vettore è salvato come una stringa, quindi si deve ritrasformare; se non proviene dal dizionario, non eseguire alcuna trasformazione
        if not isinstance(bloch_vector, np.ndarray):
            # verifica se ogni elemento è una stringa prima di applicare .strip()
            bloch_vector = bloch_vector.apply(
                lambda x: np.fromstring(x.strip('[]'), sep = ' ') if isinstance(x, str) else x
            )
        # se bloch_vector è già un array NumPy, non fare nulla
        else:
            pass
        
        '''
        L'effetto di np.stack è il seguente:

            >>> arrays = [
            ...    np.array([1.0, 2.0, 3.0]),
            ...    np.array([4.0, 5.0, 6.0]),
            ...    np.array([7.0, 8.0, 9.0])
            ...]

        Si combinano lungo l'asse 0 (righe)
            
            >>> np.stack(arrays, axis = 0)
            >>> [[1. 2. 3.]
                [4. 5. 6.]
                [7. 8. 9.]]

        Lo shape risultante è (3, 3) (3 righe, 3 colonne), dove ogni riga corrisponde al vettore originale.
        È necessario poiché df['Alice Bloch vector'].apply(np.fromstring) restituisce una colonna di oggetti (array separati), 
        e SkLearn non può gestire colonne di oggetti direttamente, perché si aspetta una matrice numerica 2D (dove ogni cella è uno scalare).
        Infatti, se si prova:

            >>> df["Alice Bloch vector"].apply(lambda x: np.fromstring(x.strip('[]'), sep = ' '))
            >>> 0         [4.5660541e-18, 2.76289721e-18, 0.0]
                1        [-0.0234375, -0.00828641, 0.03172391]
                2      [-0.02734375, -0.00276214, -0.00309724]
                3         [0.0234375, 0.01795388, -0.01795388]
                4        [7.47163493e-18, 7.00606702e-18, 0.0]
                                        ...                   
                922      [-0.00585937, 0.00552427, 0.01010073]
                923       [0.02539062, -0.0041432, 0.01781508]
                924     [0.02148437, -0.00828641, -0.01905734]
                925       [3.47954192e-18, 9.7979233e-19, 0.0]
                926      [6.07169712e-18, 3.72598045e-18, 0.0]
                Name: Alice Bloch vector, Length: 927, dtype: object

        Ovvero, "dtype: object".
        Dopo np.stack:

            >>> array([[ 4.56605410e-18,  2.76289721e-18,  0.00000000e+00],
                    [-2.34375000e-02, -8.28641000e-03,  3.17239100e-02],
                    [-2.73437500e-02, -2.76214000e-03, -3.09724000e-03],
                    ...,
                    [ 2.14843700e-02, -8.28641000e-03, -1.90573400e-02],
                    [ 3.47954192e-18,  9.79792330e-19,  0.00000000e+00],
                    [ 6.07169712e-18,  3.72598045e-18,  0.00000000e+00]])

        Se si fa .dtype, si ottiene "float64".
        L'unico problema è che rigenerare un dataframe con:

            >>> df["Alice Bloch vector"] = np.stack(df["Alice Bloch vector"].values)

        Inserisce nella colonna solo la prima componente del vettore, quindi si deve variare un po' il codice.
        '''

        # dunque, np.stack è necessario in quanto nell'addestramento non si può passare una lista di vettori intesi come "oggetti"
        bloch_vector       = np.stack(bloch_vector.values)

        # aggiungi rumore solo se richiesto
        if n_measurements is not None:
            sigma = 1 / np.sqrt(n_measurements)
            bloch_vector += np.random.normal(0, sigma, bloch_vector.shape)

        # genero un dataframe con le 3 componenti del vettore, il cui nome è determinato dall'input "bloch_vector_name"
        return pd.DataFrame(
            bloch_vector,
            columns = [f"{bloch_vector_name}_bloch_vector_{i}" for i in range(1, 4)],
        )

    # genero un dataframe con le 3 componenti di Alice e le 3 componenti di Bob
    alice_bloch_vector_df = _process_bloch_vector(
        bloch_vector = df["Alice Bloch vector"],
        bloch_vector_name = "alice",
        n_measurements = n_measurements
    )
    
    bob_bloch_vector_df   = _process_bloch_vector(
        bloch_vector = df["Bob Bloch vector"],
        bloch_vector_name = "bob",
        n_measurements = n_measurements
    )

    # !IMPORTANT: devo allineare gli indici dei dataframe di Alice e Bob con quelli del dataframe originale per non creare dati NaN (non è possibile farlo a priori)
    alice_bloch_vector_df.index = df.index
    bob_bloch_vector_df.index = df.index

    return pd.concat([
        df.drop(columns = ["Alice Bloch vector", "Bob Bloch vector"]),
        alice_bloch_vector_df,
        bob_bloch_vector_df
    ], axis = 1)

################################################################################################################################################################################################################

# funzione ausiliaria per "process"
def merge_dicts(dicts):

    '''
    Essendo chsh_samples_aggregated_statistics una lista di dizionari, l'uso di .apply(pd.Series) ritorna k colonne (quanto sono gli elementi in lista)
    aventi indici interi. È necessario che le colonne sia le chiavi dei dizionari: quindi, si devono "spacchettare" in un unico dizionario, in modo che siano
    considerabili un unico pd.Series.
    '''
    
    merged = {}
    for d in dicts:
        merged = {**merged, **d}
    return merged

def process(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Funzione ausiliaria che processa il dataframe:
    
    - droppa le colonne
        + contenententi data leakage;
        + informative, quindi non utili
        + `CHSH`, dato che l'addestramento deve essere indipendente da questa feature.

    - rielabora alcune features.

    :param df: Dataframe da estrarre.
    
    :return pd.DataFrame: Dataframe estratto.
    '''

    '''
    1. Si applica la funzione di aggregazione a ogni riga del dataframe;
    2. Poiché l'output "chsh_samples_aggregated_statistics" è una lista di k dizionari, devo spacchettarli e inserirli in un unica struttura dati;
    3. Infine, applico pd.Series per utilizzare le chiavi dei dizionari come nomi delle colonne.
    '''
    
    chsh_samples_aggregated_statistics = df["CHSH samples"].apply(get_chsh_samples_statistics).apply(merge_dicts).apply(pd.Series) 

    data_leakage_columns    = [ # non è possibile conoscere i bit di differenza tra Alice/Bob e Eve
        "Alice and Eve mismatched bits",
        "Bob and Eve mismatched bits"
    ] 
    informative_columns     = [ # non utili ai fini dell'addestramento
        "Coincidence counting",
        "Alice and Bob mismatched bits",
        "QBER",
        #"Early stopping point",

        "Alice key",
        "Bob key",
        
        'Gamma',
        'Fiber optic\'s spectral density',
        'Depolarizing probability',
        'Readout error probability',

        "Backend noise model",
        "Backend version",
        "Backend last update",
        
        "LHS RNG seed",

        "Datetime"
    ]
    confusing_columns = [
        "CHSH samples variance (mode)",
        "CHSH samples kurtosis (mode)",

        "CHSH samples variance (median)",
        "CHSH samples kurtosis (median)",

        "CHSH samples variance (mean)",
        "CHSH samples kurtosis (mean)",

        "CHSH samples mode (variance)",
        "CHSH samples median (variance)",
        "CHSH samples mean (variance)",
        "CHSH samples variance (variance)",
        "CHSH samples skewness (variance)",
        "CHSH samples kurtosis (variance)",

        "CHSH samples mode (skewness)",
        "CHSH samples median (skewness)",
        "CHSH samples mean (skewness)",
        "CHSH samples variance (skewness)",
        "CHSH samples skewness (skewness)",
        "CHSH samples kurtosis (skewness)",

        "CHSH samples mode (kurtosis)",
        "CHSH samples median (kurtosis)",
        "CHSH samples mean (kurtosis)",
        "CHSH samples variance (kurtosis)",
        "CHSH samples skewness (kurtosis)",
        "CHSH samples kurtosis (kurtosis)",
    ]

    df = df[df["CHSH"] <= np.sqrt(2)] # rimuovo le righe anomale
    df = df.drop(columns = ["CHSH", "CHSH samples", *data_leakage_columns, *informative_columns], axis = 1) # posso droppare CHSH samples dopo averlo processato
    df = pd.concat([df, chsh_samples_aggregated_statistics], axis = 1)
    df = df.drop(confusing_columns, axis = 1) # posso droppare le statistiche dei CHSH samples dopo aver generato il dataframe: dropoo quelle che, dopo analisi, risultano essere confusionarie

    return df

################################################################################################################################################################################################################
