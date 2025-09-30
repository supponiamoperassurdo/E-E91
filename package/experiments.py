import os
from datetime import datetime
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.stats.qmc import LatinHypercube, scale
from qiskit_aer import AerSimulator # type: ignore
from .extract import get_anticorrelation_distribution
from .utils import * 
from .E91 import E91 
from .EE91 import EE91
from .classifiers import *
from .noise import *

################################################################################################################################################################################################################

def generate(
        n: int = 1000,
        backend_model: str = None,
        noise: str = None,
        protocol: str = None,
        datatransformer: DataTransformer = None,
        classifiers: list[Classifier] = None,
        seed: int = 42,
        output: str = None,
        params: dict = {"gamma": 0.6, "f": [0.001, 0.1], "p_depolarizing": 0.2, "p_readout_error": [0.01, 0.1], "gamma_epsilon": 0.001, "p_depolarizing_epsilon": 0.001},
    ) -> None:

    '''
    Esegue il protocollo EE91 `n` volte e conserva i risultati di volta in volta in un file CSV. \\
    Se backend_model = None verrà utilizzato un `AerSimulator` che approssima a rotazione uno dei tre processore IBM Quantum nella cartella `/imbq_backends_properties`.
    - `!important`: gli errori di gate sono disabilitati, poiché inducono una forte latenza e **non influiscono sul CHSH** e sul QBER (dunque, è possibile rilassare il problema disabilitandolo);
    - si può visualizzare una forte somiglianza nei dataset con e senza la presenza di tali errori. 

    Nella generazione degli esperimenti, verranno immessi a rotazione casuale 1 dei 6 tipi di errori quantum. Per la selezione della probabilità di errore:
    
    - verranno scelti *range realisitici* che simulano le probabilità di errore in canali di comunicazione degradanti tipicamente utilizzati nel QKD; in particolare, i canali che da approssimare saranno:
        + Canale free-space atmosferico (FSO);
        + Link satellitare (uplink/downlink);
        + Canale in fibra ottica: le simulazioni avverranno tenendo in considerazione questa categoria di canale, gli altri sono più complessi da simulare.
    
    - il rumore dei canali di comunicazione saranno approssimati tramite *combinazioni di rumore quantum classico*, in particolare:
        + [perdita di fotoni](https://www.sciencedirect.com/topics/engineering/latin-hypercube-sampling) (amplitude damping): Free-space, fibra, satellitare: tutti introducono un’attenuazione esponenziale o fading della trasmittanza,
        cioè la perdita di fotoni, che in un qubit dual-rail o single-rail si traduce in amplitude damping. L’amplitude damping channel è infatti il telegrafo qubit del pure-loss bosonic channel,
        limitato allo spazio {|0⟩, |1⟩} tramite un vincolo di energia;
        + [perdita di coerenza o fluttuazioni di fase](https://www.mindspore.cn/mindquantum/docs/en/master/middle_level/noise.html) (phase damping): turbolenza, dispersione, jitter ottico impongono fluttuazioni casuali di fase sul fotone, degradando la coerenza ma non la popolazione, tipico di un phase damping channel 
        + [rumore di fondo](https://en.wikipedia.org/wiki/Quantum_depolarizing_channel) (depolarizzazione): Rumore Raman in fibra, radiazione solare/termica in FSO, e spontaneous emission dai ripetitori sottomarini introducono fotoni casuali e miscele incoerenti di polarizzazioni, assimilabili a un depolarizing channel
        + Errori di polarizzazione (bit-flip o phase-flip): decodifica errata, cross-talk e errori discreti di polarizzazione possono essere descritti come bit-flip (X) o phase-flip (Z) su qubit di polarizzazione
        In particolare, combineremo amplitude dampling, phase dampling e depolarizzazione. Bit flip e phase flip sono ignorabili in quanto "inglobati" nella depolarizzazione. 
           
    - non si utilizzeranno le tecniche Monte-Carlo ma il Latin Hypercube Sampling: questo tipo di campionamento, rispetto a quello classico,
    offre una migliore copertura dello spazio di input con lo stesso numero di punti, evitando [cluster casuali e vuoti](https://analytica.com/blog/latin-hypercube-vs-monte-carlo-sampling/),
    e si basa sulla suddivisione dello spazio di input in M intervalli da cui avverranno le estrazioni casuali (piuttosto che sull'intero spazio). Si intuisce
    che richiede più capacità computazionale, ma un [minor numero di simulazioni](https://www.sciencedirect.com/topics/engineering/latin-hypercube-sampling). L'unica accortezza da tenere
    a mente nell'uso dell'LHS è che, nella generazione dell'ipercubo, è necessario per non invalidare l'esperimento eseguire TUTTI gli esperimenti sugli n punti (rigenerare di volta in volta
    la matrice invalida la distribuzione uniforme a strati di cui gode, e si eseguirebbero sempre i primi k punti se il processo di esecuzione venisse sempre terminato prematuratamente):
    per tale ragione, è possibile - e necessario - salvare in memoria l'ultimo punto di esecuzione della matrice.

    - il traspilatore deve essere impostato con logica di ottimizzazione a 0, in modo che non rimuova i gate id o delay (vedi [qui](https://quantumcomputing.stackexchange.com/questions/26206/fail-to-add-amplitude-damping-error-to-noisemodel)):
    l'inserimento di rumore sui gate ID simula il rumore di canale su fibra ottica. Il gate id è spesso rimosso o "ottimizzato" dal transpiler quando optimization_level > 0, mentre il gate Delay è pensato esplicitamente per modellare gli idle time:
    a seconda dell'implementazione (id o idle), i risultati possono variare.

    :param n: Numero di esperimenti che si vogliono eseguire.
    :param backend_model: File JSON contenente le proprietà di rumore del backend IBM Quantum che si vuole simulare.
    :param noise: Tipo di rumore da simulare. I possibili parametri sono

        - non inserendo alcuna entry, la simulazione sarà ideale (`NoiseModel` vuoto). Valore di default;
        - inserendo `hardware`, la simulazione terrà conto del rumore hardware (escluso il rumore di gate per problemi temporali);
        - inserendo `fiber`, la simulazione terrà conto del rumore di fibra ottica;
        - inserendo `full`, la simulazione terrà conto del rumore sia hardware che di fibra ottica.

    :param protocol: Sceglie il protocollo con cui eseguire esperimenti ("E91" oppure "E-E91"): se non viene specificato un nome di output, il nome verrà deciso in base al protocollo.
    :param output: Nome del file CSV di output.
    :param seed: Seed per il RNG di Numpy, necessario per la generazione del LHS.
    :param params: Dizionario contenente i parametri e gli epsilon per la generazione dell'LHS.

        - gamma (float): Parametro di perdita di fotoni (amplitude damping). Float tra 0 e 1.
        - f (list[float]): Fattore di forma della densità spettrale del canale di fibra ottica (phase damping). Lista di due float (lower and upper bound) tra 0 e 1.
        - p_depolarizing (float): Probabilità di depolarizzazione (depolarizing). Float tra 0 e 1.
        - p_readout_error (list[float]): Probabilità di errore di lettura del fotone (readout error). Lista di due float (lower and upper bound) tra 0 e 1.
        - gamma_epsilon (float): Epsilon per il parametro gamma, per generare un range intorno al valore centrale. Float tra 0 e 1.
        - p_depolarizing_epsilon (float): Epsilon per il parametro p_depolarizing, per generare un range intorno al valore centrale. Float tra 0 e 1.

    :return None:
    '''

    # I nomi dei file con i parametri LHS saranno tutti standard
    lsh_filename = os.path.join("data/lhs", f'lhs_parameters_seed_{seed}.csv')

    # Creo il filepath verso l'output
    if protocol.upper() == "E91":
        folder = "data/training"
        if not os.path.exists(folder):
            os.makedirs(folder) # crea la cartella se non esiste
    elif protocol.upper() == "E-E91":
        folder = "data/experiments"
        if not os.path.exists(folder):
            os.makedirs(folder) # crea la cartella se non esiste
    else:
        raise ValueError("il protocollo per l'esecuzione degli esperimenti può solo essere l'E91 o l'E-E91.")

    # Definisco il path dell'output e della matrice LHS
    output_filename = os.path.join(folder, output)
    lsh_filename = os.path.join("data/lhs", f'lhs_parameters_seed_{seed}.csv')

    # Se il file dell'ipercubo LHS non dovesse esistere, significa che si stanno eseguendo esperimenti con un seed nuovo (a.k.a. con un ipercubo differente): se esiste, non generare nuovi parametri
    if not os.path.exists(lsh_filename):

        '''
        Istanziazione dell'LHS: sulla base dell'ultima osservazione fatta, per evitare che (per esempio) su n = 1000 vengano sempre eseguiti i primi 100-200 a causa di arresti
        volontari o non, e poiché la rigenerazione della matrice LHS non è contemplabile perché in questo modo non si garantirebbe la randomicità promessa (tanto vale utilizzare Monte-Carlo),
        si istanza un seed per l'RNG fissato (es. 42), in modo che fino all'esecuzione dell'esperimento n, la matrice LHS non venga mutata.
        Una volta generata la matrice con quel seed, bisogna mantenere una "memoria" del punto raggiunto.
        '''

        # Limiti inferiori e superiori del LHS
        bounds = {                                        
            # km di fibra ottica: amplitude damping corrispondente a 20 km di fibra e α = 0.16-0.25 dB/km, con un epsilon oscillante     
            'gamma':           [params["gamma"] - params["gamma_epsilon"], params["gamma"] + params["gamma_epsilon"]],

            # Fattore di forma della densità spettrale del canale di fibra ottica: phase damping, di fatto non percettibile
            'f':               [params["f"][0], params["f"][1]],                    

            # Depolarizing: simulazione di condizioni atmosferiche estreme (pioggia intensa, nebbia fitta) o rumore di fondo in fibra ottica
            'p_depolarizing':  [params["p_depolarizing"] - params["p_depolarizing_epsilon"], params["p_depolarizing"] + params["p_depolarizing_epsilon"]], 
            
            # Readout error: errore di lettura del fotone, che può essere causato da rumore di fondo o interferenze; sono valori realistici per un canale di comunicazione degradante
            'p_readout_error': [params["p_readout_error"][0],  params["p_readout_error"][1]],                    
            
            # Presenza di intercettazione: è trattato come float, ma se la soglia è >0.5, si trasforma in True
            'eavesdropper':    [0, 1],
        }

        # Crea il sampler con seed fisso per riproducibilità
        sampler = LatinHypercube(d = len(bounds), rng = np.random.default_rng(seed)) # la riproducibilità è garantita in quanto il seed utilizzato dal RNG è lo stesso
        
        # Genera e scala n campioni
        samples = sampler.random(n = n)
        samples = scale(samples, [r[0] for r in bounds.values()], [r[1] for r in bounds.values()])

        # Converti la sesta colonna (o quinta, contando da 0, 'eavesdropper') in booleana (True se > 0.5)
        samples = samples.astype(object) # un array NumPy non può avere tipi misti nativamente
        samples[:, 4] = (samples[:, 4] > 0.5).astype(bool)

        pd.DataFrame(samples, columns = ['gamma', 'f', 'p_depolarizing', 'p_readout_error', 'eavesdropper']).to_csv(lsh_filename, header = not os.path.exists(lsh_filename), index = False)

    # [#1] carico il file CSV con i parametri
    parameters = pd.read_csv(lsh_filename)
    total      = len(parameters)

    # [!important: #2] carico il checkpoint deterministicamente dal file di checkpoint corretto: in questo modo, se dall'input dovessi cambiare il dataset per la generazione di dati, non riceverò errore
    checkpoint_path = Path(output_filename).with_suffix(".json") # nome del file di checkpoint .json
    if not os.path.exists(checkpoint_path):                      # se non esiste, crealo
        save_last_done(checkpoint_path, 0, seed, n)

    # Carico le informazioni importanti 
    last_done, last_seed, last_n = load_last_done(checkpoint_path)

    # [!important: #3] controllo coerenza seed-numero di esperimenti (validazione input utente)
    if last_seed is None:
        save_last_done(output, 0, seed, n) # significa che questa è la prima esecuzione: inizializza il checkpoint
    elif last_seed != seed or last_n != n:
        raise RuntimeError( # se invece si sta lavorando sullo stesso file con settaggi diversi, lancia un'eccezione
            f"Impostazioni diverse da quelle salvate nel checkpoint ({last_seed = }, {last_n = }) sullo stesso file di esperimenti.\n"
            "Se vuoi cambiare seed o il numero di esperimenti, elimina la matrice LHS utilizzata per l'esperimento o usa un percorso diverso."
        )
        
    # Prompting delle informazioni estratte
    print(f'>>> Parametri dell\'esperimento\n')
    print(f'- numero di campioni della matrica LHS: {n}')
    print(f'- seed LHS: {seed}')
    print(f'- checkpoint esperimento: {last_done + 1}\n')
    print(f'N.B.: per creare nuovo set di esperimenti (non sullo stesso dataset, o li invaliderebbe), bisogna cambiare il seed e il filename,')
    print('altrimenti gli esperimenti si eseguiranno finché la matrice LHS non verrà estinta; o, più brutalmente, si può cancellare la matrice.')

    # Gli esperimenti si faranno sulla totalità dell'ipercubo LHS (che corrispondono al numero di esperimenti), iniziando però dall'ultimo checkpoint
    for i in range (last_done, n):

        try:

            # Sanificazione dell'input
            protocol = protocol.upper()

            # Si istanzia un modello vuoto di rumore
            noise_model = NoiseModel()

            # Impostazioni di rumore HARDWARE
            if noise == "hardware" or noise == "full":
                
                if backend_model is None:
                    # selezione randomica del backend
                    json_folder              = "imbq_backends_properties" # cartella predefinita contenente le proprietà dei backend in formato JSON
                    json_list                = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')] # lista dei file con estensione .json
                    if not json_list:        raise ValueError("nessun file JSON trovato nella cartella specificata.")

                    # istanzio, di volta in volta, il modello di rumore sia per un fattore di casualità, sia per evitare che l'errore di depolarizzazione si accumuli: errori di gate disabilitati
                    backend_info, noise_model    = noise_model_from_json(json_file = np.random.choice(json_list), gate_error = False, warnings = False)
                else:
                    # selezione specifica del backend: se è stato inserito un backend specifico, utilizza quello di input
                    backend_info, noise_model        = noise_model_from_json(json_file = backend_model, gate_error = False, warnings = False)     
            else:
                backend_info = [None, None, None] # per compatibilità con il codice, se non ci sono modelli di rumore, non si inserisce nulla

            # Estraggo i parametri dell'esperimento dall'LHS ed eseguo conversione dei float nella matrice LHS in boolean
            gamma, f, p_depolarizing, p_readout_error, eavesdropper =  parameters.loc[i, ['gamma', 'f', 'p_depolarizing', 'p_readout_error', 'eavesdropper']]

            # Impostazioni di rumore di FIBRA OTTICA
            '''
            Gli errori di amplitude damping, phase damping e depolarizzazione si applicano tutti SOLAMENTE sull'istruzione ID o Delay, poiché rappresenta il canale stesso.
            Se si applicasse su altre istruzione, sarebbe un errore logico e non simulerebbe il canale comunicativo: aggiungere errori a gate come X, H, U1, U2, U3,
            significa modellare imperfezioni nelle operazioni locali di Alice e Bob, non nel canale.
            Questo non è realistico per la maggior parte degli scenari QKD, per esempio:
            
            - un errore su H simula un errore nell'implementazione dell'operatore di Hadamard da parte di Alice o Bob.
            - se Alice o Bob hanno hardware perfetto (ipotesi comune in simulazioni di canale), questi errori non dovrebbero esistere.
            '''
            
            if noise == "fiber" or noise == "full":
                noise_model = fiber_optic_channel(
                    noise_model = noise_model,
                    gamma = gamma,
                    f = f,
                    p_depolarizing = p_depolarizing,
                    p_readout_error = p_readout_error
                )
            else:
                gamma, f, p_depolarizing, p_readout_error = None, None, None, None # per questione di "compatibilità" e leggibilità sul file CSV, ma si potrebbe lasciare così com'è
            
            # Impostazioni del BACKEND
            '''            
            - !important: per mantenere stabilità sulla RAM e non far crashare il kernel Python dovuto alla pesantezza delle simulazioni con gli errori di amplitude e phase dampling,
            è necessario impostare method = 'density_matrix'.
                + si è osservato un calo di uso di CPU dal 90% (che provoca un crash) al 15%;
                + al più, gli unici errori osservati non sono segmentation fault, ma eccezioni non fatali di tipo NoiseError che non arrestano l'esecuzione.
            - le altre opzioni sono meno rilevanti;
            - per ottimizzare la dimensionalità della matrice di Kraus, phase_amplitude_damping_error riassume i due tipi di errori.
            '''

            backend = AerSimulator(
                noise_model = noise_model,
                method = 'density_matrix',
            )

            # Esecuzione dell'esperimento
            if protocol                  == "E91":
                results                  = E91(
                    number_of_singlets = 512,
                    experiment_number = i + 1,
                    backend = backend,
                    eavesdropper = eavesdropper,
                    verbose = False,
                ).run()
                
            elif protocol                == "E-E91":
                results                  = EE91(
                    number_of_singlets = 512,
                    experiment_number = i + 1,
                    datatransformer = datatransformer,
                    classifiers = classifiers,
                    backend = backend,
                    eavesdropper = eavesdropper, 
                    verbose = False
                ).run()

            # Estrazione di altri parametri dell'esperimento
            anticorrelation_distribution = get_anticorrelation_distribution(chsh_samples = results["chsh_samples"])
            experiment_time              = datetime.now().astimezone(tz = datetime.now().astimezone().tzinfo)

            if protocol                  == "E91":

                # Inserimento dei risultati nel dataframe temporaneo
                df = pd.DataFrame({

                    # Target
                    'Eavesdropper':                    [eavesdropper], 

                    # "Target" di ricostruzione per l'auto-encoder
                    'Alice key':                       [''.join(results['alice_key'])],
                    'Bob key':                         [''.join(results['bob_key'])],

                    # Info
                    'CHSH':                            [results["chsh"]],
                    'Alice and Bob mismatched bits':   [results['alice_and_bob_mismatched_key_bits']],
                    'QBER':                            [results["qber"]],
                    'Coincidence counting':            [results["coincidence_counting"]],
                    
                    # Data leakage
                    'Alice and Eve mismatched bits':   [results['alice_and_eve_mismatched_key_bits']],
                    'Bob and Eve mismatched bits':     [results['bob_and_eve_mismatched_key_bits']],

                    # Features utili, ottenibili in qualunque momento dell'esperimento
                    "CHSH samples":                    [results["chsh_samples"]],

                    "(X, W) anticorrelation":          [anticorrelation_distribution[('X', 'W')]],
                    "(X, V) anticorrelation":          [anticorrelation_distribution[('X', 'V')]],
                    "(Z, W) anticorrelation":          [anticorrelation_distribution[('Z', 'W')]],
                    "(Z, V) anticorrelation":          [anticorrelation_distribution[('Z', 'V')]],

                    # Info
                    'Gamma':                           [gamma],
                    'Fiber optic\'s spectral density': [f],
                    'Depolarizing probability':        [p_depolarizing],
                    'Readout error probability':       [p_readout_error],

                    # Backend info
                    'Backend noise model':             [backend_info[0]],
                    'Backend version':                 [backend_info[1]],
                    'Backend last update':             [backend_info[2]],
                        
                    # Seed
                    'LHS RNG seed':                    [seed],

                    # Datetime
                    'Datetime':                        [experiment_time],
                })

            elif protocol                == "E-E91":

                # Inserimento dei risultati nel dataframe temporaneo
                df = pd.DataFrame({

                    # Target
                    'Eavesdropper':                    [eavesdropper], 
                    'Predict':                         [results["eavesdropper_predict"]],
                    'Predict_Proba':                   [results["eavesdropper_predict_proba"]],
                    'Early stopping point':            [results['early_stopping_point']],

                    # Info
                    'Alice key':                       [''.join(results['alice_key'])],
                    'Bob key':                         [''.join(results['bob_key'])],
                    
                    # Info
                    'CHSH':                            [results["chsh"]],
                    'Alice and Bob mismatched bits':   [results['alice_and_bob_mismatched_key_bits']],
                    'QBER':                            [results["qber"]],
                    'Coincidence counting':            [results["coincidence_counting"]],
                    
                    # Data leakage
                    'Alice and Eve mismatched bits':   [results['alice_and_eve_mismatched_key_bits']],
                    'Bob and Eve mismatched bits':     [results['bob_and_eve_mismatched_key_bits']],

                    # Features utili, ottenibili in qualunque momento dell'esperimento
                    "CHSH samples":                    [results["chsh_samples"]],

                    "(X, W) anticorrelation":          [anticorrelation_distribution[('X', 'W')]],
                    "(X, V) anticorrelation":          [anticorrelation_distribution[('X', 'V')]],
                    "(Z, W) anticorrelation":          [anticorrelation_distribution[('Z', 'W')]],
                    "(Z, V) anticorrelation":          [anticorrelation_distribution[('Z', 'V')]],

                    # Info
                    'Gamma':                           [gamma],
                    'Fiber optic\'s spectral density': [f],
                    'Depolarizing probability':        [p_depolarizing],
                    'Readout error probability':       [p_readout_error],

                    # Backend info
                    'Backend noise model':             [backend_info[0]],
                    'Backend version':                 [backend_info[1]],
                    'Backend last update':             [backend_info[2]],
                    
                    # Seed
                    'LHS RNG seed':                    [seed],

                    # Datetime
                    'Datetime':                        [experiment_time],
                })

            # Si sostituiscono i False e True con 0 e 1 per renderli trattabili
            df["Eavesdropper"]           = df["Eavesdropper"].replace({False: 0, True: 1})

            # Si salva in CSV in modalità append, senza header dopo la prima scrittura
            df.to_csv(output_filename, mode = 'a', header = not os.path.exists(output_filename), index = False) # il "not" verifica se il file esiste già, in modo da non generare i nomi delle colonne a ogni esecuzione

            # [!important: #4] aggiorno il checkpoint 
            save_last_done(checkpoint_path, i + 1, seed, n)

        except Exception as e: print(f" | Esperimento arrestato a causa dell'errore {e.__class__.__name__}: si proseguirà coi successivi: {e}\n")

################################################################################################################################################################################################################
