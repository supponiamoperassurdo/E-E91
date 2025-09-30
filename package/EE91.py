import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
from .classifiers import *
from .E91 import *
from .extract import *

##################################################################################################################################################################################################

warnings.filterwarnings( # questo warning si presenta perché alla matrice di confusione non vengono passate delle label e inoltre non vengono passati abbastanza dati (solo uno)
    "ignore",
    message = "A single label was found in 'y_true' and 'y_pred'",
    category = UserWarning
)

warnings.filterwarnings( # a volte potrebbe verificarsi una "cancellazione catastrofica" su dataset a basso regime (vedi: https://en.wikipedia.org/wiki/Catastrophic_cancellation)
    "ignore",
    message = "Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.",
    category = RuntimeWarning
)

warnings.filterwarnings( # fit con dataframe dove non si è fatto X = X.values, ma i dati sono corretti
    "ignore",
    message = "X does not have valid feature names, but",
    category = RuntimeWarning
)

##################################################################################################################################################################################################

class EE91(E91):

    '''
    # Enhanced-E91

    Protocollo E91 potenziato dal machine learning.
    
    - analizza passo passo la presenza di un intercettatore;
    - se l'esito è negativo, il canale verrà classificato come "non intercettato" e l'ultimo esito spetterà al test di Bell;
    - per quanto irrisorio il rischio matematico - non sperimentale - di falsi negativi:
        + nel mondo reale, le intercettazioni sono rare;
        + se anche dovessero accadere, a fine protocollo (qui non implementato) avverrà sempre QEC (N.B.: si osservi che la chiave ricavata da Eve sarà corrotta come quella di Alice e Bob, e dopo la correzione di Alice e Bob sarà diversa da quella di quest'ultima) e privacy amplification (rimuove ulteriori informazioni che Eve potrebbe avere);
        + poiché error correction e privacy amplification avvengono sempre nell'E91, l'utilità del machine learning sta nell'interruzione preventiva della generazione della chiave in caso di rilevamento di un'intrusione;
        + in generale, però, il sistema proposto non presenta falsi negativi.
    
    Per riuscire a ottenere zero falsi negativi, sono necessari:
    
    - classificatori con recall superiore a 0.95;
    - un numero di classificatori dispari;
    - che la "natura" di questi sia differente tra loro per bilanciare eventuali scelte errate eseguite da un'altro classificatore (es. RandomForest contro una rete neurale);
    - che classifiers_decision_threshold_proba sia impostata a 0.85, soglia individuata sperimentalmente.

    Ovviamente, data la sicurezza intrinseca del protocollo E91, non è necessario raggiungere un recall così alto: il protocollo E91 è già sicuro di per sé,
    e il machine learning serve solo a prevenire l'interruzione della generazione della chiave in caso di intercettazione.
    
    Alcune supposizioni sono necessarie per il protocollo EE91, identiche all'E91:

    - lo stato entangled di Bell viene preparato da una "terza parte fidata" Charlie;
    - l'attaccante interviene durante l'esecuzione del codice, non parallelamente;
    - il test di Bell può essere eseguito solamente avendo accesso mutuale alle misure di Alice e Bob sulle basi non compatibili:
        * essendo queste comunicate su un canale pubblico, è essenziale che, per prevenire la falsificazione delle misure da parte di Eve, la comunicazione di queste avvenga in un canale AUTENTICATO;
        * l'autenticazione avviene tipicamente con MAC, quindi è necessaria una chiave pre-condivisa tra Charlie-Alice e Charlie-Bob.

    Ritorna un dizionario contenente i valori elencanti in "Returns". \\
    P.S.: si è osservato che:
    
    - sulla base di una [issue su GitHub](https://github.com/Qiskit/qiskit-aer/issues/1997), nell'uso di un simulatore Aer conviene sempre impostare "optimization_level = 0" del traspilatore per evitare crash,.
    - configuare il simulatore Aer con "method = 'density_matrix'": in questo modo, le simulazioni degli stati avverranno solamente con le matrici di densità, e non savraccaricherano il kernel Python,

    :param number_of_singlets: Numero di singoletti (stati di Bell) con cui si vuole generare la chiave. Si osservi che solo il 22%, più precisamente 2/9, di essi è utile alla generazione della chiave.
    :param experiment_number: Valore opzionale utilizzato in una funzione secondaria, da ignorare in quanto utilizzato da un altro modulo.
    :param datatransformer: Oggetto utilizzato per trasformare i dati tramite imputatore e scalatore ai fini della classificazione, appartenente al modulo `package.classifiers`: ingloba elementi appartenenti al modulo `sklearn`, caricabili in formato pickle `.pkl` oppure passati come oggetti già addestrati.
    :param classifiers: Lista di oggetti utilizzati per la classificazione del canale di comunicazione, appartenente al modulo `package.classifiers`.
    :param backend: ([`qiskit.provider`](https://docs.quantum.ibm.com/api/qiskit/providers#backend) oppure [`qiskit_aer.AerSimulator`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html#qiskit_aer.AerSimulator)) backend IBM Quantum con cui si vuole eseguire l'esperimento. Se non specificato, di default si utilizzerà `AerSimulator()`.
    :param eavesdropper: Abilita la simulazione di un intercettatore Eve.
    :param progress: Mostra la percentuale di progresso dell'esperimento.
    :param verbose: Se abilitato, a fine esecuzione descrive tutti i risultati ottenuti.
    
    :return evs: Valori attesi del test di Bell.
    :return chsh: Parametro del test CHSH.
    :return chsh_samples: Campioni estratti per il test di Bell.
    :return alice_key: Chiave di Alice (N.B.: non è formattata come stringa).
    :return bob_key: Chiave di Bob.
    :return eve_key_from_alice: Chiave di Eve, l'intercettatore, ottenuta da Alice.
    :return eve_key_from_bob: Chiave di Eve, l'intercettatore, ottenuta da Bob.
    :return alice_and_bob_compatible_bases_frequencies: Frequenza di basi compatibili tra Alice e Bob.
    :return alice_and_bob_mismatched_key_bits: Bit di chiave differenti tra Alice e Bob.
    :return qber: Quantum Bit Error Rate, tasso di errore del qubit, definito come il rapporto tra il numero di bit differenti tra Alice e Bob e la lunghezza di chiave (ovvero, il numero di qubit che si è riusciti a trasmettere).
    :return coincidence_counting: [Coincidence counting](https://en.wikipedia.org/wiki/Coincidence_counting_%28physics%29), consiste (a livello fisico) nel far passare i pulse di due fotodiodi entro una finestra Δt in un circuito elettronico dedicato: Aer fa già il lavoro sporco. Conta il numero di "coincidenze" tra due particelle entangled.
    :return alice_and_eve_mismatched_key_bits: Bit di chiave differenti tra Alice e Eve.
    :return bob_and_eve_mismatched_key_bits: Bit di chiave differenti tra Bob e Eve.
    :return eavesdropper_predict: Predizione sull'intercettatore.
    '''

##################################################################################################################################################################################################

    def __init__(self,
            number_of_singlets: int = 512, 
            experiment_number: int = None, 
            datatransformer: DataTransformer = None, 
            classifiers: list[Classifier] = None, 
            backend: any = None, 
            eavesdropper: bool = False, 
            progress: bool = True, 
            verbose: bool = False
        ) -> None:
        
        '''
        :param number_of_singlets: Numero di singoletti (stati di Bell) con cui si vuole generare la chiave. Si osservi che solo il 22%, più precisamente 2/9, di essi è utile alla generazione della chiave.
        :param experiment_number: Valore opzionale utilizzato in una funzione secondaria, da ignorare in quanto utilizzato da un altro modulo.
        :param datatransformer: Oggetto utilizzato per trasformare i dati tramite imputatore e scalatore ai fini della classificazione, appartenente al modulo `package.classifiers`: ingloba elementi appartenenti al modulo `sklearn`, caricabili in formato pickle `.pkl` oppure passati come oggetti già addestrati.
        :param classifiers: Lista di oggetti utilizzati per la classificazione del canale di comunicazione, appartenente al modulo `package.classifiers`.
        :param backend: ([`qiskit.provider`](https://docs.quantum.ibm.com/api/qiskit/providers#backend) oppure [`qiskit_aer.AerSimulator`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html#qiskit_aer.AerSimulator)) backend IBM Quantum con cui si vuole eseguire l'esperimento. Se non specificato, di default si utilizzerà `AerSimulator()`.
        :param eavesdropper: Abilita la simulazione di un intercettatore Eve.
        :param progress: Mostra la percentuale di progresso dell'esperimento.
        :param verbose: Se abilitato, a fine esecuzione descrive tutti i risultati ottenuti.
        '''

        super().__init__(
            number_of_singlets = number_of_singlets,
            experiment_number = experiment_number,
            backend = backend,
            eavesdropper = eavesdropper,
            progress = progress,
            verbose = verbose
        )

        # Inizializzazione degli attributi di machine learning
        self.datatransformer = datatransformer
        self.classifiers = classifiers # lista di classificatori
        
        if self.classifiers is None:
            raise ValueError("l'E-E91 richiede una lista di classificatori.")
        
        if self.verbose:
            print("\n>>> Lista dei classificatori utilizzati\n")
            for classifier in self.classifiers:
                print(f"- {classifier}")

        # Lista delle "fraction" in cui si esegue la classificazione, "weight" è il peso nella votazione e "active" è il flag che stabilisce se il classificatore ha votato o meno in quel fraction
        self.thresholds = self._create_thresholds(n = 10, k = 1.5, s = 0.65, verbose = verbose)

        # Soglia di decisione standard per i classificatori con cui si eseguirà majority voting
        self.classifiers_decision_threshold_proba = 0.5
        
        # È ponderata sulla base della quantum fidelity: individuata sperimentalmente
        self.ensemble_decision_threshold = 0.38
        self.weighted_votes = 0

        # Flag utilizzato per arrestare la sessione in caso di rilevamento di un'intrusione
        self.halt = False                    

        # Lista delle "anti-fidelity" dei singoletti
        self.singlet_fidelities = []

        # Risultati del protocollo: indice di performance, ci dice a che punto si arresta il protocollo in termini di risparmio computazionale
        self.results["early_stopping_point"] = None

    def _create_thresholds(self, n: int = 10, k: float = 1.5, s: float = 0.65, verbose: bool = False):
        '''
        Funzione ausiliaria che definisce i checkpoint in cui eseguire la classificazione.
        Ogni checkpoint ha una "fraction" (soglia), un "peso" normalizzato e un flag "active".
        
        :param n: Numero di frazioni di classificazione.
        :param k: Parametro che regola la curva delle frazioni (più è grande, più si espone verso destra).
        :param weight_mode: Modalità di calcolo dei pesi. Con "linear", i pesi sono proporzionali a i. Con "power" sono proporzionali a i^p. Con "exp", pesi esponenziali exp(alpha * i). Con "fraction", pesi proporzionali alla stessa curva delle frazioni.
        :param p: Esponente usato se weight_mode = "power".
        :param alpha: Parametro usato se weight_mode = "exp".
        :param verbose: Se True, stampa le soglie e i pesi.
        
        :return thresholds: Lista di dizionari {fraction, weight, active}.
        '''
        
        thresholds = []
        progress_list = [i / n for i in range(1, n + 1)]  # progresso lineare

        # Calcolo delle fractions
        fractions = [round(1 - (1 - prog) ** k, 4) for prog in progress_list]

        # Mix tra forward (i, ...) e reverse (n + 1 - i, ...)
        forward = [i for i in range(1, n + 1)]
        reverse = [(n + 1 - i) for i in range(1, n + 1)]
        s = float(s) # serve a bilanciare forward e reverse: tra 0.6 e 0.7 è un ottimo valore
        raw = [(1 - s) * f + s * r for f, r in zip(forward, reverse)]
       
        # Normalizzazione
        total = sum(raw)
        weights = [w / total for w in raw][::-1]

        # Costruzione della lista
        for fraction, weight in zip(fractions, weights):
            thresholds.append(
                {
                    "fraction": fraction,
                    "weight": round(weight, 4),
                    "active": True
                }
            )

        if verbose:
            print("\n>>> Threshold di classificazione\n")
            for threshold in thresholds:
                print(f"Threshold: {threshold['fraction']:.2f}\t\tWeight: {threshold['weight']:.2f}")
            print("")

        return thresholds

    def predict(self) -> int:

        '''
        Funzione di classificazione del protocollo Enhanced-E91 finalizzata all'individuazione di un'intrusione.

        :param None:

        :return predict, predict_proba: Ritorna la predizione e la probabilità di predizione sull'intercettatore.
        '''

        def __bootstrap_resample(chsh_samples: dict, noise_scale: float = 0.01):

            '''
            Esegue il boostrapping del dataset.

            :param chsh_samples: Campioni estratti per il test di Bell.
            :param noise_scale: Scala del rumore gaussiano da aggiungere ai campioni. Il rumore viene aggiunto in modo proporzionale alla deviazione standard dei campioni.

            :return resampled: Ritorna un dizionario con i campioni risampled.
            '''

            resampled = {}
            for key, values in chsh_samples.items():

                # Step 1: bootstrap, ovvero campionamento con reinserimento
                resampled_values = np.random.choice(values, size = len(values), replace = True)
                
                ''' N.B.: induceva solo disturbo nella classificazione
                # Step 2: aggiunta di rumore gaussiano proporzionale alla deviazione standard
                if len(resampled_values) > 0:
                    noise = np.random.normal(loc = 0, scale = noise_scale * np.std(resampled_values), size = len(resampled_values))
                    resampled_values = resampled_values + noise
                '''
                
                resampled[key] = resampled_values.tolist()
            
            return resampled

        classifier_predicts, classifier_predict_probas = [], []

        for classifier in self.classifiers:

            # Estrazione dei parametri dell'esperimento
            chsh_samples                 = __bootstrap_resample(chsh_samples = self.results["chsh_samples"])
            anticorrelation_distribution = get_anticorrelation_distribution(chsh_samples = chsh_samples)

            # Inserimento dei risultati nel dataframe temporaneo
            df = pd.DataFrame({
                "Eavesdropper":                  [self.eavesdropper], # colonna target
                
                "CHSH samples":                  [chsh_samples],

                "(X, W) anticorrelation":        [anticorrelation_distribution[('X', 'W')]],
                "(X, V) anticorrelation":        [anticorrelation_distribution[('X', 'V')]],
                "(Z, W) anticorrelation":        [anticorrelation_distribution[('Z', 'W')]],
                "(Z, V) anticorrelation":        [anticorrelation_distribution[('Z', 'V')]],
            })

            # Manipolazione dei campioni CHSH per ottenere le statistiche
            chsh_samples_aggregated_statistics = df["CHSH samples"].apply(get_chsh_samples_statistics).apply(merge_dicts).apply(pd.Series) 
            df = df.drop(columns = ["CHSH samples"], axis = 1) # posso droppare le statistiche dei CHSH samples dopo aver generato il dataframe: droppo quelle che, dopo analisi, risultano essere confusionarie

            # Ora ho lo stesso dataframe di addestramento, e droppo le colonne "confusionarie" per i classificatori
            df = pd.concat([df, chsh_samples_aggregated_statistics], axis = 1)
            df = df.drop(
                columns = [ # vedi in "extract.process" per il significato di queste colonne
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
                ], 
                axis = 1
            )

            # Trasformo il dataframe con il datatransformer
            X  = df.drop(columns = ["Eavesdropper"], axis = 1)
            X  = self.datatransformer.transform(X)

            # Eseguo la predizione sul dato X
            classifier_predict       = int(classifier.predict(X = X)[0])
            classifier_predict_proba = classifier.predict_proba(X = X)[0]

            classifier_predicts.append(classifier_predict)
            classifier_predict_probas.append(classifier_predict_proba)

        # Se si sta utilizzando un solo classificatore
        if len(self.classifiers) == 1: 

            # La predizione non è un problema di majority voting
            predict_proba = classifier_predict_probas[0]
            predict = classifier_predicts[0]

            if self.verbose: print(f" | Predict probas: [{predict_proba[0]:.4f}, {predict_proba[1]:.4f}] | Predict: {predict}", end = "")

        # Se si vuole utilizzare almeno più di un classificatore...
        else:
    
            # Probabilità media valutata sulle probabilità valutate dei classificatori: il primo elemento è la probabilità che sia classe 0 (non intercettazione)
            predict_proba = np.mean(classifier_predict_probas, axis = 0)

            # Stampa delle probabilità e delle predizioni dei classificatori
            if self.verbose: print(f" | Predict probas: [{predict_proba[0]:.4f}, {predict_proba[1]:.4f}] | Predicts: {classifier_predicts}", end = "")
        
            # Se non si raggiunge la maggioranza in hard voting (dunque c'è parità), si procede in soft voting (cioè con le probabilità)
            if sum(classifier_predicts) < len(classifier_predicts) // 2 and predict_proba[0] >= self.classifiers_decision_threshold_proba:
                predict = 0
            else:
                predict = 1

        return predict, predict_proba

##################################################################################################################################################################################################

    def run(self) -> dict:
        
        '''
        Esegue il protocollo.

        :return evs: Valori attesi del test di Bell.
        :return chsh: Parametro del test CHSH.
        :return chsh_samples: Campioni estratti per il test di Bell.
        :return alice_key: Chiave di Alice (N.B.: non è formattata come stringa).
        :return bob_key: Chiave di Bob.
        :return eve_key_from_alice: Chiave di Eve, l'intercettatore, ottenuta da Alice.
        :return eve_key_from_bob: Chiave di Eve, l'intercettatore, ottenuta da Bob.
        :return alice_and_bob_compatible_bases_frequencies: Frequenza di basi compatibili tra Alice e Bob.
        :return alice_and_bob_mismatched_key_bits: Bit di chiave differenti tra Alice e Bob.
        :return qber: Quantum Bit Error Rate, tasso di errore del qubit, definito come il rapporto tra il numero di bit differenti tra Alice e Bob e la lunghezza di chiave (ovvero, il numero di qubit che si è riusciti a trasmettere).
        :return coincidence_counting: [Coincidence counting](https://en.wikipedia.org/wiki/Coincidence_counting_%28physics%29), consiste (a livello fisico) nel far passare i pulse di due fotodiodi entro una finestra Δt in un circuito elettronico dedicato: Aer fa già il lavoro sporco. Conta il numero di "coincidenze" tra due particelle entangled.
        :return alice_and_eve_mismatched_key_bits: Bit di chiave differenti tra Alice e Eve.
        :return bob_and_eve_mismatched_key_bits: Bit di chiave differenti tra Bob e Eve.
        :return eavesdropper_predict: Predizione sull'intercettatore.
        '''

        for i in range(self.number_of_singlets):

            # Esegue il corpo del protocollo E91, che si occupa di generare gli stati di Bell, le misure e le basi
            super(EE91, self)._E91__body(idx = i)

            '''
            Il rilevamento di un intrusione al k% avviene per maggioranza e si esegue il Quantum Weighted Voting
            Vengono utilizzati k "differenti" dataset, che fondamentalmente è
            sempre lo stesso self.results["chsh_samples"] ma con più campioni.
            
            - banalmente, quanto più sono i campioni in self.results["chsh_samples"], quanto più saranno precisi e raffinati i risultati delle funzioni "get_chsh_samples_statistics" e "get_anticorrelation_distribution"
            - quindi, emula un po' il concetto di "votazione" dei metodi ensemble, seppur non ne rispecchi le tecniche principali;
            - l'idea è di salvare tutte le votazioni e a metà esperimento eseguire un early stopping se viene rilevata un'intrusione: altrimenti, si esegue la classificazione finale.
            
            Verrà inoltre aggiunto del rumore a ogni sotto-dataset per renderlo differente dall'altro: ciò renderà le predizioni più indipendenti. 
            '''

            # Se si è raggiunta la frazione di classificazione, esegui la predizione e salva il voto per valutare la presenza di intercettazione
            for threshold in self.thresholds:
                
                # ...se si è superata la frazione corrispondente (es. 20%) e il flag è attivo (il flag dice se con la soglia del 20% puoi entrare nell'if: se è True, puoi entrare)
                if i / self.number_of_singlets > threshold["fraction"] and threshold["active"]:

                    # Disattiva il flag per questa soglia
                    threshold["active"] =  False 

                    # Predizione del voto
                    votes = self.predict()[0]

                    # Aggiungi il voto ponderato
                    self.weighted_votes += threshold["weight"] * votes

                    if self.verbose: print(f" | Weight: {threshold['weight']:.2f} | Weighted votes: {self.weighted_votes:.4f}")

            # Se i voti ponderati raggiungono (o superano) la soglia di decisione, si arresta la sessione
            if self.weighted_votes >= self.ensemble_decision_threshold:

                # La maggioranza stabilisce che c'è stata un'intercettazione
                self.results["eavesdropper_predict"] = 1

                # Le predict_proba utile da salvare in CSV è solo quella della classificazione finale, quella nel caso in cui i classificatori non rilevino nulla fino all'ultimo step
                self.results["eavesdropper_predict_proba"] = None # segno "None" per indicare che i classificatori hanno rilevato un'intrusione
                self.halt = True

                # Punto in cui è stato arrestato il protocollo espresso in termini percentuali: servirà per valutare le performance
                self.results["early_stopping_point"] = i / self.number_of_singlets

                # Verbosità
                if self.verbose: print("Halt! Rilevata potenziale intrusione.")
                    
                # Arresto la sessione
                break 

        # Classificazione finale con ulteriori informazioni solo se non c'è stato un halt (arresto precoce del protocollo)
        if not self.halt:

            predict, predict_proba                     = self.predict()
            self.results['eavesdropper_predict']       = predict
            self.results['eavesdropper_predict_proba'] = predict_proba

            if self.verbose:
                print(f"Predizione finale: ", end = "")
                if self.results['eavesdropper_predict'] == 1:
                    print(f"✗ presenza di un intercettatore al {(self.results['eavesdropper_predict_proba'][1] * 100):.2f}%")
                else:
                    print(f"✓ canale sicuro al {(self.results['eavesdropper_predict_proba'][0] * 100):.2f}%")

        # Calcolo dei parametri finali
        self.results['evs'], self.results['chsh'] = self.compute_chsh(chsh_samples = self.results["chsh_samples"])
        self.results['qber']                      = self.results['alice_and_bob_mismatched_key_bits'] / len(self.results['alice_key'])

        # Verbose dei risultati finali
        if self.verbose:
            self._prompt_results(
                evs = self.results['evs'],
                chsh = self.results['chsh'],
                chsh_samples = self.results['chsh_samples'],
                alice_key = self.results['alice_key'],
                bob_key = self.results['bob_key'],
                eve_key_from_alice = self.results['eve_key_from_alice'],
                eve_key_from_bob = self.results['eve_key_from_bob'],
                alice_and_bob_compatible_bases_frequencies = self.results['alice_and_bob_compatible_bases_frequencies'],
                alice_and_bob_mismatched_key_bits = self.results['alice_and_bob_mismatched_key_bits'],
                qber = self.results['qber'],
                coincidence_counting = self.results['coincidence_counting'],
                alice_and_eve_mismatched_key_bits = self.results['alice_and_eve_mismatched_key_bits'],
                bob_and_eve_mismatched_key_bits = self.results['bob_and_eve_mismatched_key_bits']
            )

        return self.results

############################################################################################################################################################################################################

"""

class EE91_no_fidelity(E91):

    '''
    # Enhanced-E91

    Protocollo E91 potenziato dal machine learning.
    
    - analizza passo passo la presenza di un intercettatore;
    - se l'esito è negativo, il canale verrà classificato come "non intercettato" e l'ultimo esito spetterà al test di Bell;
    - per quanto irrisorio il rischio matematico - non sperimentale - di falsi negativi:
        + nel mondo reale, le intercettazioni sono rare;
        + se anche dovessero accadere, a fine protocollo (qui non implementato) avverrà sempre QEC (N.B.: si osservi che la chiave ricavata da Eve sarà corrotta come quella di Alice e Bob, e dopo la correzione di Alice e Bob sarà diversa da quella di quest'ultima) e privacy amplification (rimuove ulteriori informazioni che Eve potrebbe avere);
        + poiché error correction e privacy amplification avvengono sempre nell'E91, l'utilità del machine learning sta nell'interruzione preventiva della generazione della chiave in caso di rilevamento di un'intrusione;
        + in generale, però, il sistema proposto non presenta falsi negativi.
    
    Per riuscire a ottenere zero falsi negativi, sono necessari:
    
    - classificatori con recall superiore a 0.95;
    - un numero di classificatori dispari;
    - che la "natura" di questi sia differente tra loro per bilanciare eventuali scelte errate eseguite da un'altro classificatore (es. RandomForest contro una rete neurale);
    - che decision_threshold_proba sia impostata a 0.85, soglia individuata sperimentalmente.

    Ovviamente, data la sicurezza intrinseca del protocollo E91, non è necessario raggiungere un recall così alto: il protocollo E91 è già sicuro di per sé,
    e il machine learning serve solo a prevenire l'interruzione della generazione della chiave in caso di intercettazione.
    
    Alcune supposizioni sono necessarie per il protocollo EE91, identiche all'E91:

    - lo stato entangled di Bell viene preparato da una "terza parte fidata" Charlie;
    - l'attaccante interviene durante l'esecuzione del codice, non parallelamente;
    - il test di Bell può essere eseguito solamente avendo accesso mutuale alle misure di Alice e Bob sulle basi non compatibili:
        * essendo queste comunicate su un canale pubblico, è essenziale che, per prevenire la falsificazione delle misure da parte di Eve, la comunicazione di queste avvenga in un canale AUTENTICATO;
        * l'autenticazione avviene tipicamente con MAC, quindi è necessaria una chiave pre-condivisa tra Charlie-Alice e Charlie-Bob.

    Ritorna un dizionario contenente i valori elencanti in "Returns". \\
    P.S.: si è osservato che:
    
    - sulla base di una [issue su GitHub](https://github.com/Qiskit/qiskit-aer/issues/1997), nell'uso di un simulatore Aer conviene sempre impostare "optimization_level = 0" del traspilatore per evitare crash,.
    - configuare il simulatore Aer con "method = 'density_matrix'": in questo modo, le simulazioni degli stati avverranno solamente con le matrici di densità, e non savraccaricherano il kernel Python,

    :param number_of_singlets: Numero di singoletti (stati di Bell) con cui si vuole generare la chiave. Si osservi che solo il 22%, più precisamente 2/9, di essi è utile alla generazione della chiave.
    :param experiment_number: Valore opzionale utilizzato in una funzione secondaria, da ignorare in quanto utilizzato da un altro modulo.
    :param datatransformer: Oggetto utilizzato per trasformare i dati tramite imputatore e scalatore ai fini della classificazione, appartenente al modulo `package.classifiers`: ingloba elementi appartenenti al modulo `sklearn`, caricabili in formato pickle `.pkl` oppure passati come oggetti già addestrati.
    :param classifiers: Lista di oggetti utilizzati per la classificazione del canale di comunicazione, appartenente al modulo `package.classifiers`.
    :param backend: ([`qiskit.provider`](https://docs.quantum.ibm.com/api/qiskit/providers#backend) oppure [`qiskit_aer.AerSimulator`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html#qiskit_aer.AerSimulator)) backend IBM Quantum con cui si vuole eseguire l'esperimento. Se non specificato, di default si utilizzerà `AerSimulator()`.
    :param eavesdropper: Abilita la simulazione di un intercettatore Eve.
    :param progress: Mostra la percentuale di progresso dell'esperimento.
    :param verbose: Se abilitato, a fine esecuzione descrive tutti i risultati ottenuti.
    
    :return evs: Valori attesi del test di Bell.
    :return chsh: Parametro del test CHSH.
    :return chsh_samples: Campioni estratti per il test di Bell.
    :return alice_key: Chiave di Alice (N.B.: non è formattata come stringa).
    :return bob_key: Chiave di Bob.
    :return eve_key_from_alice: Chiave di Eve, l'intercettatore, ottenuta da Alice.
    :return eve_key_from_bob: Chiave di Eve, l'intercettatore, ottenuta da Bob.
    :return alice_and_bob_compatible_bases_frequencies: Frequenza di basi compatibili tra Alice e Bob.
    :return alice_and_bob_mismatched_key_bits: Bit di chiave differenti tra Alice e Bob.
    :return qber: Quantum Bit Error Rate, tasso di errore del qubit, definito come il rapporto tra il numero di bit differenti tra Alice e Bob e la lunghezza di chiave (ovvero, il numero di qubit che si è riusciti a trasmettere).
    :return coincidence_counting: [Coincidence counting](https://en.wikipedia.org/wiki/Coincidence_counting_%28physics%29), consiste (a livello fisico) nel far passare i pulse di due fotodiodi entro una finestra Δt in un circuito elettronico dedicato: Aer fa già il lavoro sporco. Conta il numero di "coincidenze" tra due particelle entangled.
    :return alice_and_eve_mismatched_key_bits: Bit di chiave differenti tra Alice e Eve.
    :return bob_and_eve_mismatched_key_bits: Bit di chiave differenti tra Bob e Eve.
    :return eavesdropper_predict: Predizione sull'intercettatore.
    '''

##################################################################################################################################################################################################

    def __init__(self,
            number_of_singlets: int = 512, 
            experiment_number: int = None, 
            datatransformer: DataTransformer = None, 
            classifiers: list[Classifier] = None, 
            backend: any = None, 
            eavesdropper: bool = False, 
            progress: bool = True, 
            verbose: bool = False
        ) -> None:
        
        '''
        :param number_of_singlets: Numero di singoletti (stati di Bell) con cui si vuole generare la chiave. Si osservi che solo il 22%, più precisamente 2/9, di essi è utile alla generazione della chiave.
        :param experiment_number: Valore opzionale utilizzato in una funzione secondaria, da ignorare in quanto utilizzato da un altro modulo.
        :param datatransformer: Oggetto utilizzato per trasformare i dati tramite imputatore e scalatore ai fini della classificazione, appartenente al modulo `package.classifiers`: ingloba elementi appartenenti al modulo `sklearn`, caricabili in formato pickle `.pkl` oppure passati come oggetti già addestrati.
        :param classifiers: Lista di oggetti utilizzati per la classificazione del canale di comunicazione, appartenente al modulo `package.classifiers`.
        :param backend: ([`qiskit.provider`](https://docs.quantum.ibm.com/api/qiskit/providers#backend) oppure [`qiskit_aer.AerSimulator`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html#qiskit_aer.AerSimulator)) backend IBM Quantum con cui si vuole eseguire l'esperimento. Se non specificato, di default si utilizzerà `AerSimulator()`.
        :param eavesdropper: Abilita la simulazione di un intercettatore Eve.
        :param progress: Mostra la percentuale di progresso dell'esperimento.
        :param verbose: Se abilitato, a fine esecuzione descrive tutti i risultati ottenuti.
        '''

        super().__init__(
            number_of_singlets = number_of_singlets,
            experiment_number = experiment_number,
            backend = backend,
            eavesdropper = eavesdropper,
            progress = progress,
            verbose = verbose
        )

        # Inizializzazione degli attributi di machine learning
        self.datatransformer = datatransformer
        self.classifiers = classifiers # lista di classificatori
        
        if self.classifiers is None:
            raise ValueError("l'E-E91 richiede una lista di classificatori.")
        
        if self.verbose:
            print("\n>>> Lista dei classificatori utilizzati\n")
            for classifier in self.classifiers:
                print(f"- {classifier}")

        # Lista delle "fraction" in cui si esegue la classificazione, "weight" è il peso nella votazione e "active" è il flag che stabilisce se il classificatore ha votato o meno in quel fraction
        self.thresholds = self._create_thresholds(n = 10, k = 1.5, s = 0.65, verbose = verbose)
        self.decision_threshold = 0.40      # soglia di decisione dopo il quale si esegue l'early-stop del protocollo: se i voti ponderati raggiungono questa soglia, si arresta la sessione
        self.decision_threshold_proba = 0.4 # si può polarizzare dove si vuole: la soglia dell'85% ho visto essere la soglia "perfetta" e anti-falsi negativi, ma vista la perfetta difesa del protocollo, è inutile: meglio l'accuratezza generica e si sposta a 0.6
        self.halt = False                   # flag utilizzato per arrestare la sessione in caso di rilevamento di un'intrusione
        self.weighted_votes = 0

        # Risultati del protocollo: indice di performance, ci dice a che punto si arresta il protocollo in termini di risparmio computazionale
        self.results["early_stopping_point"] = None

    def _create_thresholds(self, n: int = 10, k: float = 1.5, s: float = 0.65, verbose: bool = False):
        '''
        Funzione ausiliaria che definisce i checkpoint in cui eseguire la classificazione.
        Ogni checkpoint ha una "fraction" (soglia), un "peso" normalizzato e un flag "active".
        
        :param n: Numero di frazioni di classificazione.
        :param k: Parametro che regola la curva delle frazioni (più è grande, più si espone verso destra).
        :param weight_mode: Modalità di calcolo dei pesi. Con "linear", i pesi sono proporzionali a i. Con "power" sono proporzionali a i^p. Con "exp", pesi esponenziali exp(alpha * i). Con "fraction", pesi proporzionali alla stessa curva delle frazioni.
        :param p: Esponente usato se weight_mode = "power".
        :param alpha: Parametro usato se weight_mode = "exp".
        :param verbose: Se True, stampa le soglie e i pesi.
        
        :return thresholds: Lista di dizionari {fraction, weight, active}.
        '''
        
        thresholds = []
        progress_list = [i / n for i in range(1, n + 1)]  # progresso lineare

        # Calcolo delle fractions
        fractions = [round(1 - (1 - prog) ** k, 4) for prog in progress_list]

        # Mix tra forward (i, ...) e reverse (n + 1 - i, ...)
        forward = [i for i in range(1, n + 1)]
        reverse = [(n + 1 - i) for i in range(1, n + 1)]
        s = float(s) # serve a bilanciare forward e reverse: tra 0.6 e 0.7 è un ottimo valore
        raw = [(1 - s) * f + s * r for f, r in zip(forward, reverse)]
       
        # Normalizzazione
        total = sum(raw)
        weights = [w / total for w in raw][::-1]

        # Costruzione della lista
        for fraction, weight in zip(fractions, weights):
            thresholds.append(
                {
                    "fraction": fraction,
                    "weight": round(weight, 4),
                    "active": True
                }
            )

        if verbose:
            print("\n>>> Threshold di classificazione\n")
            for threshold in thresholds:
                print(f"Threshold: {threshold['fraction']:.2f}\t\tWeight: {threshold['weight']:.2f}")
            print("")

        return thresholds

    def predict(self) -> int:

        '''
        Funzione di classificazione del protocollo Enhanced-E91 finalizzata all'individuazione di un'intrusione.

        :param None:

        :return predict, predict_proba: Ritorna la predizione e la probabilità di predizione sull'intercettatore.
        '''

        def __bootstrap_resample(chsh_samples: dict, noise_scale: float = 0.01):

            '''
            Esegue il boostrapping del dataset.

            :param chsh_samples: Campioni estratti per il test di Bell.
            :param noise_scale: Scala del rumore gaussiano da aggiungere ai campioni. Il rumore viene aggiunto in modo proporzionale alla deviazione standard dei campioni.

            :return resampled: Ritorna un dizionario con i campioni risampled.
            '''

            resampled = {}
            for key, values in chsh_samples.items():

                # Step 1: bootstrap, ovvero campionamento con reinserimento
                resampled_values = np.random.choice(values, size = len(values), replace = True)
                
                ''' N.B.: induceva solo disturbo nella classificazione
                # Step 2: aggiunta di rumore gaussiano proporzionale alla deviazione standard
                if len(resampled_values) > 0:
                    noise = np.random.normal(loc = 0, scale = noise_scale * np.std(resampled_values), size = len(resampled_values))
                    resampled_values = resampled_values + noise
                '''
                
                resampled[key] = resampled_values.tolist()
            
            return resampled

        classifier_predicts, classifier_predict_probas = [], []

        for classifier in self.classifiers:

            # Estrazione dei parametri dell'esperimento
            chsh_samples                 = __bootstrap_resample(chsh_samples = self.results["chsh_samples"])
            anticorrelation_distribution = get_anticorrelation_distribution(chsh_samples = chsh_samples)

            # Inserimento dei risultati nel dataframe temporaneo
            df = pd.DataFrame({
                "Eavesdropper":                  [self.eavesdropper], # colonna target
                
                "CHSH samples":                  [chsh_samples],

                "(X, W) anticorrelation":        [anticorrelation_distribution[('X', 'W')]],
                "(X, V) anticorrelation":        [anticorrelation_distribution[('X', 'V')]],
                "(Z, W) anticorrelation":        [anticorrelation_distribution[('Z', 'W')]],
                "(Z, V) anticorrelation":        [anticorrelation_distribution[('Z', 'V')]],

            })

            # Manipolazione dei campioni CHSH per ottenere le statistiche
            chsh_samples_aggregated_statistics = df["CHSH samples"].apply(get_chsh_samples_statistics).apply(merge_dicts).apply(pd.Series) 
            df = df.drop(columns = ["CHSH samples"], axis = 1) # posso droppare le statistiche dei CHSH samples dopo aver generato il dataframe: droppo quelle che, dopo analisi, risultano essere confusionarie

            # Ora ho lo stesso dataframe di addestramento, e droppo le colonne "confusionarie" per i classificatori
            df = pd.concat([df, chsh_samples_aggregated_statistics], axis = 1)
            df = df.drop(
                columns = [ # vedi in "extract.process" per il significato di queste colonne
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
                ], 
                axis = 1
            )

            # Trasformo il dataframe con il datatransformer
            X  = df.drop(columns = ["Eavesdropper"], axis = 1)
            X  = self.datatransformer.transform(X)

            # Eseguo la predizione sul dato X
            classifier_predict       = int(classifier.predict(X = X)[0])
            classifier_predict_proba = classifier.predict_proba(X = X)[0]

            classifier_predicts.append(classifier_predict)
            classifier_predict_probas.append(classifier_predict_proba)

        # Probabilità media valutata sulle probabilità valutate dei classificatori: il primo elemento è la probabilità che sia classe 0 (non intercettazione)
        predict_proba = np.mean(classifier_predict_probas, axis = 0)

        # Stampa delle probabilità e delle predizioni dei classificatori
        if self.verbose: print(f" | [{predict_proba[0]:.4f}, {predict_proba[1]:.4f}] | {classifier_predicts}")
    
        # Se la maggioranza dei classificatori vota per 0 e la probabilità di essere 0 è maggiore della soglia, si classifica come 0 (non intercettazione)
        if sum(classifier_predicts) < len(classifier_predicts) // 2 and predict_proba[0] > self.decision_threshold_proba:
            predict = 0
        else:
            predict = 1 

        return predict, predict_proba

##################################################################################################################################################################################################

    def run(self) -> dict:
        
        '''
        Esegue il protocollo.

        :return evs: Valori attesi del test di Bell.
        :return chsh: Parametro del test CHSH.
        :return chsh_samples: Campioni estratti per il test di Bell.
        :return alice_key: Chiave di Alice (N.B.: non è formattata come stringa).
        :return bob_key: Chiave di Bob.
        :return eve_key_from_alice: Chiave di Eve, l'intercettatore, ottenuta da Alice.
        :return eve_key_from_bob: Chiave di Eve, l'intercettatore, ottenuta da Bob.
        :return alice_and_bob_compatible_bases_frequencies: Frequenza di basi compatibili tra Alice e Bob.
        :return alice_and_bob_mismatched_key_bits: Bit di chiave differenti tra Alice e Bob.
        :return qber: Quantum Bit Error Rate, tasso di errore del qubit, definito come il rapporto tra il numero di bit differenti tra Alice e Bob e la lunghezza di chiave (ovvero, il numero di qubit che si è riusciti a trasmettere).
        :return coincidence_counting: [Coincidence counting](https://en.wikipedia.org/wiki/Coincidence_counting_%28physics%29), consiste (a livello fisico) nel far passare i pulse di due fotodiodi entro una finestra Δt in un circuito elettronico dedicato: Aer fa già il lavoro sporco. Conta il numero di "coincidenze" tra due particelle entangled.
        :return alice_and_eve_mismatched_key_bits: Bit di chiave differenti tra Alice e Eve.
        :return bob_and_eve_mismatched_key_bits: Bit di chiave differenti tra Bob e Eve.
        :return eavesdropper_predict: Predizione sull'intercettatore.
        '''

        for i in range(self.number_of_singlets):

            # Esegue il corpo del protocollo E91, che si occupa di generare gli stati di Bell, le misure e le basi
            super(EE91_no_fidelity, self)._E91__body(idx = i) 

            '''
            Il rilevamento di un intrusione al k% avviene per maggioranza.
            Vengono utilizzati k "differenti" dataset, che fondamentalmente è
            sempre lo stesso self.results["chsh_samples"] ma con più campioni.
            
            - banalmente, quanto più sono i campioni in self.results["chsh_samples"], quanto più saranno precisi e raffinati i risultati delle funzioni "get_chsh_samples_statistics" e "get_anticorrelation_distribution"
            - quindi, emula un po' il concetto di "votazione" dei metodi ensemble, seppur non ne rispecchi le tecniche principali;
            - l'idea è di salvare tutte le votazioni e a metà esperimento eseguire un early stopping se viene rilevata un'intrusione: altrimenti, si esegue la classificazione finale.
            
            Verrà inoltre aggiunto del rumore a ogni sotto-dataset per renderlo differente dall'altro: ciò renderà le predizioni più indipendenti. 
            '''

            # Se si è raggiunta la frazione di classificazione, esegui la predizione e salva il voto per valutare la presenza di intercettazione
            for threshold in self.thresholds:
                
                # ...se si è superata la frazione corrispondente (es. 20%) e il flag è attivo (il flag dice se con la soglia del 20% puoi entrare nell'if: se è True, puoi entrare)
                if i / self.number_of_singlets > threshold["fraction"] and threshold["active"]:

                    threshold["active"] =  False # disattiva il flag per questa soglia
                    self.weighted_votes += threshold["weight"] * self.predict()[0] # esegue la predizione e salva il voto ponderato

            # Se i voti ponderati raggiungono (o superano) la soglia di decisione, si arresta la sessione
            if self.weighted_votes >= self.decision_threshold:

                # La maggioranza stabilisce che c'è stata un'intercettazione
                self.results["eavesdropper_predict"] = 1

                # Le predict_proba utile da salvare in CSV è solo quella della classificazione finale, quella nel caso in cui i classificatori non rilevino nulla fino all'ultimo step
                self.results["eavesdropper_predict_proba"] = None # segno "None" per indicare che i classificatori hanno rilevato un'intrusione
                self.halt = True

                # Punto in cui è stato arrestato il protocollo espresso in termini percentuali: servirà per valutare le performance
                self.results["early_stopping_point"] = i / self.number_of_singlets

                if self.verbose: print(" | Halt! Rilevata potenziale intrusione.")
                
                # Arresto la sessione
                break 

        # Classificazione finale con ulteriori informazioni solo se non c'è stato un halt (arresto precoce del protocollo)
        if not self.halt:

            predict, predict_proba                     = self.predict()
            self.results['eavesdropper_predict']       = predict
            self.results['eavesdropper_predict_proba'] = predict_proba

            if self.verbose:
                print(f"Predizione finale: ", end = "")
                if self.results['eavesdropper_predict'] == 1:
                    print(f"✗ presenza di un intercettatore al {(self.results['eavesdropper_predict_proba'][1] * 100):.2f}%")
                else:
                    print(f"✓ canale sicuro al {(self.results['eavesdropper_predict_proba'][0] * 100):.2f}%")

        # Calcolo dei parametri finali
        self.results['evs'], self.results['chsh'] = self.compute_chsh(chsh_samples = self.results["chsh_samples"])
        self.results['qber']                      = self.results['alice_and_bob_mismatched_key_bits'] / len(self.results['alice_key'])

        # Verbose dei risultati finali
        if self.verbose:
            self._prompt_results(
                evs = self.results['evs'],
                chsh = self.results['chsh'],
                chsh_samples = self.results['chsh_samples'],
                alice_key = self.results['alice_key'],
                bob_key = self.results['bob_key'],
                eve_key_from_alice = self.results['eve_key_from_alice'],
                eve_key_from_bob = self.results['eve_key_from_bob'],
                alice_and_bob_compatible_bases_frequencies = self.results['alice_and_bob_compatible_bases_frequencies'],
                alice_and_bob_mismatched_key_bits = self.results['alice_and_bob_mismatched_key_bits'],
                qber = self.results['qber'],
                coincidence_counting = self.results['coincidence_counting'],
                alice_and_eve_mismatched_key_bits = self.results['alice_and_eve_mismatched_key_bits'],
                bob_and_eve_mismatched_key_bits = self.results['bob_and_eve_mismatched_key_bits']
            )

        return self.results

"""