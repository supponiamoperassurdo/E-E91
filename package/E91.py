# Numpy
import numpy as np

# Qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

##################################################################################################################################################################################################

class E91:

    '''
    # E91

    Classe che simula il protocollo di QKD centralizzato [E91](https://en.wikipedia.org/wiki/Quantum_key_distribution#E91_protocol:_Artur_Ekert_.281991.29). \\
    Alcune supposizioni sono necessarie per il funzionamento del protocollo:

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
    '''

##################################################################################################################################################################################################

    def __init__(self,
            number_of_singlets: int = 512, 
            experiment_number: int = None, 
            backend: any = None, 
            eavesdropper: bool = False, 
            progress: bool = True,
            verbose: bool = False, 
        ) -> None:
        
        '''
        :param number_of_singlets: Numero di singoletti (stati di Bell) con cui si vuole generare la chiave. Si osservi che solo il 22%, più precisamente 2/9, di essi è utile alla generazione della chiave.
        :param experiment_number: Valore opzionale utilizzato in una funzione secondaria, da ignorare in quanto utilizzato da un altro modulo.
        :param backend: ([`qiskit.provider`](https://docs.quantum.ibm.com/api/qiskit/providers#backend) oppure [`qiskit_aer.AerSimulator`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html#qiskit_aer.AerSimulator)) backend IBM Quantum con cui si vuole eseguire l'esperimento. Se non specificato, di default si utilizzerà `AerSimulator()`.
        :param eavesdropper: Abilita la simulazione di un intercettatore Eve.
        :param progress: Mostra la percentuale di progresso dell'esperimento.
        :param verbose: Se abilitato, a fine esecuzione descrive tutti i risultati ottenuti.
        '''

        # Inizializzazione degli attributi del protocollo
        self.number_of_singlets = number_of_singlets
        self.experiment_number  = experiment_number
        self.eavesdropper       = eavesdropper
        self.progress           = progress
        self.verbose            = verbose

        # Inizializzazione del backend, del sampler e del traspilatore
        self.backend            = AerSimulator() if backend is None else backend
        self.sampler            = Sampler(mode = self.backend)
        self.sampler            .options.default_shots = 1
        self.pm                 = generate_preset_pass_manager(backend = self.backend, optimization_level = 0)
            
        # Inizializzazione dei partecipanti
        self.charlie            = self.Charlie()
        self.alice              = self.Alice()
        self.bob                = self.Bob()
        self.eve                = self.Eve() if eavesdropper else None

        # Inizializzazione delle basi dei partecipanti 
        self.alice.generate_bases(number_of_singlets = self.number_of_singlets)
        self.bob.generate_bases(number_of_singlets = self.number_of_singlets)
        if eavesdropper: self.eve.generate_bases(number_of_singlets = self.number_of_singlets) 

        # Inizializzazione delle strutture dati su cui conservare i risultati
        self.results = {
            'evs': [],
            'chsh': 0.0,
            'chsh_samples': {basis: [] for basis in [("X", "W"), ("X", "V"), ("Z", "W"), ("Z", "V")]},
            'alice_key': [],
            'bob_key': [],
            'eve_key_from_alice': [],
            'eve_key_from_bob': [],
            'alice_and_bob_compatible_bases_frequencies': {"W": 0, "Z": 0},
            'alice_and_bob_mismatched_key_bits': 0,
            'qber': 0.0,
            'coincidence_counting': 0,
            'alice_and_eve_mismatched_key_bits': 0,
            'bob_and_eve_mismatched_key_bits': 0
        }

    def _prompt_results(self,
            evs: list,
            chsh: float,
            chsh_samples: dict[tuple[str, str], list[float]], # inserito per poter fare l'unpack **
            alice_key: list,
            bob_key: list,
            eve_key_from_alice: list,
            eve_key_from_bob: list,
            alice_and_bob_compatible_bases_frequencies: dict,
            alice_and_bob_mismatched_key_bits: int,
            qber: float,
            coincidence_counting: int,
            alice_and_eve_mismatched_key_bits: int,
            bob_and_eve_mismatched_key_bits: int
        ) -> None:
        
        key_length = len(alice_key)
        eve_key_length = len(eve_key_from_alice)
        print("\n>>> Report dell'esperimento\n")
        print(f"- valori attesi:                                {evs[0]:.4f}, {evs[1]:.4f}, {evs[2]:.4f}, {evs[3]:.4f}")
        print(f"- parametro CHSH:                               {chsh}")
        print(f"- chiave di Alice:                              {''.join(alice_key)}")
        print(f"- chiave di Bob:                                {''.join(bob_key)}")
            
        print("- esito del test di Bell:                       ", end = "")
        if abs(chsh) > 2:
            print("✓ realismo locale violato: nessuna presenza di un intercettatore")
        else:
            print("✗ realtà fisica introdotta: [potenziale] presenza di un intercettatore")

        if self.eavesdropper:
            print("\n>>> Chiave di Eve\n")
            print(f"- chiave derivata da Alice:                     {''.join(eve_key_from_alice)}")
            print(f"- chiave derivata da Bob:                       {''.join(eve_key_from_bob)}")
            print(f"- conoscenza della chiave di Alice:             {(100 * (key_length - alice_and_eve_mismatched_key_bits) / key_length):.4f}%")
            print(f"- conoscenza della chiave di Bob:               {(100 * (key_length - bob_and_eve_mismatched_key_bits) / key_length):.4f}%")

        print("\n>>> Informazioni aggiuntive\n")
        print(f"- frequenza delle basi di Alice e Bob:          {alice_and_bob_compatible_bases_frequencies}")
        print(f"- lunghezza della chiave di Alice e Bob:        {key_length}")
        print(f"- efficienza della chiave di Alice e Bob:       {(100 * (key_length / self.number_of_singlets)):.4f}%")
        print(f"- bit di chiave differenti tra Alice e Bob:     {alice_and_bob_mismatched_key_bits}")
        print(f"- Quantum Bit Error Rate (QBER):                {(qber * 100):.4f}%")
        print(f"- coincidence counting:                         {coincidence_counting}")
        
        if self.eavesdropper:
            print(f"- lunghezza della chiave di Eve:                {eve_key_length}")
            print(f"- bit di chiave differenti tra Alice e Eve:     {alice_and_eve_mismatched_key_bits}")
            print(f"- bit di chiave differenti tra Bob e Eve:       {bob_and_eve_mismatched_key_bits}")

    def __extract_results(self,
            qc: QuantumCircuit,
            sampler: any,
            pm: any
        ) -> tuple[int, int]:

        '''
        Estrae i risultati delle misure, dato in circuito di input, tramite un SamplerV2. \\
        N.B.: l'attributo di `.data` è:

        - `data.<classical_register_name>` quando le misure vengono eseguite tramite la funzione `.measure(QuantumRegister, ClassicalRegister)`;
        - `data.meas` quando le misure vengono eseguite tramite la funzione `.measure_all()`.

        :param qc: Circuito quantistico da cui si vogliono estrarre i risultati delle misure: è necessario che preliminariamente siano stati applicate le funzioni `.measure(QuantumRegister, ClassicalRegister)` o `.measure_all()`
        :param sampler: Primitiva Qiskit che si occupa del sampling, ovvero `qiskit_ibm_runtime.SamplerV2`.
        :param pm: Componente Qiskit che si occupa della traspilazione (usa sorta di "compilazione" per i computer quantistici), del modulo `qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager`.

        :return bob_result, alice_result: Tupla ce rappresenta il risultato della misura sul MSquBit (quello di Bob, a sinistra), e sul LSquBit (quello di Alice, a destra): si osservi che Qiskit utilizza una notazione little endian

        Se si stampano i counts, si scopre che i primi due bit di sinistra sono i bit classici del registro di Eve:
        
            counts = {'0010': 1} # casistica in cui Bob ha ottenuto "1" e Alice "0"

        A sinistra si hanno i dati sui registri classici, a destra il numero di shots (sempre "1"):
        estraiamo il numero di chiavi e assegnamo accuratatamente
        i bit a chi di appartenenza tramite unpacking Python.
        '''

        data = sampler.run([(pm.run(qc))]).result()[0].data
        counts = data.c.get_counts()
        bits = list(counts.keys())[0]
        bob_result, alice_result = int(bits[2]), int(bits[3])

        return bob_result, alice_result

    def compute_chsh(self, 
            chsh_samples: dict[tuple[str, str], list[float]]
        ) -> float:

        '''
        Calcola i valori attesi 𝔼[...] di "chsh_samples", il prodotto delle misure di Alice e Bob su basi incompatibili.
               
        :param chsh_samples: dizionario di campioni per ogni set di basi scelte per il test di Bell
        
        :return evs: lista dei quattro EVs calcolato su ogni lista
        :return chsh: parametro CHSH del test di Bell
        '''

        evs = [np.mean(v) for v in chsh_samples.values()]
        chsh = evs[0] - evs[1] + evs[2] + evs[3]
        return evs, chsh

    def _compute_singlet_fidelity_from_qc(self,
            qc: QuantumCircuit
        ) -> float:

        """
        Calcola la fedeltà tra lo stato quantistico rappresentato dalla matrice di densità
        e lo stato singoletto ideale |ψ⟩ = 1/√2 (|01⟩ - |10⟩).

        :param rho: Matrice di densità dello stato quantistico.
        :return: Fedeltà con lo stato singoletto.
        """

        def remove_intermediate_measurements(qc, keep_final = False) -> QuantumCircuit:

            '''
            Rimuove tutte le misure intermedie, mantenendo opzionalmente quelle finali

            :param qc: Circuito quantistico da cui rimuovere le misure intermedie.
            :param keep_final: Se True, mantiene le misure finali.

            :return new_qc: Circuito quantistico senza misure intermedie.
            '''
            
            # Crea un nuovo circuito con gli stessi registri
            new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
            new_qc.global_phase = qc.global_phase
            new_qc.metadata = qc.metadata.copy()
            
            # Identifica le misure finali se necessario
            final_measurements = []
            if keep_final:
                for i, (instr, qargs, cargs) in enumerate(qc.data):
                    if instr.name == 'measure':
                        # Controlla se ci sono operazioni successive su questo qubit
                        qubit = qargs[0]
                        is_final = True
                        for j in range(i + 1, len(qc.data)):
                            future_instr, future_qargs, _ = qc.data[j]
                            if qubit in future_qargs and future_instr.name != 'measure':
                                is_final = False
                                break
                        if is_final:
                            final_measurements.append(i)
            
            # Copia solo le istruzioni che non sono misure intermedie
            for i, (instr, qargs, cargs) in enumerate(qc.data):
                if instr.name == 'measure':
                    if keep_final and i in final_measurements:
                        new_qc.append(instr, qargs, cargs)
                    # Salta le misure intermedie
                else:
                    new_qc.append(instr, qargs, cargs)
            
            return new_qc

        # È necessario che per la conversione in density matrix, non ci siano misure intermedie: rho sarà "pura" se Eve fa intercept-resend
        qc_no_measurements = remove_intermediate_measurements(qc = qc)
        rho = DensityMatrix.from_instruction(instruction = qc_no_measurements)

        # Singoletto ideale
        psi_singlet = np.array([0, 1, -1, 0], dtype = complex) / np.sqrt(2)
        
        # Calcolo fidelity
        fidelity = float(np.real(np.vdot(psi_singlet, rho @ psi_singlet)))
        
        return max(0.0, min(1.0, fidelity))

##################################################################################################################################################################################################

    class Charlie:

        '''
        La componente centralizzata del protocollo che si occupa di generare il singoletto. \\
        È la terza parte fidata per Alice e Bob.
        '''
        
        def create_singlet(self) -> QuantumCircuit:

            '''
            Charlie prepara uno spin singlet per Alice e Bob, cioè uno specifico stato di Bell, noto come:

                |ψ⟩ = 1/√2 (|01⟩ - |10⟩) 
            
            :param None:
            
            :return qc: Circuito quantistico che implementa lo stato di Bell sopra descritto. Si osservi che instanziamo direttamente 4 bit classici nel caso in cui si volesse simulare la presenza di un intercettatore.
            '''

            qr = QuantumRegister(2, name = "q")
            cr = ClassicalRegister(4, name = "c") # c[0]: Alice, c[1]: Bob, c[2-3]: Eve
            qc = QuantumCircuit(qr, cr)
            qc.x([0, 1])
            qc.h(0)
            qc.cx(0, 1)


            '''
            qc.delay(): Definisce la durata di un "tick" su cui il simulatore o l'hardware si basa per schedulare gate e delay.
                        Se n_ticks = 2, le operazioni verrano eseguite in 2 ticks.
                        Tipicamente, i backend IBM hanno una proprietà "dt" che definisce il tempo di tick. 
            
            '''
            
            qc.barrier()

            qc.delay(duration = 1, qarg = 0, unit = 'dt') # delay sul qubit di Alice: in un tick si applica l'errore
            qc.delay(duration = 1, qarg = 1, unit = 'dt') # delay sul qubit di Bob

            return qc

##################################################################################################################################################################################################

    class Alice:

        '''
        Prima parte comunicante del protocollo: le verrà assegnato il qubit q[0] e il bit c[0].
        '''
        
        def __init__(self):
            self.bases = []
            self.data = {}
        
        def generate_bases(self, number_of_singlets) -> str:

            '''
            Si generano randomicamente e una tantum - a partire da ["X", "W", "Z"] - le basi di Alice, per quanto sono il numero di esperimenti selezionati. \\
            I risultati sono salvati nella lista di classe self.bases.

            :param number_of_singlets: Numero di singoletti che determinano il numero di esecuzioni dell'esperimento.

            :return None: 
            '''
            
            alice_bases = np.random.choice(["X", "W", "Z"], number_of_singlets)
            self.bases = alice_bases

        def apply_measurement(self, qc: QuantumCircuit, basis: str) -> None:

            '''
            Esegue operazioni discrete per ruotare la sfera di Bloch verso la base scelta, per poi misurare rispetto alla base Z. \\
            Per la sequenza di rotazioni, mi sono ispirato al seguente [notebook](https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/awards/teach_me_qiskit_2018/e91_qkd/e91_quantum_key_distribution_protocol.ipynb).

            :param qc: Circuito quantistico su cui applicare le rotazioni e successivamente la misura.
            :param alice_basis: Base scelta da Alice.

            :return None:
            '''

            if basis == "X":
                qc.h(0)
            elif basis == "W":
                qc.s(0)
                qc.h(0)
                qc.t(0)
                qc.h(0)
            elif basis == "Z":
                pass
            
            qc.measure(0, 0)

##################################################################################################################################################################################################

    class Bob:

        '''
        Seconda parte comunicante del protocollo: gli verrà assegnato il qubit q[1] e il bit c[1].
        '''

        def __init__(self):
            self.bases = []
            self.data = {}
        
        def generate_bases(self, number_of_singlets) -> str:

            '''
            Si generano randomicamente e una tantum - a partire da ["W", "Z", "V"] - le basi di Bob, per quanto sono il numero di esperimenti selezionati. \\
            I risultati sono salvati nella lista di classe self.bases.

            :param number_of_singlets: Numero di singoletti che determinano il numero di esecuzioni dell'esperimento.

            :return None: 
            '''
            
            bob_bases = np.random.choice(["W", "Z", "V"], number_of_singlets)
            self.bases = bob_bases
        
        def apply_measurement(self, qc: QuantumCircuit, basis: str) -> None:

            '''
            Esegue operazioni discrete per ruotare la sfera di Bloch verso la base scelta, per poi misurare rispetto alla base Z. \\
            Per la sequenza di rotazioni, mi sono ispirato al seguente [notebook](https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/awards/teach_me_qiskit_2018/e91_qkd/e91_quantum_key_distribution_protocol.ipynb).

            :param qc: Circuito quantistico su cui applicare le rotazioni e successivamente la misura.
            :param bob_basis: Base scelta da Bob.

            :return None:
            '''

            if basis == "W":
                qc.s(1)
                qc.h(1)
                qc.t(1)
                qc.h(1)
            elif basis == "Z":
                pass
            elif basis == "V":
                qc.s(1)
                qc.h(1)
                qc.tdg(1)
                qc.h(1)
            
            qc.measure(1, 1)

##################################################################################################################################################################################################

    class Eve:

        '''
        Simula la parte di un intercettatore Eve: le verranno assegnati i due bit c[2] (per misurare il qubit di Alice) e c[3] (per misurare il qubit di Bob). \\
        Nel setting delle basi, Eve sceglie in maniera "intelligente" le basi di Alice e Bob:
        
        - poiché Eve vuole generare una chiave che sia il più simile possibile a quella di Alice e Bob, sceglie per entrambi solo il sottoinsieme di basi compatibili. \\
          Per esempio:
            * consideriamo le basi [X, W, Z] e [W, Z, V];
            * si segue un'intersezione e si estraggono solo [W, Z].
        - ovviamente, poiché il fine è generare una chiave...
            * Eve fa in modo che le basi scelte per misurare il qubit q[0] e il qubit q[1] siano le STESSE, dunque deve scegliere una coppia di basi uguali, per esempio WW o ZZ;
            * simula, cioè, il caso in cui Alice e Bob abbiano scelto la stessa base;
            * Eve conserverà il bit di chiave solo quando Alice e Bob sceglieranno la stessa base scelta da Eve.
        '''
            
        def __init__(self):
            self.bases = [] # insieme instersezione delle basi di Alice e Bob 
            self.results = {}

        def generate_bases(self, number_of_singlets) -> str:

            '''
            Si generano randomicamente e una tantum - a partire da ["W", "Z"] - le basi di Eve, per quanto sono il numero di esperimenti selezionati. \\
            I risultati sono salvati nella lista di classe self.bases.

            :param number_of_singlets: Numero di singoletti che determinano il numero di esecuzioni dell'esperimento.

            :return None: 
            '''
            
            eve_bases = np.random.choice(["W", "Z"], number_of_singlets)
            self.bases = eve_bases
        
        def apply_measurement(self, qc: QuantumCircuit, basis: str) -> None:

            '''
            Esegue operazioni discrete per ruotare la sfera di Bloch verso la base scelta, per poi misurare rispetto alla base Z.

            :param qc: Circuito quantistico su cui applicare le rotazioni e successivamente la misura.
            :param basis: Base scelta da Eve.

            :return None:
            '''

            # Aiuta alla visualizzazione
            qc.barrier()

            # Poiché la misura avverrà in maniera intelligente, applica gli gate a entrambi i qubit di Alice e Bob 
            if basis == "W":
                qc.s([0, 1])
                qc.h([0, 1])
                qc.t([0, 1])
                qc.h([0, 1])
            elif basis == "Z":
                pass

            # Eve misura il qubit di Alice (q[0]) e Bob (q[1]) nel suo registro classico
            qc.measure(0, 2) 
            qc.measure(1, 3)

        def __extract_eve_results(self, qc: QuantumCircuit, sampler: any, pm: any) -> tuple[int, int]:
                    
            '''
            Esegue un attacco intercept-resend implementato mid-circuit. \\
            L'attacco si basa sulla seguente condizione:

            - misura Alice: q[0] → c[2], resetta q[0], poi X se il bit misurato è 1
            - misura Bob:   q[1] → c[3], resetta q[1], poi X se il bit misurato è 1

            Dopo queste operazioni, i due qubit portano lo stato collassato scelto da Eve. \\
            Si ricordi che:

            - le misurazioni di Eve siano state eseguite su c[2] e c[3]
                * nel setting del codice, il registro classico totale ha 4 bit
                * gli ultimi 2 (contando in little endian) sono quelli di Eve
            - l'output del run (con un solo shot) sarà una stringa di 4 caratteri, dove:
                * bits[0] corrisponde a c[3] (Eve, qubit per Bob)
                * bits[1] corrisponde a c[2] (Eve, qubit per Alice)

            Dunque, l'ordine del creg è: <c[3], c[2], c[1], c[0]> (little endian Qiskit) = <Eve-to-Bob, Eve-to-Alice, Bob, Alice> = <bits[0], bits[1], bits[2], bits[3]> (ordinamento Python). \\
            Estraiamo i bit relativi a Eve:
            
            - eve_result_to_bob:   c[3] = bits[0]
            - eve_result_to_alice: c[2] = bits[1]
            
            :param qc: circuito quantistico da cui si vogliono estrarre i risultati delle misure.
            :param sampler: Primitiva Qiskit che si occupa del sampling.
            :param pm: Componente Qiskit che si occupa della traspilazione.
            
            :return eve_result_to_bob, eve_result_to_alice: Tupla in cui il risultato della misura sul primo qubit (quello di Bob), e il secondo elemento quello sul secondo qubit (quello di Alice).
            '''

            data = sampler.run([pm.run(qc)]).result()[0].data
            counts = data.c.get_counts()
            bits = list(counts.keys())[0] 
            eve_result_to_bob, eve_result_to_alice = int(bits[0]), int(bits[1])

            return eve_result_to_bob, eve_result_to_alice # c[3], c[2]

        def intercept(self, qc: QuantumCircuit, sampler = any, pm = any) -> tuple[int, int]:
            
            '''
            Eve esegue un attacco di tipo "intercept-resend": misura il qubit di Alice e di Bob e rimanda uno stato collassato.

            1. dopo la misura, Eve rimanda a Bob (misura x) e Alice (misura y) lo stato collassato |xy⟩;
            2. dunque, se per esempio dalla misura ottiene Bob = 0 e Alice = 1, rimanda lo stato |01⟩;
            3. prepara fondamentalmente un nuovo stato in funzione del risultato di Eve: è un modo per "confondere" Alice e Bob;
            4. però, essendo non entangled, verrà rilevata la manomissione e l'assenza di correlazione solo durante il test di Bell.

            Nel mondo reale, create un singoletto da parte di Eve è praticamente impossibile ma, se ne fosse capace,
            sarebbe possibile riuscire a bypassare il test di Bell: non riuscirebbe, in ogni caso, a ottenere informazioni
            utili sulla chiave grazie al no-cloning theorem. Dunque, se anche fosse possibile, sarebbe un inutile spreco di risorse.
            
            :param qc: Circuito quantistico di input, su cui operano Alice e Bob;
            :param sampler: Primitiva Qiskit che si occupa del sampling.
            :param pm: Componente Qiskit che si occupa della traspilazione.
            
            :return eve_result_to_bob, eve_result_to_alice: Tupla in cui il risultato della misura sul primo qubit (quello di Bob), e il secondo elemento quello sul secondo qubit (quello di Alice).
            '''

            eve_result_to_bob, eve_result_to_alice = self.__extract_eve_results(qc = qc, sampler = sampler, pm = pm)

            if eve_result_to_bob == 0 and eve_result_to_alice == 1:
                qc.initialize('01')
            elif eve_result_to_bob == 1 and eve_result_to_alice == 0:
                qc.initialize('10')

            # In questo modo il circuito "ri-preparato" riflette in qualche modo la misura di Eve
            return eve_result_to_bob, eve_result_to_alice

##################################################################################################################################################################################################

    def __body(self, idx: int) -> dict:

        '''
        Corpo del protocollo E91.
        Si osservi che, per ogni singoletto, si esegue il seguente processo:
        1. Charlie crea un singoletto; 
        2. Eve, se presente, intercetta il qubit di Alice e Bob, antepone il suo osservabile ed esegue la misura prima di loro;
        3. Alice e Bob recuperano le basi generate randomicamente e misurano i loro corrispettivi qubit;
        4. Alice e Bob misurano i loro qubit;  
        5. Coincidence count: se le misure sono anticorrelate, vi è stata una coincidenza;
        6. Annuncio pubblico della basi: se sono compatibili, si conservano le misure come bit di chiave;
        7. Se non sono compatibili - e appartengono al set di basi del test di Bell - si conservano le misure e si utilizzano per il test di Bell;
        8. Se Eve è presente, scruta il canale pubblico: se Alice e Bob hanno annunciato la stessa base, e Eve ha scelto la stessa coppia, conserva il bit di chiave;
            - Però, Eve non è in grado di ottenere informazioni utili sulla chiave, poiché non può clonare gli stati quantistici e, dunque, non può ottenere informazioni utili sulla chiave.
            - Alice e Bob sono in grado di rilevare la presenza di Eve grazie al test di Bell;
            - Alice e Bob sono in grado di generare una chiave sicura, poiché gli stati quantistici sono entangled e - ovviamente - non possono essere clonati, poiché godono del no-cloning theorem.

        :param idx: Indice del singoletto corrente.

        :return dict:    
        '''

        # Charlie instanzia un singoletto, che viene passato a Alice e Bob
        qc = self.charlie.create_singlet()
        
        # Eve, se presente, intercetta il qubit di Alice e Bob, antepone il suo osservabile ed esegue la misura prima di loro
        if self.eavesdropper:
            eve_basis = self.alice.bases[idx] 
            self.eve.apply_measurement(qc = qc, basis = eve_basis)
            eve_result_to_bob, eve_result_to_alice = self.eve.intercept(qc = qc, sampler = self.sampler, pm = self.pm)

        # Alice e Bob recuperano le basi generate randomicamente e misurano i loro corrispettivi qubit 
        alice_basis = self.alice.bases[idx]
        bob_basis = self.bob.bases[idx]
        self.alice.apply_measurement(qc = qc, basis = alice_basis)
        self.bob.apply_measurement(qc = qc, basis = bob_basis)

        # Alice e Bob misurano i loro qubit
        bob_result, alice_result = self.__extract_results(qc = qc, sampler = self.sampler, pm = self.pm)

        # Fidelità dei singoletti
        # self.results['singlet_fidelities'].append(self._compute_singlet_fidelity_from_qc(qc))

        # Coincidence count: se le misure sono anticorrelate, vi è stata una coincidenza
        if alice_result != bob_result:
            self.results['coincidence_counting'] += 1

        # Annuncio pubblico della basi: se sono compatibili, si conservano le misure come bit di chiave
        if alice_basis == bob_basis:

            # Frequenza delle basi compatibili tra Alice e Bob
            self.results['alice_and_bob_compatible_bases_frequencies'][alice_basis] += 1 

            # Bit di chiave differenti tra Alice e Bob: se le misure non sono anticorrelate, allora i bit sono differenti
            if not (alice_result != bob_result):
                self.results['alice_and_bob_mismatched_key_bits'] += 1

            # Alice conserva il suo risultato nella chiave
            self.results['alice_key'].append(str(alice_result))

            # Bob, sapendo che il suo risultato deve essere anti-correlato da quello di Alice, nega il suo bit
            self.results['bob_key'].append(str((bob_result - 1) % 2))

        # ...se non sono compatibili - e appartengono al set di basi del test di Bell - si conservano le misure e si utilizzano per il test di Bell
        elif (alice_basis, bob_basis) in self.results["chsh_samples"]:

            # cambio i bit "0" in "-1" (corrispondenti autovalori) ai fini del test di Bell, e salvo il prodotto tra i risultati di Alice e Bob in "chsh_samples"
            bob_result = 1 if bob_result == 1 else -1 ; alice_result = 1 if alice_result == 1 else -1
            self.results["chsh_samples"][(alice_basis, bob_basis)].append(alice_result * bob_result)
        
        # Se Eve è presente, scruta il canale pubblico
        if self.eavesdropper:

            # ... se Alice e Bob hanno annunciato la stessa base, e Eve ha scelto la stessa coppia, conserva il bit di chiave (N.B.: per come è scritto l'algoritmo, avrà necessariamente scelto la stessa coppia ZZ o WW, mai ZW o WZ)
            if alice_basis == bob_basis and (eve_basis == alice_basis and eve_basis == bob_basis):
            
                # Bit di chiave differenti tra Alice e Eve
                if alice_result != eve_result_to_alice:
                    self.results['alice_and_eve_mismatched_key_bits'] += 1 

                # Bit di chiave differenti tra Bob e Eve
                if bob_result != eve_result_to_bob:
                    self.results['bob_and_eve_mismatched_key_bits'] += 1 

                # Eve conserva il suo risultato nella chiave
                self.results['eve_key_from_alice'].append(str(eve_result_to_alice))
                self.results['eve_key_from_bob'].append(str((eve_result_to_bob - 1) % 2))

        # Progress bar 
        if self.experiment_number is not None: # "experiment_number" è utilizzato da funzioni esterne: denota il numero di esperimenti e non verrà utilizzato se settato a "None"
            print(f"\rEsecuzione esperimento n. {self.experiment_number} - Progresso: {(idx + 1) / self.number_of_singlets * 100:.1f}%", end = "", flush = True)
        else:
            print(f"\rProgresso: {(idx + 1) / self.number_of_singlets * 100:.1f}%", end = "", flush = True)

        return self.results

    def run(self) -> dict:

        '''
        Esegue il protocollo.
        
        :param None:

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
        '''

        # Itero su tutti i singoletti, e per ogni singoletto...
        for i in range(self.number_of_singlets):
            self.__body(idx = i)
                    
        # Calcolo i parametri finali e mostro i risultati
        self.results['evs'], self.results['chsh'] = self.compute_chsh(chsh_samples = self.results["chsh_samples"])
        self.results['qber']                      = self.results['alice_and_bob_mismatched_key_bits'] / len(self.results['alice_key'])
        if self.verbose:                            self._prompt_results(**self.results)
        
        return self.results

##################################################################################################################################################################################################
