import json

# Questo import per la versione successiva di Qiskit Aer
from qiskit_ibm_runtime.models.backend_properties import BackendProperties

# Questo import per la versione precedente di Qiskit Aer: mantienili entrambi
from qiskit.providers.models.backendproperties import BackendProperties

from qiskit_aer.noise import RelaxationNoisePass, NoiseModel, ReadoutError, amplitude_damping_error, phase_damping_error, depolarizing_error # type: ignore

################################################################################################################################################################################################################

def noise_model_to_json(backend: any) -> None:

    '''
    Salva le proprietà fisiche di un backend in un file JSON, comprese di:

    - configurazione
    - defaults
    - proprietà
    - status
    
    :param backend: Backend IBM Quantum, simulato o reale, di cui si vuole eseguire il dump delle proprietà.
    
    :return None:
    '''

    # estraggo le informazioni e le converto in forma di dizionario
    full_backend = {
        'configuration': backend.configuration().to_dict(), # include dt, n_qubits, basis_gates, parametri di pulse, canali, ecc.
        'defaults': backend.defaults().to_dict(),           # mappe di scheduling (circuit_instruction_map), parametri di apertura pulse, ecc
        'properties': backend.properties().to_dict(),       # contiene T₁, T₂, errori di gate, readout error e timestamp
        'status': backend.status().to_dict()                # se il device è operativo, quanti job in coda, messaggi
    }

    with open(f'{backend.name}_properties.json', 'w') as file:
        json.dump(full_backend, file, indent = 4, sort_keys = True, default = str)
        print(f"Dumped {backend.name}_properties.json")

################################################################################################################################################################################################################

def noise_model_from_json(json_file: any, gate_error: bool = True, readout_error: bool = True, thermal_relaxation_error: bool = True, warnings: bool = True) -> tuple[tuple[str], NoiseModel]:

    '''
    Genera un `NoiseModel()` a partire da un file JSON contenente un `backend.properties().to_dict()` IBM Quantum, che include errori di readout, decoerenza e di gate.
    
    :param json_file: File in formato JSON contenente le proprietà del backend.
    :param gate_error: Se impostato a 'False' disabilita gli errori di gate. È fortemente CONSIGLIATO se si vuole esponenzialmente ridurre l'esecuzione degli esperimenti (es.: da 6 minuti a 30 secondi circa).
    :param readout_error: Se impostato a 'False' disabilita gli errori di readout.
    :param thermal_relaxation_error: Se impostato a 'False' disabilita gli errori di decoerenza.
    
    :return backend_info, noise_model: Informazioni del backend che si sta approssimando (nome, versione, ultimo aggiornamento) e `NoiseModel()` che approssima il backend caricato dal file JSON.
    '''

    # loading del file JSON
    with open(json_file, "r") as file:
        noise_data = json.load(file)

    configuration = noise_data['configuration']
    defaults = noise_data['defaults']
    properties = noise_data['properties']
    status = noise_data['status']

    # estrazione dei gate_lengths da noise_data['properties']['gates']
    gate_lengths = []
    for gate in properties['gates']:
        name = gate['gate']
        gl = next(p['value'] for p in gate['parameters'] if p['name'] == 'gate_length')
        gate_lengths.append((name, gl)) # otterò una lista di tuple [("cx", 270), ("sx", 60), ...]
        
    # si instanzia un modello di rumore
    noise_model = NoiseModel.from_backend_properties(
        backend_properties = BackendProperties.from_dict(properties),
        gate_error = gate_error,
        readout_error = readout_error,
        thermal_relaxation = thermal_relaxation_error,
        # temperature = 0, # la lascio di default
        gate_lengths = None,
        gate_length_units = "ns",
        dt = configuration['dt'] # è già espresso in secondi
    )

    '''
    >>> Gli errori di decoerenza sono applicati solo alle porte identità I

    I tempi di decoerenza sono misurati a livello di qubit e rappresentano il degrado dello stato quantistico nel tempo:
    
    - tipicamente, durante i periodi in cui il qubit è inattivo, e l'operazione "id" (I, identity) simula un'azione in cui il qubit non subisce una trasformazione attiva
    - quindi, mentre i gate specifici (come "cx", "u2", "u3", ecc.) hanno errori propri derivanti dalle proprietà misurate del backend, i parametri di decoerenza
    si applicano a livello di qubit (cioè, tramite errori "id") per modellare il decadimento naturale quando il qubit è inattivo.

    Questa convenzione è ampiamente adottata nei modelli di rumore per simulare realisticamente l'effetto del tempo di inattività sui qubit.

    >>> Spiegazione "meno teorica"

    Si immaginano che i qubit siano in un ambiente in cui, se non eccitati, col tempo deteriorano.
    Quando non si eseguae alcun "movimento" - cioè, non si applica al una porta attiva -, il qubit "rimane fermo"
    e perde coerenza a causa del tempo che passa. In un circuito quantistico, questo periodo di inattività
    viene spesso rappresentato dalla porta "id" (I, identity). 
    Quindi:

    - errori di decoerenza: legati al tempo in cui il qubit è inattivo. Per modellare questo, usiamo una "id" per indicare che il qubit sta "aspettando" e, in quel periodo, subisce la decoerenza;
    - errori dei gate (come cx, u2, ecc.): sono legati a operazioni specifiche.

    Per questo motivo, nella funzione, gli errori di decoerenza vengono applicati alla porta "id".
    È una scelta legata al fatto che il qubit non esegue un'operazione attiva, e la sua degradazione
    si manifesta proprio come se avesse subito un'operazione "identity" rumorosa.

    In sintesi:

    - gli errori sui gate specifici sono per le operazioni attive...  
    - gli errori di decoerenza si verificano mentre il qubit "sta fermo" (cioè durante l'id).

    Questa suddivisione rende il modello di rumore più realistico.
    '''

    backend_info = [properties['backend_name'], properties['backend_version'], properties['last_update_date']]

    return backend_info, noise_model

################################################################################################################################################################################################################

def fiber_optic_channel(noise_model: NoiseModel = NoiseModel(), gamma: float = 0.5, f: float = 1e-3, p_depolarizing: float = 1e-4, p_readout_error: float = 0.01): #, p_dark_counts: float = 1e-8):

    '''
    Restituisce una serie di QuantumError che simula l'attenuazione dovuta a una comunicazione su fibra ottica.
    
    1. amplitude damping: descrive la perdita di energia dei fotoni. Come riferimento è stata utilizzata la fibra SMF-28 a 1550 nm che tipicamente possiede un'attenuazione di:
        1. [α = 0.197 dB/km](https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-5-6010&id=382202) [(min ≃ 0.16 dB/km, max ≃ 0.25 dB/km)](https://www.corning.com/media/worldwide/coc/documents/Fiber/product-information-sheets/PI-1470-AEN.pdf), sono range di scenari *normali*, non eccessivamente soggetti a rumore (altrimenti α incrementerebbe).
        1. per quanto concerne la lunghezza L, questa è meno "impattante" per gli esperimenti, se viene mantenuta fissata (gli effetti si possono individuare derivando la funzione gamma) rispetto a piccole variazioni di α.
        1. si possono esplorare diversi scenari rispetto ad α o la lunghezza L: libertà lasciata all'utente, ma range tipici in città sono 30-50 km;
        1. si può valutare la robustezza facendo oscillare α entro ±0.02–0.05 dB/km.
        1. ERRORE: questi parametri si possono usare nel caso di un canale di erasure.
            1. Un canale di erasure simula la probabilità che un fotone venga perso con p_loss = 1 - 10 ** (- α * L / 10);
            1. poiché stiamo modellando un canale di amplitude damping, possiamo al più utilizzare un range di γ_eff = [0.4, 0.8] che possa approssimare la p_loss;
            1. Per α=0.2 dB/km, risolvendo p_loss = 1 - 10 ** (- 0.2 * L / 10) ≃ [0.5, 0.8] si ottiene L ≃ 15-35 km.
        Il canale amplitude damping con γ = [0.5, 0.8] rappresenta un modello effective error in cui si combina perdita di fotoni (erasure)
        e rumore di rilevazione (dark counts/trattamento del vuoto), traducendo la probabilità di perdita in errore di bit.
        Sebbene non sia un modello fisico esatto, permette di esplorare scenari con CHSH fortemente degradato, equivalenti a condizioni di link molto rumoroso.


    2. phase damping: descrive la degradazione della coerenza quantistica dovuta a interazioni con un ambiente di oscillatori senza perdita di energia:
        1. studi di disseminazione di frequenza ultrastabile su link in fibra segnalano densità spettrale di rumore di fase per unità di lunghezza attorno a [Sφ(1 Hz) ≃ 38.5 rad²/Hz/km](https://arxiv.org/pdf/1512.02799), dovuto a fluttuazioni termiche o acustiche lungo il percorso.
        1. il problema è "tradurlo" in una fase - che chiameremo f - da utilizzare come parametro della funzione del nostro NoiseModel. Il parametro "f", inteso come tasso
        di dephasing per km (es. f ∈ [1e-3, 1e-1]) deve emergere dall'integrazione di Sφ(f) su banda utile: per esempio, se Sφ(1 Hz) ≃ 38.5 rad²/Hz/km,
        e si considera una banda efficace Δf intorno a qualche Hz fino a kHz (dipende dall'inviluppo temporale del segnale),
        la varianza di fase accumulata per km può cadere nell'ordine di 1e-3-1e-1 rad², a seconda di L e della banda integrata;
        1. sulla base di quanto detto, non si può dire che esista un valore "universale" del parametro "f", in quanto dipendente dalla sensibilità del setup,
        però il range O(1e-3-1e-1) rad²/km per varianza di fase è [coerente con misure sperimentali](https://www.sciencedirect.com/science/article/abs/pii/S0030399222008842) di rumore di fase in link di fibra metropolitana 30-100 km.

    3. depolarizing: un canale depolarizzante di probabilità p può approssimare errori casuali di flipping su base singolo-fotone (bit-flip o phase-flip). In simulazioni di sicurezza, p viene scelto in funzione della QBER sperimentale di fondo: se QBER ≃ 0.01-0.05, si può mappare p_dep ≃ QBER (o leggermente inferiore, a seconda di contributi distinti):
        1. valori tipici che modellano il rumore di fondo sono [1e-4, 1e-1];
        1. questo range [1e-4, 1e-1] copre scenari da rumore molto basso (fibra molto pulita, link breve, detectors eccellenti, quando p ≃ 1e-4-1e-3)
        fino a link più rumorosi o integrazione con traffico, dove p può avvicinarsi a 1e-2 e 1e-1.

    4. readout: modella gli errori dei dispositivi di misura:
        1. si utilizzerà un range tipico delle simulazioni, [0.01, 0.1].
        
    Se non viene specificato alcun input, inizializza un NoiseModel nullo che simula il rumore di fibra ottica inserendo i rumori descritti in precedenza con i settaggi minimi. \\
    A seguire, la lista dei parametri necessari alla costruzione del canale.

    :param noise_model: NoiseModel qiskit che si vuole modellare.
    :param gamma: Probabilità di amplitude damping, stimata dal parametro α di attenutazione della fibra ottica *in dB/km*.
    :param b: Probabilità di phase damping valutata a partire dalla densità spettrale della fibra ottica.
    :param p_depolarizing: Probabilità di depolarizzazione del canale.
    :param p_readout_error: Probabilità di misurare erronamente i fotoni.
    
    :return NoiseModel: NoiseModel che simula il rumore all'interno di un canale di fibra ottica.
    '''

    # probabilità di perdita di un fotone in fibra ottica
    # gamma = 1 - 10 ** (- alpha * fiber_length / 10)
    amplitude_damping = amplitude_damping_error(gamma)
    
    # probabilità di perdita di coerenza dei fotoni
    phase_damping = phase_damping_error(f)
    
    # probabilità di depolarizzazione
    depolarizing = depolarizing_error(p_depolarizing, 1)

    # aggiungo il readout solo nell'ipotesi in cui non ci sia
    if noise_model._local_readout_errors == {}: # ci sono problemi di composizione e serializzazione per un bug Qiskit: "RuntimeError: [json.exception.type_error.302] type must be number, but is array", eviteremo composizioni di errori di readout
        # print(f"Errore di readout assente: verra aggiunto con probabilità {p_readout_error * 100:2f}%.")
        noise_model.add_all_qubit_readout_error(
            error = ReadoutError([
                [1 - p_readout_error, p_readout_error],
                [p_readout_error, 1 - p_readout_error]
            ]),
            warnings = False
        )
    
    noise_model.add_all_qubit_quantum_error(error = amplitude_damping, instructions = ["delay"], warnings = False) # perdita di energia dei fotoni
    noise_model.add_all_qubit_quantum_error(error = phase_damping, instructions = ["delay"], warnings = False) # perdita di coerenza
    noise_model.add_all_qubit_quantum_error(error = depolarizing, instructions = ["delay"], warnings = False) # rumore di fondo modellato uniformemente nelle tre basi di misura
    
    return noise_model

################################################################################################################################################################################################################

def get_dt_from_noise_model(noise_model: NoiseModel):

    '''
    Estrae il parametro _dt (tempo di campionamento, memorizzato in secondi: attributo privato di NoiseModel).

    :param noise_model: Modello di rumore di cui si vuole estrarre il tempo di campionamento.
    
    :return dt: Tempo di campionamento in secondi. Sarà "None" se il NoiseModel non possiede errori termici o se dt = None.
    '''

    '''
    Il parametro "dt" non viene memorizzato come attributo pubblico di NoiseModel, ma è passato internamente al qiskit_aer.noise.RelaxationNoisePass:
    questi gestisce gli errori di rilassamento termico sui delay. Di conseguenza, per recuperare dt dopo aver creato il simulatore, è necessario estrarre
    il RelaxationNoisePass dalla lista privata NoiseModel._custom_noise_passes e leggerne l'attributo privato "_dt".

    N.B.: se thermal_relaxation = False o si è passato dt = None, la lista _custom_noise_passes sarà vuota e non si troverà alcun RelaxationNoisePass.
    '''

    # cerca il pass che gestisce i delay
    if len(noise_model._custom_noise_passes) > 0: 
        delay_pass = next(
            p for p in noise_model._custom_noise_passes
            if isinstance(p, RelaxationNoisePass)
        )
    
        # estraggo l'attributo privato
        dt = delay_pass._dt
    else:
        # nell'ipotesi in cui la lista sia vuoto, ritorno un valore di dt pari a None, per rispettare i parametri di input iniziali
        dt = None

    return dt

################################################################################################################################################################################################################
