import json
import itertools
from pathlib import Path

################################################################################################################################################################################################################

def xorencrypt(key: str, text: str) -> bytes:

    '''
    Cifra - o decifra - con una semplice operazione di XOR tra plaintext e chiave. \\
    È più simile a un cifrario di Vernam.

    :param key: Chiave del cifrario.
    :param text: Testo da cifrare o decifrare, come stringa o bytes.

    :return: Testo cifrato (o decifrato) in forma di bytes.
    '''

    # Se "key" è una stringa, codificala, altrimenti usala direttamente
    key_bytes = key.encode('utf-8') if isinstance(key, str) else key
    
    # Se "text" è una stringa, codificala, altrimenti usala direttamente
    text_bytes = text.encode('utf-8') if isinstance(text, str) else text
    
    # Necessario per eseguire lo XOR su tutti i bytes di "text"
    keystream = itertools.cycle(key_bytes)

    return bytes(b ^ next(keystream) for b in text_bytes)

################################################################################################################################################################################################################

def load_last_done(checkpoint_path: Path) -> int:
    
    '''
    Legge da `checkpoint_path` l'indice dell'ultimo esperimento completato.
    Se il file non esiste o è vuoto, restituisce 0.

    Questa funzione ausiliaria è utilizzata principalmente per tenere traccia dell'ultimo esperimento generato nel caso di utilizzo
    di matrici LHS (la cui rigenerazione della matrice invalida l'esperimento) o di matrici pre-generate Numpy (file .npy).
    '''
    
    if not checkpoint_path.exists():
        return 0
    try:
        with checkpoint_path.open("r") as f:
            data = json.load(f)
        return (
            data.get("last_done", 0),
            data.get("seed", None),
            data.get("n", None),
        )
    except (json.JSONDecodeError, IOError):
        raise RuntimeError(
            f"Il file di checkpoint {checkpoint_path} è corrotto; "
            "eliminalo o rinominalo per rigenerare un nuovo checkpoint."
        )

################################################################################################################################################################################################################

def save_last_done(checkpoint_path: Path, last_done: int, seed: int, n: int) -> None:
    
    '''
    Salva in `checkpoint_path` l'indice (0-based) dell'ultimo esperimento completato.
    Se il file non esiste, lo crea; se esiste, lo sovrascrive.

    Questa funzione ausiliaria è utilizzata principalmente per tenere traccia dell'ultimo esperimento generato nel caso di utilizzo di metodi LHS (la cui rigenerazione della matrice invalida l'esperimento).
    '''
    
    payload = {
        "last_done": last_done,
        "seed":      seed,
        "n":         n
    }
    checkpoint_path.parent.mkdir(parents = True, exist_ok = True) # crea cartella se serve
    with checkpoint_path.open("w") as f:
        json.dump(payload, f, indent = 2)

################################################################################################################################################################################################################

"""
class E91BasesManager:

    '''
    Classe ausiliaria per la pre-generazione e gestione di basi del protocollo E91, in modo che possano essere riutilizzate
    in più esecuzioni o in situazioni differenti, mantenendo il setting identico.
    '''

    def __init__(self, number_of_experiments: int = None, number_of_singlets: int = None, seed: float = None):

        '''
        :param number_of_experiments: Numero di esperimenti totali (es. 1000).
        :param number_of_singlets: Numero di basi per singolo esperimento (es. 512).
        :param seed: Seed Numpy, necessario per la riproducibilità della generazione pseudo-casuale delle basi.

        '''
        
        self.number_of_singlets    = number_of_singlets
        self.number_of_experiments = number_of_experiments

        # Inizialmente non generate
        self.alice_bases = None
        self.bob_bases   = None
        self.eve_bases   = None

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed) # crea un generatore con seed
        else:
            self.seed = None                       # il seed non è stato specificato e verrà generato casualmente da Numpy
            self.rng = np.random.default_rng()     # crea un generatore senza seed

    def generate_all_bases(self):

        '''
        Genera la matrice di basi [number_of_experiments, number_of_singlets] per il protocollo E91.

        :param None:

        :return None:
        '''

        # genera matrici 2D: [number_of_experiments, number_of_singlets]
        self.alice_bases = self.rng.choice(["X", "W", "Z"], size = (self.number_of_experiments, self.number_of_singlets))
        self.bob_bases   = self.rng.choice(["W", "Z", "V"], size = (self.number_of_experiments, self.number_of_singlets))
        self.eve_bases   = self.rng.choice(["W", "Z"],      size = (self.number_of_experiments, self.number_of_singlets))

    def get_seed(self) -> int:
        
        '''
        Ritorna il seed utilizzato per generare le basi. Se il seed non è stato specificato, ritorna None.

        :param None:

        :return: Il seed utilizzato per la generazione delle basi.
        '''
        
        return self.seed

    def save_bases(self, filepath: str = None) -> None:

        '''
        Salva la matrice di dati generata in un file compresso Numpy, per renderlo successivamente riutilizzabile. \\
        Il file compresso .npz conterrà tre array: alice, bob ed eve, che rappresentano le basi scelte da ciascun partecipante per ogni esperimento, 
        e il nome di default del file sarà "data/bases/<seed>.npz", dove <seed> è il seed utilizzato per generare le basi.
            
        :param filename: Nome con cui si vuole salvare la matrice di dati nel disco come file compresso .npz.

        :return None:
        '''

        if filepath is None:
            filepath = f"data/bases/{self.seed}.npz"

        if self.alice_bases is None or self.bob_bases is None or self.eve_bases is None:
            raise RuntimeError("devi prima generare tutte le basi con .generate_all_bases().")
        
        np.savez(
            filepath,
            alice = self.alice_bases,
            bob   = self.bob_bases,
            eve   = self.eve_bases
        )
        
        print(f"Tutti e tre gli array salvati in {filepath}")

    @classmethod
    def load_bases(cls, filepath: str):
        
        '''
        Carica il file compresso Numpy per renderlo utilizzabile. \\
        Per utilizzare il metodo .get_bases(idx), è necessario dunque precederlo da load_bases, per aver nota la matrice di dati da utilizzare. \\
        N.B.: si osservi che per CARICARE (non salvare) i dati salvati in memoria, si utilizza come metodo di classe.

        ```python
        # si utilizza come metodo di classe:
        bases_manager = E91BasesManager.load_bases("data/bases/<seed>.npz")

        # e per ottenere le basi di un esperimento specifico, si utilizza:
        bases_manager.get_bases(idx = 1)
        ```

        :param filepath: File compresso .npz da caricare in memoria.

        :return: Un'istanza di E91BasesManager con le basi caricate.
        '''

        # carico il file compresso .npz
        loaded = np.load(filepath)

        # creo una nuova istanza senza generare basi
        obj = cls()
        obj.alice_bases = loaded["alice"]
        obj.bob_bases   = loaded["bob"]
        obj.eve_bases   = loaded["eve"]
        
        return obj

    def get_bases(self, idx: int) -> tuple[list[str], list[str], list[str]]:

        '''
        Ritorna i tre array di basi corrispondenti all'esperimento 'idx' (0 <= idx < number_of_experiments).
        Per esempio:

        - idx = 0 sono prime 512 basi (corrispondenti al primo esperimento)
        - idx = 1 sono le seconde 512 basi (corrispondenti al secondo esperimento)
        - etc.

        Le tre liste di basi per l'esperimento idx sono:

        - alice = self.alice_bases[idx]
        - bob   = self.bob_bases[idx]
        - eve   = self.eve_bases[idx]
        
        :param idx: Indice dell'esperimento a cui si vogliono fare corrispondere le basi salvate in memoria.

        :return (alice_bases, bob_bases, eve_bases): Una tupla contenente tre liste di basi.
        '''
        
        if idx < 0 or idx >= len(self.alice_bases): # se l'indice supera la lunghezza massima della lista (basta scegliere una delle liste tra Alice, Bob o Eve)
            raise IndexError("Indice esperimento fuori range.")
        return (
            self.alice_bases[idx],
            self.bob_bases[idx],
            self.eve_bases[idx]
        )
"""
        
