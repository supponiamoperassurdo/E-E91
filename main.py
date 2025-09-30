import sys

def Simulate_E91():

    '''
    Simulazione del protocollo E91 di distribuzione di chiavi quantistiche.
    Si può scegliere se eseguire la simulazione in un canale ideale o rumoroso, e se simulare o meno la presenza di un intercettatore.
    In quest'ultimo caso, si può vedere come l'intercettazione vada a compromettere la correttezza della chiave condivisa tra Alice e Bob.
    In ogni caso, si può vedere come l'intercettatore non riesca a decifrare il messaggio cifrato con la chiave di Alice.
    '''

    eavesdropper       = True if input("Vuoi eseguire la simulazione di una intercettazione? ('n' / 'y') ") == "y" else False
    real               = True if input("Vuoi eseguire la simulazione in un canale rumoroso? ('n' / 'y') ") == "y" else False

    if real:
        # noise_model    = noise_model_from_json(json_file = "imbq_backends_properties/ibm_brisbane_properties.json", gate_error = False, warnings = False)[1]
        noise_model    = fiber_optic_channel()
        backend        = AerSimulator(noise_model = noise_model) 
    else:
        backend        = AerSimulator()

    protocol           = E91(backend = backend, eavesdropper = eavesdropper, verbose = True) 
    results            = protocol.run()

    alice_key          = ''.join(results['alice_key'])
    bob_key            = ''.join(results['bob_key'])
    eve_key_from_alice = ''.join(results['eve_key_from_alice'])
    eve_key_from_bob   = ''.join(results['eve_key_from_bob'])

    plaintext = '''
    Se ni' mondo esistesse un po' di bene
    e ognun si honsiderasse suo fratello
    ci sarebbe meno pensieri e meno pene
    e il mondo ne sarebbe assai più bello.
    '''

    # cifriamo il testo con la chiave di Alice
    alice_sended_ciphertext = xorencrypt(key = alice_key, text = plaintext)

    # decifriamo con la chiave di Bob (che dovrebbe essere uguale a quella di Alice nel caso ideale)
    bob_received_plaintext = xorencrypt(key = bob_key, text = alice_sended_ciphertext)
    print(f'\n>>> Tentativo di decifrazione di Bob\n{bob_received_plaintext.decode("utf-8")}')

    if eavesdropper:

        eve_received_plaintext_from_alice_key = xorencrypt(key = eve_key_from_alice, text = alice_sended_ciphertext)
        print(f'\n>>> Tentativo di decifrazione di Eve con la chiave di Alice\n{eve_received_plaintext_from_alice_key.decode("utf-8")}')

        eve_received_plaintext_from_bob_key = xorencrypt(key = eve_key_from_bob, text = alice_sended_ciphertext)
        print(f'\n>>> Tentativo di decifrazione di Eve con la chiave di Bob\n{eve_received_plaintext_from_bob_key.decode("utf-8")}')

def Simulate_EE91(datatransformer: any, classifiers: list):

    '''
    Simulazione del protocollo E-E91 di distribuzione di chiavi quantistiche con rilevamento di intercettatori tramite machine learning.
    Si può scegliere se eseguire la simulazione in un canale ideale o rumoroso, e se simulare o meno la presenza di un intercettatore.
    In quest'ultimo caso, si può vedere come l'intercettazione vada a compromettere la correttezza della chiave condivisa tra Alice e Bob.
    In ogni caso, si può vedere come l'intercettatore non riesca a decifrare il messaggio cifrato con la chiave di Alice.
    Inoltre, se si esegue la simulazione in un canale rumoroso, i classificatori di machine learning cercheranno di rilevare la presenza di un intercettatore.
    
    Per i classificatori, si possono usare quelli salvati in precedenza in formato .pkl/pth, oppure crearne di nuovi.
    Nel mega-notebook, andare in "Ricerca di un'intercettatore nell'E91 con il machine learning > Training e testing di un modello per il riconoscimento di un'intercettatore in un canale rumoroso":
    lì si possono creare nuovi modelli, testarli e salvarli.
    In quest'ultimo caso, bisogna salvare i modelli nella cartella "E-E91/models" con i nomi usati qui sotto.
    '''

    eavesdropper       = True if input("Vuoi eseguire la simulazione di una intercettazione? ('n' / 'y') ") == "y" else False
    real               = True if input("Vuoi eseguire la simulazione in un canale rumoroso? ('n' / 'y') ") == "y" else False

    if real:
        # backend_model  = "imbq_backends_properties/ibm_brisbane_properties.json"
        # _, noise_model = noise.noise_model_from_json(json_file = backend_model, gate_error = False, warnings = False) 
        noise_model    = fiber_optic_channel()
        backend        = AerSimulator(noise_model = noise_model)
    else:
        backend        = AerSimulator()

    protocol           = EE91(backend = backend, datatransformer = datatransformer, classifiers = classifiers, eavesdropper = eavesdropper, verbose = True) 
    results            = protocol.run()

    alice_key          = ''.join(results['alice_key'])
    bob_key            = ''.join(results['bob_key'])
    eve_key_from_alice = ''.join(results['eve_key_from_alice'])
    eve_key_from_bob   = ''.join(results['eve_key_from_bob'])

    plaintext = '''
    Se ni' mondo esistesse un po' di bene
    e ognun si honsiderasse suo fratello
    ci sarebbe meno pensieri e meno pene
    e il mondo ne sarebbe assai più bello.
    '''

    # cifriamo il testo con la chiave di Alice
    alice_sended_ciphertext = xorencrypt(key = alice_key, text = plaintext)

    # decifriamo con la chiave di Bob (che dovrebbe essere uguale a quella di Alice nel caso ideale)
    bob_received_plaintext = xorencrypt(key = bob_key, text = alice_sended_ciphertext)
    print(f'\n>>> Tentativo di decifrazione di Bob\n{bob_received_plaintext.decode("utf-8")}')

    if eavesdropper:

        eve_received_plaintext_from_alice_key = xorencrypt(key = eve_key_from_alice, text = alice_sended_ciphertext)
        print(f'\n>>> Tentativo di decifrazione di Eve con la chiave di Alice\n{eve_received_plaintext_from_alice_key.decode("utf-8")}')

        eve_received_plaintext_from_bob_key = xorencrypt(key = eve_key_from_bob, text = alice_sended_ciphertext)
        print(f'\n>>> Tentativo di decifrazione di Eve con la chiave di Bob\n{eve_received_plaintext_from_bob_key.decode("utf-8")}')

if __name__ == "__main__":

    if sys.prefix == sys.base_prefix:
        print(">>> Attenzione: ambiente virtuale non è attivo. Alcuni moduli potrebbero non essere trovati: assicurati di aver attivato l'ambiente virtuale con le librerie specificate in \"requirements.txt\" prima di eseguire questo script.")
        exit(1)

    from qiskit_aer import AerSimulator
    from package.utils import xorencrypt
    from package.noise import fiber_optic_channel
    from package import classifiers
    from package.E91 import E91
    from package.EE91 import EE91

    while True:

        try:
            protocol_choice = int(input("\nSimulazione del protocollo E91 ed E-E91 con Qiskit.\nQuale protocollo vuoi eseguire? (0 = E91, 1 = E-E91, 2 = Arresta il programma) "))
        except ValueError:
            print(">>> Input non valido. Inserisci 0, 1 o 2.")
            continue

        if protocol_choice == 0:
            Simulate_E91()  

        elif protocol_choice == 1:
            
            datatransformer    = classifiers.DataTransformer(imputer = "data/models/imputer_SimpleImputer.pkl", scaler = "data/models/scaler_RobustScaler.pkl")
            rf                 = classifiers.Classifier.load_model(filepath = "data/models/model_RandomForestClassifier.pkl")
            svc                = classifiers.Classifier.load_model(filepath = "data/models/model_SVC.pkl")
            sgd                = classifiers.Classifier.load_model(filepath = "data/models/model_SGDClassifier.pkl")
            ada                = classifiers.Classifier.load_model(filepath = "data/models/model_AdaBoostClassifier.pkl")
            net                = classifiers.NeuralNetwork.load_model(filepath = "data/models/model_NeuralNetwork.pth")
            mhn                = classifiers.ModernHopfieldNetwork.load_model(filepath = "data/models/model_ModernHopfieldNetwork.pth")

            Simulate_EE91(datatransformer = datatransformer, classifiers = [rf, svc, sgd, ada, net, mhn])

        elif protocol_choice == 2:
            print(">>> Arresto del programma.")
            break

        else:
            print(">>> Input non valido. Inserisci 0, 1 o 2.")
            continue
