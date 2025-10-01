import sys

def Simulate_E91():

    '''
    Simulation of the E91 quantum key distribution protocol.
    You can choose whether to run the simulation on an ideal or noisy channel,
    and whether to simulate the presence of an eavesdropper.
    In the latter case, you can see how the interception compromises the correctness
    of the shared key between Alice and Bob.
    In any case, you can observe that the eavesdropper cannot decrypt the message
    encrypted with Alice's key.
    '''

    eavesdropper       = True if input("Do you want to simulate an eavesdropping attack? ('n' / 'y') ") == "y" else False
    real               = True if input("Do you want to run the simulation on a noisy channel? ('n' / 'y') ") == "y" else False

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
    If there were some goodness in the world
    and everyone considered one another as brother,
    there would be fewer worries and fewer pains,
    and the world would be much more beautiful.
    '''

    # encrypt the text with Alice's key
    alice_sended_ciphertext = xorencrypt(key = alice_key, text = plaintext)

    # decrypt with Bob's key (which should be equal to Alice's in the ideal case)
    bob_received_plaintext = xorencrypt(key = bob_key, text = alice_sended_ciphertext)
    print(f'\n>>> Bob decryption attempt\n{bob_received_plaintext.decode("utf-8")}')

    if eavesdropper:

        eve_received_plaintext_from_alice_key = xorencrypt(key = eve_key_from_alice, text = alice_sended_ciphertext)
        print(f'\n>>> Eve decryption attempt with Alice\'s key\n{eve_received_plaintext_from_alice_key.decode("utf-8")}')

        eve_received_plaintext_from_bob_key = xorencrypt(key = eve_key_from_bob, text = alice_sended_ciphertext)
        print(f'\n>>> Eve decryption attempt with Bob\'s key\n{eve_received_plaintext_from_bob_key.decode("utf-8")}')

def Simulate_EE91(datatransformer: any, classifiers: list):

    '''
    Simulation of the E-E91 quantum key distribution protocol with eavesdropper detection via machine learning.
    You can choose whether to run the simulation on an ideal or noisy channel,
    and whether to simulate the presence of an eavesdropper.
    In the latter case, you can see how the interception compromises the correctness
    of the shared key between Alice and Bob.
    In any case, you can observe that the eavesdropper cannot decrypt the message
    encrypted with Alice's key.
    Moreover, if the simulation is run on a noisy channel, the machine learning classifiers
    will attempt to detect the presence of an eavesdropper.
    
    For classifiers, you can use models previously saved in .pkl/.pth format, or create new ones.
    In the mega-notebook, go to "Detecting an eavesdropper in E91 with machine learning > Training and testing a model for eavesdropper recognition in a noisy channel":
    there you can create new models, test them and save them.
    In that case, models must be saved to the "E-E91/models" folder with the names used below.
    '''

    eavesdropper       = True if input("Do you want to simulate an eavesdropping attack? ('n' / 'y') ") == "y" else False
    real               = True if input("Do you want to run the simulation on a noisy channel? ('n' / 'y') ") == "y" else False

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
    If there were some goodness in the world
    and everyone considered one another as brother,
    there would be fewer worries and fewer pains,
    and the world would be much more beautiful.
    '''

    # encrypt the text with Alice's key
    alice_sended_ciphertext = xorencrypt(key = alice_key, text = plaintext)

    # decrypt with Bob's key (which should be equal to Alice's in the ideal case)
    bob_received_plaintext = xorencrypt(key = bob_key, text = alice_sended_ciphertext)
    print(f'\n>>> Bob decryption attempt\n{bob_received_plaintext.decode("utf-8")}')

    if eavesdropper:

        eve_received_plaintext_from_alice_key = xorencrypt(key = eve_key_from_alice, text = alice_sended_ciphertext)
        print(f'\n>>> Eve decryption attempt with Alice\'s key\n{eve_received_plaintext_from_alice_key.decode("utf-8")}')

        eve_received_plaintext_from_bob_key = xorencrypt(key = eve_key_from_bob, text = alice_sended_ciphertext)
        print(f'\n>>> Eve decryption attempt with Bob\'s key\n{eve_received_plaintext_from_bob_key.decode("utf-8")}')

if __name__ == "__main__":

    if sys.prefix == sys.base_prefix:
        print(">>> Warning: virtual environment is not active. Some modules may not be found: make sure to activate the virtual environment with the libraries specified in \"requirements.txt\" before running this script.")
        exit(1)

    from qiskit_aer import AerSimulator
    from package.utils import xorencrypt
    from package.noise import fiber_optic_channel
    from package import classifiers
    from package.E91 import E91
    from package.EE91 import EE91

    while True:

        try:
            protocol_choice = int(input("\nSimulation of the E91 and E-E91 protocols with Qiskit.\nWhich protocol would you like to run? (0 = E91, 1 = E-E91, 2 = Exit) "))
        except ValueError:
            print(">>> Invalid input. Enter 0, 1 or 2.")
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
            print(">>> Exiting program.")
            break

        else:
            print(">>> Invalid input. Enter 0, 1 or 2.")
            continue
