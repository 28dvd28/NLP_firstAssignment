import os
import json
import pickle

from batch_training_utils import logisticRegressionTrainer

# se chiamata salva o aggiorna l'istanza della classe d'allenamento 
def updateTrainSession():
    with open('logic_regression_trainer.dat', 'wb') as file:
        pickle.dump(logistic_regression_trainer, file)

# scarica un'istanza già presente
def downloadTrainSession():
    with open('logic_regression_trainer.dat', 'rb') as file:
        logistic_regression_trainer = pickle.load(file)
    return logistic_regression_trainer


def print_help():

    print("\ttraining:")
    print("\t\tavvia il training, verrà richiesto di inserire un valore della cross")
    print("\t\tentropy che di default sarà settata ad uno se non inserito")

    print("\ttest-text:")
    print("\t\tavvia il test")

    print("\tget-training-data:")
    print("\t\trestituisce l'elenco delle parole con i rispettivi pesi calcolati")

    print("\treset-learning:")
    print("\t\tripristina a 0 l'apprendimento generato")

    print("\thelp:")
    print("\t\tstampa i comandi della console del trainer")

    print("\tclose:")
    print("\t\ttermina l'esecuzione")

    print("\tclear")
    print("\t\tpulisce il terminale")
    print()


if __name__ == "__main__":
        
    input_utente = ""

    if os.path.exists('logic_regression_trainer.dat'):

        while True:
            input_utente = input("[TRAINING CONSOLE] > Istanza del trainer già presente, scaricare? [y/n] ")
            if(input_utente == "y" or input_utente == "n"):
                break
    
    if input_utente == "y" and os.path.exists('logic_regression_trainer.dat'):
        logistic_regression_trainer = downloadTrainSession() # se presente scarico l'istanza già presente

    else:

        # ottengo i path di tutto il training corpus, ne calcolo i labels e sulla
        # base di ciò inizializzo il trainer

        cartella1 = "medical_train_set"
        test_texts1 = os.listdir(cartella1)

        cartella2 = "non_medical_train_set"
        test_texts2 = os.listdir(cartella2)

        file_names = []
        labels = []

        for file in test_texts1:

            path = os.path.join(cartella1, file)
            if os.path.isfile(path):
                file_names.append(path)
                labels.append(1)
        
        for file in test_texts2:

            path = os.path.join(cartella2, file)
            if os.path.isfile(path):
                file_names.append(path)
                labels.append(0)

        logistic_regression_trainer = logisticRegressionTrainer(file_names, labels)
        updateTrainSession()

    print("[TRAINING CONSOLE] >  Comandi disponibili: ")
    print_help()

    while True:

        input_utente = input("[TRAINING CONSOLE] > ")

        if input_utente == "training":

            logistic_regression_trainer.training()
            updateTrainSession() # aggiorno l'istanza salvata
        
        elif input_utente == "test-text":

            #ottengo tutti i path dei file di test e ne calcolo il label reale
            cartella1 = "medical_test_set"
            medical_test_texts = os.listdir(cartella1)

            cartella2 = "non_medical_test_set"
            non_medical_test_texts = os.listdir(cartella2)

            test_file_names = []
            test_labels = []    

            for file_name in medical_test_texts:

                path = os.path.join(cartella1, file_name)

                if os.path.isfile(path):

                    test_file_names.append(path)
                    test_labels.append(1)
            
            for file_name in non_medical_test_texts:

                path = os.path.join(cartella2, file_name)

                if os.path.isfile(path):    

                    test_file_names.append(path)
                    test_labels.append(0)

            logistic_regression_trainer.classification(test_file_names, test_labels)

        elif input_utente == "close":
            break
        
        elif input_utente == "get-training-data":

            bow = logistic_regression_trainer.getBoW()
            percorso_del_file = "bow.txt"
            with open(percorso_del_file, "w") as file:
                json.dump(bow, file, indent=4)
            print(">> Bag of words visualizzabile nel file bow.txt")

        elif input_utente == "reset-learning":

            logistic_regression_trainer.resetTrain()
            updateTrainSession()
            print(">> Addestramento dell'istanza azzerrato")

        elif input_utente == "help":

            print(">>  Comandi disponibili: ")
            print_help()

        elif input_utente == "clear":

            os.system('cls' if os.name == 'nt' else 'clear')

        else:
            print(">>  Comando non valido")
