import os
import json

from utils import text_elaboration
from utils import text_classification
from utils import cross_entropy


if __name__ == "__main__":

    cartella1 = "medical_train_set"
    test_texts1 = os.listdir(cartella1)

    cartella2 = "non_medical_train_set"
    test_texts2 = os.listdir(cartella2)

    medical_bag_of_words = {}
    with open("medical_bag_of_words.txt", "r") as file:
        medical_bag_of_words = json.load(file)

    nomedical_bag_of_words = {}
    with open("no_medical_bag_of_words.txt", "r") as file:
        nomedical_bag_of_words = json.load(file)


    labels = []
    prediction_value = []

    print("Inizio labelizzazione")

    #per ogni file del train set eseguo la labelizzazione, prima per i testi medici, poi per i testi non medici
    for file_name in test_texts1:

        path = os.path.join(cartella1, file_name)

        if os.path.isfile(path):

            labels.append(1) #per ogni file letto valido aggiungo 1 alla lista contenente i label corretti

            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                text_elaborated = text_elaboration(content)

            prediction_value.append(text_classification(text_elaborated, medical_bag_of_words, nomedical_bag_of_words))

    print("Terminato label medici")


    for file_name in test_texts2:
        
        path = os.path.join(cartella2, file_name)

        if os.path.isfile(path):

            labels.append(0) #per ogni file letto valido aggiungo 0 alla lista contenente i label corretti

            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                text_elaborated = text_elaboration(content)

            prediction_value.append(text_classification(text_elaborated, medical_bag_of_words, nomedical_bag_of_words))

    print("Terminato label non medici")

    cross_entropy_val = cross_entropy(prediction_value, labels) # calcolo della cross entropy tra le probabilità calcolate e i valori reali
    
    print("Cross_entropy media è: ", cross_entropy_val)
