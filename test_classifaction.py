import os
import json
import numpy as np

from utils import text_elaboration
from utils import text_classification

class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

if __name__ == "__main__":

    cartella1 = "medical_test_set"
    medical_test_texts = os.listdir(cartella1)

    cartella2 = "non_medical_test_set"
    non_medical_test_texts = os.listdir(cartella2)

    med_labels = []
    nonmed_labels = []

    medical_bag_of_words = {}
    with open("medical_bag_of_words.txt", "r") as file:
        medical_bag_of_words = json.load(file)

    nomedical_bag_of_words = {}
    with open("no_medical_bag_of_words.txt", "r") as file:
        nomedical_bag_of_words = json.load(file)

    #eseguo la labelizzazione su tutti i file di test, prima medici e poi non medici, 
    #per ogni file valido aggiungo alla rispettiva lista il valore indicante il corretto label per quel file

    med_predicted_labels = []
    for file_name in medical_test_texts:
        path = os.path.join(cartella1, file_name)

        if os.path.isfile(path):

            med_labels.append(1)

            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                text_elaborated = text_elaboration(content)

            #la logistic regression restituisce un valore tra 0 e 1 che deve essere però valutato per ottenere un valore che sia 0 (non medico)
            #o 1 (medico). Lo si fa valutando se tale probabilità è maggiore di 0.5
            if text_classification(text_elaborated, medical_bag_of_words, nomedical_bag_of_words) > 0.5:
                med_predicted_labels.append(1)
            else:
                med_predicted_labels.append(0)
    
    nonmed_predicted_labels = []
    for file_name in non_medical_test_texts:
        path = os.path.join(cartella2, file_name)

        if os.path.isfile(path):    

            nonmed_labels.append(0)

            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                text_elaborated = text_elaboration(content)

            if text_classification(text_elaborated, medical_bag_of_words, nomedical_bag_of_words) > 0.5:
                nonmed_predicted_labels.append(1)
            else:
                nonmed_predicted_labels.append(0)

    predicted_labels = med_predicted_labels + nonmed_predicted_labels
    labels = med_labels + nonmed_labels #ottengo la lista completa

    #output dei label predetti e della accuratezza, precisione e recall
    print("Predicted labels:")
    for x in range (len(predicted_labels)):

        if(predicted_labels[x] == labels[x]):
            print(bcolors.GREEN + bcolors.BOLD + str(predicted_labels[x]) + bcolors.ENDC, end=" ")
        else:
            print(bcolors.RED + str(predicted_labels[x]) + bcolors.ENDC, end=" ")

    confusion_matrix = [[sum(x for x in med_predicted_labels if x == 1), sum(x+1 for x in med_predicted_labels if x == 0)] ,
                        [sum(x for x in nonmed_predicted_labels if x == 1), sum(x+1 for x in nonmed_predicted_labels if x == 0)]]
    
    print("\n")
    confusion_matrix = np.array(confusion_matrix)
    print("Confusion matrix:\n", confusion_matrix)

    accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1]) / (confusion_matrix[0,0] + confusion_matrix[1,0] + confusion_matrix[0,1] + confusion_matrix[1,1])
    print("Accuracy: ", accuracy)

    precision = confusion_matrix[0, 0] / (confusion_matrix[0,0] + confusion_matrix[1,0])
    print("Precision: ", precision)

    recall = confusion_matrix[0, 0] / (confusion_matrix[0,0] + confusion_matrix[0,1])
    print("Recall: ", recall)
