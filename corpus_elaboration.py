import os
import json

from utils import text_elaboration


bag_of_med_words = {}
bag_of_nonmed_words = {}


def corpus_word_frequency(article_word_set, bow):
    for word in article_word_set:
        if word in bow:
            bow[word] += 1
        else:
            bow[word] = 1
    


if __name__ == "__main__":

    cartella = "medical_train_set"
    corpus_med = os.listdir(cartella)

    for file_name in corpus_med:

        path = os.path.join(cartella, file_name)

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                article_word_set = set(text_elaboration(content)) #creo un set così da contenere solo una volta ogni parola che compare
                corpus_word_frequency(article_word_set, bag_of_med_words)        
    
    print("Terminata bow di documenti medici")


    cartella = "non_medical_train_set"
    corpus_nonmed = os.listdir(cartella)

    for file_name in corpus_nonmed:

        path = os.path.join(cartella, file_name)

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                article_word_set = set(text_elaboration(content)) 
                corpus_word_frequency(article_word_set, bag_of_nonmed_words)

    print("Terminata bow di documenti non medici")
    print("Inizio elaborazione sulle BoWs")
    
    med_bow = bag_of_med_words.copy()
    nonmed_bow = bag_of_nonmed_words.copy()

    for word in bag_of_nonmed_words:
        if word in med_bow and bag_of_nonmed_words[word] > 10:
            del med_bow[word]

    for word in bag_of_med_words:
        if word in nonmed_bow and bag_of_med_words[word] > 10:
            del nonmed_bow[word]

    med_bow = {chiave: valore for chiave, valore in med_bow.items() if valore >= 5}
    nonmed_bow = {chiave: valore for chiave, valore in nonmed_bow.items() if valore >= 5} 
    
    med_bow = dict(sorted(med_bow.items(), key=lambda x: x[1], reverse=True))
    nonmed_bow = dict(sorted(nonmed_bow.items(), key=lambda x: x[1], reverse=True))
    
    percorso_del_file = "medical_bag_of_words.txt"
    with open(percorso_del_file, "w") as file:
        json.dump(med_bow, file, indent=4)

    percorso_del_file = "no_medical_bag_of_words.txt"
    with open(percorso_del_file, "w") as file:
        json.dump(nonmed_bow, file, indent=4)
