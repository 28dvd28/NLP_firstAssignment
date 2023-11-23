import os
import json

from utils import text_elaboration


bag_of_med_words = {}
bag_of_nonmed_words = {}

#funzione che aggiorna le bag of words dato un testo del training corpus
def corpus_word_frequency(article_word_set, bow):
    for word in article_word_set:
        if word in bow:
            bow[word] += 1
        else:
            bow[word] = 1
    


if __name__ == "__main__":

    cartella = "medical_train_set"
    corpus_med = os.listdir(cartella)


    #per ogni file, una volta effettuata l'elaborazione, si crea un set, così che per ogni parola la si conterà al massimo una volta all'interno
    #di un testo. In questo modo si conta in quanti testi tale parola compare
    for file_name in corpus_med:

        path = os.path.join(cartella, file_name)

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                article_word_set = set(text_elaboration(content))  #creo un set così da contenere solo una volta ogni parola che compare
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

    #da ogni bow si eliminano le parole presenti anche nell'altra in un numero sufficientemente grande, altrimenti c'è il rischio che una
    #parola poco presente in uno venga cancellata dall'altro nonostante in questo sia importante
    for word in bag_of_nonmed_words:
        if word in med_bow and bag_of_nonmed_words[word] > 10:
            del med_bow[word]

    for word in bag_of_med_words:
        if word in nonmed_bow and bag_of_med_words[word] > 10:
            del nonmed_bow[word]

    #eliminazione della coda di valori poco presenti e quindi inutili alla classificazione (parole eliminate in questo modo più di 80000)
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
