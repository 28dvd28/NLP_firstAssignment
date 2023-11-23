from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import json
import math

def text_elaboration(text):
    global bag_of_words

    punctuation = "!#$%&()*+,-./:;<=>?@[\]^_`{|}~"

    text = text.translate(str.maketrans("", "", punctuation))

    elenco_stopword = stopwords.words('english')
    tokenized_text = word_tokenize(text)
    clean_text = [word for word in tokenized_text if word.lower() not in elenco_stopword]
    lemmatized_text = [WordNetLemmatizer().lemmatize(word) for word in clean_text]

    return lemmatized_text

#funzione che calcola la probabilità che un testo sia medico,
#restituiendone il valore e il conteggio delle parole utili presenti nel testo
def text_classification(text_elaborated):

    probability_of_medical = 0
    words_counter = {}

    for word in text_elaborated:
        if word in medical_bag_of_words:

            if word in words_counter:
                words_counter[word] += 1
            else:
                words_counter[word] = 1

            probability_of_medical += medical_bag_of_words[word]

        if word in nomedical_bag_of_words:

            if word in words_counter:
                words_counter[word] += 1
            else:
                words_counter[word] = 1

            probability_of_medical -= nomedical_bag_of_words[word]        

    probability_of_medical = probability_of_medical / len(text_elaborated) #divisione per limitare la dimensione del valore ed impedire l'overflow di math.exp    
    probability_of_medical = 1 / (1 + math.exp(-probability_of_medical))

    return probability_of_medical, words_counter
    
#funzione che implementa la cost function per un training batch che in questo caso
#utilizza come valore m l'intero dataset
def cross_entropy(predicted_labels, labels):

    print(predicted_labels)

    result = 0

    for i in range(len(predicted_labels)):

        y = labels[i]
        y_predicted = predicted_labels[i]

        if y == 1:
            result += y * math.log(y_predicted) # + (1 - y) * math.log(1 - y_predicted)
        else:
            result += (1 - y) * math.log(1 - y_predicted) #y * math.log(y_predicted)
    
    result = result / len(predicted_labels)
    return abs(result)

#funzione che restituisce un dizionario contenenente per ogni chiave il corrispettivo gradiente 
#le chiavi sono le parole presenti nel BoW che sono state ritrovate nel mio dataset (dovrebbero essere tutte)
def batch_gradient(predicted_labels, labels, word_dict_list):

    result = {}

    for dictionary in word_dict_list:
        for w in dictionary:

            for i in range (len(predicted_labels)):
                y = labels[i]
                y_predicted = predicted_labels[i]

                if w in result:
                    result[w] += (y_predicted - y) * dictionary[w]
                else:
                    result[w] = (y_predicted - y) * dictionary[w]
    
    for r in result:
        result[r] = result[r] / len(predicted_labels)

    return result



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

    while(True):
        
        prediction_value = []
        words_presenti = [] #conterrà una lista di dizionari indicanti per ogni testo, per ogni parola presente nei BoW, il numero di occorrenze nel testo

        print("Inizio labelizzazione")

        for file_name in test_texts1:

            path = os.path.join(cartella1, file_name)

            if os.path.isfile(path):

                labels.append(1)

                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    text_elaborated = text_elaboration(content)
                
                probability_of_medical, words = text_classification(text_elaborated)
                prediction_value.append(probability_of_medical)
                words_presenti.append(words)

        print("Terminato label medici")

        for file_name in test_texts2:
        
            path = os.path.join(cartella2, file_name)

            if os.path.isfile(path):

                labels.append(0)

                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    text_elaborated = text_elaboration(content)
                
                probability_of_medical, words = text_classification(text_elaborated)
                prediction_value.append(probability_of_medical)
                words_presenti.append(words)
        
        print("Terminato label non medici")

        cross_entropy_val = cross_entropy(prediction_value, labels)

        if(cross_entropy_val < 0.1):
            print("Addestramento completato, cross_entropy è: ", cross_entropy_val)
            break
        else:

            print(cross_entropy_val)

            gradient = batch_gradient(prediction_value, labels, words_presenti)
            print("Gradient fatto: ", gradient)
            for word in gradient:
                if word in medical_bag_of_words:
                    medical_bag_of_words[word] -= 1*gradient[word] # moltiplicazione per 1 perchè 1 in questo caso è il nostro iperparametro
                if word in nomedical_bag_of_words:
                    nomedical_bag_of_words[word] += 1*gradient[word] 
                    #essendo una BoW della classe opposta del nostro classificatore, 
                    #questi valori avrebbero valore negativo se posti in una unica bag of words, quindi 
                    #nel loro caso il gradiente deve essere esattamente sommato

            
    percorso_del_file = "medical_bag_of_words.txt"
    with open(percorso_del_file, "w") as file:
        json.dump(medical_bag_of_words, file, indent=4)

    percorso_del_file = "no_medical_bag_of_words.txt"
    with open(percorso_del_file, "w") as file:
        json.dump(nomedical_bag_of_words, file, indent=4)