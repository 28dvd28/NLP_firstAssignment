from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy


#funzione per l'elaborazione del testo che procede dapprima eliminando i vari simboli elencati nella variabile punctuation, 
#viene eseguita quindi una tokenizazione, viene ripulito il testo dalle stopword e infine viene eseguita una lemmatization
def text_elaboration(text):
    global bag_of_words

    punctuation = "!#$%&()*+,-./:;<=>?@[\]^_`{|}~"

    text = text.translate(str.maketrans("", "", punctuation))

    elenco_stopword = stopwords.words('english')
    tokenized_text = word_tokenize(text)
    clean_text = [word for word in tokenized_text if word.lower() not in elenco_stopword]
    lemmatized_text = [WordNetLemmatizer().lemmatize(word) for word in clean_text]

    return lemmatized_text

#funzione che calcola la probabilità che un testo 
#sia medico, restituiendone la probabilità calcolata in un valore che andrà da 0 a 1
def text_classification(text_elaborated, class1_bow: dict, class2_bow: dict):

    z = 0

    #bias settata a -1 così che nel caso non venga trovata alcuna parola nel testo, 
    #questo possega comunque un valore di probabilità tendente alla seconda classe
    bias = -1

    for word in text_elaborated:
        if word in class1_bow:
            z += class1_bow[word]
        if word in class2_bow:
            #Tendenzialmente i pesi delle feature che appartengono alla seconda classe dovrebbero avere valore negativo.
            #Essendo salvati però in una BoW differente, con valori positivi, si risolve il problema sottrando tali valori anziché sommarli
            z -= class2_bow[word]

    #divisione per limitare la dimensione del valore ed impedire l'overflow di math.exp, 
    #si usa la lunghezza del testo così da ottenere un valore più preciso in rapporto alla grandezza del testo
    z = (z + bias) / len(text_elaborated) 
    probability_of_medical = 1 / (1 + numpy.exp(-z))

    return probability_of_medical

#funzione che calcola la cross-entropy media data una lista di valori di probabilità e i corrispettivi labels
def cross_entropy(predicted_values, labels):

    result = 0

    for i in range(len(predicted_values)):

        y = labels[i]
        y_predicted = predicted_values[i]

        if y == 1:
            result += - y * numpy.log(y_predicted) # + (1 - y) * math.log(1 - y_predicted)
        else:
            result += - (1 - y) * numpy.log(1 - y_predicted) # + y * math.log(y_predicted)
    
    result = result / len(predicted_values)
    return result
