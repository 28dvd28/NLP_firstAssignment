from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import math


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
#sia medico, restituiendone il valore
def text_classification(text_elaborated, class1_bow: dict, class2_bow: dict):

    probability_of_medical = 0
    med_words = []
    nonmed_words = []

    #bias settata a -1 così che nel caso non venga trovata alcuna parola nel testo, 
    #questo possega comunque un valore di probabilità tendente alla seconda classe
    bias = -1

    for word in text_elaborated:
        if word in class1_bow:
            med_words.append(word)
            probability_of_medical += class1_bow[word]
        if word in class2_bow:
            nonmed_words.append(word)
            probability_of_medical -= class2_bow[word]

    probability_of_medical = (probability_of_medical + bias) / len(text_elaborated) #divisione per limitare la dimensione del valore ed impedire l'overflow di math.exp    
    probability_of_medical = 1 / (1 + math.exp(-probability_of_medical))

    return probability_of_medical

#funzione che calcola la cross-entropy media data una lista di valori di probabilità e i corrispettivi labels
def cross_entropy(predicted_values, labels):

    result = 0

    for i in range(len(predicted_values)):

        y = labels[i]
        y_predicted = predicted_values[i]

        if y == 1:
            result += y * math.log(y_predicted) # + (1 - y) * math.log(1 - y_predicted)
        else:
            result += (1 - y) * math.log(1 - y_predicted) # + y * math.log(y_predicted)
    
    result = - result / len(predicted_values)
    return result
