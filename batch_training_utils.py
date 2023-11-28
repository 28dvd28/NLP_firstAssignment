import os
import numpy as np
from tqdm import tqdm

from utils import text_elaboration
from utils import cross_entropy


# funzione che serve per verificare quali parole appaiono in un testo, passato alla funzione in 
# formato di set, quindi per ogni parola che appare, si aggiorna il bow che contiene per ogni parola
# il numero di documenti in cui una parola appare
def corpus_word_frequency(article_word_set: set, bow: dict):
    for word in article_word_set:
        if word in bow:
            bow[word] += 1
        else:
            bow[word] = 1

    return bow

# data la lista di parole di un testo e la lista di parole utili per la classificazione del testo in medico o non medico, 
# ritorna una lista in cui nella posizione i-esima conta quante volte la parola i-esima appare nel testo
# la lista ritoranta verrà usata per la costruzione della matrice X
def text_count_word(text_elab: list, words_list: list):

    # l'uso di un dizionario rende l'esecuzione molto più veloce
    dictionary =  dict(zip(words_list, np.zeros(len(words_list)))) 

    for w in text_elab:
        if w in dictionary:
            dictionary[w] += 1    
    
    return list(dictionary.values()) # ritorno soltanto i valori del dizionario


# X è una matrice di m righe e f colonne, dove m è il numero di testi da classificare 
# mentre f è il numero delle feature necessarie alla classificazione, nel nostro caso parole
# con il rispettivo numero di occorrenze nel testo corrispondente. w sono i pesi di ogni parola
# mentre lenght è una lista che contiene per tutti gli m testi la rispettiva lunghezza 
def text_classification(X : np.array, w : np.array, text_lengts: list):

    bias = -1 # bias settata a -1 così che se non viene trovata alcuna parola, il testo venga considerato non medico
    z = X.dot(w) # il prodotto tra una matrice e il vettore dei pesi restituisce un array in cui in ogni posizione i corrisponde il calcolo fatto per il testo i-esimo

    for i in range(z.shape[0]):
        z[i] = (z[i] + bias) / text_lengts[i] # divido per la lunghezze del testo così da avere un valore pesato in rapporto alla lunghezza
        z[i] = 1 / (1 + np.exp(-z[i])) # applicazione della regressione logistica
    return z


# funzione per la classificazione dei testi di test, differente dalla precedente e più veloce
# data un dizionario di parole - peso, se la parola è presente nel testo, per ogni occorrenza viene sommato
# il corrispettivo peso, si divide per la lunghezza del testo e infine si applica la logistic regression
def test_text_classification(text_elaborated : list, bow : dict):

    z = 0
    bias = -1

    for word in text_elaborated:
        if word in bow:
            z += bow[word]

    z = (z + bias) / len(text_elaborated)
    probability_of_medical = 1 / (1 + np.exp(-z))
    return probability_of_medical

# funzione per leggere tutti i file, i cui path sono contenuti all'interno della variabile passata come parametro
def corpus_read(file_names: list):

    texts = []

    barra = tqdm(total=len(file_names), desc='Lettura testi', position=0, leave=False)

    for file_name in file_names:

        path = file_name

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                article = text_elaboration(content) # eseguo l'elaborazione sul testo, eliminando stop word, tokkenizzando e lemmatizzando
                texts.append(article)

        barra.update(1)
    
    barra.close()
    return texts


# la seguente funzione è necessaria per estrarre dal corpus le feature importanti per la classificazione,
# che nel nostro caso sono costituite dalle parole fondamentali, necessarie per la distinzione tra le due classi
def bow_elaboration(corpus: list, labels: list):

    barra = tqdm(total=len(corpus), desc='Elaborazione parole', position=0, leave=False)

    # vengono usati due dizionari perché così sarà possibile riconoscere ed eliminare le parole 
    # troppo comuni ad entrambi inutili per la classificazione
    bag_of_med_words = {}
    bag_of_nonmed_words = {}
        
    for article in corpus:

        article_word_set = set(article)  # creo un set così da contenere solo una volta ogni parola che compare

        if labels[corpus.index(article)] == 1:
            bag_of_med_words = corpus_word_frequency(article_word_set, bag_of_med_words)
        else:
            bag_of_nonmed_words = corpus_word_frequency(article_word_set, bag_of_nonmed_words)

        barra.update(1)

    med_bow = bag_of_med_words.copy()
    nonmed_bow = bag_of_nonmed_words.copy()
    
    for word in bag_of_nonmed_words:
        if word in med_bow and bag_of_nonmed_words[word] > 10: # elimino la parola solo se presente in entrambi con una frequenza significativa
            del med_bow[word]

    for word in bag_of_med_words:
        if word in nonmed_bow and bag_of_med_words[word] > 10:
            del nonmed_bow[word]

    testi_medici = sum(x for x in labels if x == 1)
    testi_nonmedici = sum(x+1 for x in labels if x == 0)

    # elimino la lunga coda di elementi non utili alla classificazione perchè troppo rari
    med_bow = {chiave: np.log(valore/testi_medici) for chiave, valore in med_bow.items() if valore >= 5}
    nonmed_bow = {chiave: np.log(valore/testi_nonmedici) for chiave, valore in nonmed_bow.items() if valore >= 5} 

    med_bow = dict(sorted(med_bow.items(), key=lambda x: x[1], reverse=True))
    nonmed_bow = dict(sorted(nonmed_bow.items(), key=lambda x: x[1], reverse=True))

    barra.close()


    words = list( {**med_bow, **nonmed_bow} ) # calcolare la concatenzazione dei due dizionari è fondamentale per evitare la presenza di doppioni
    return words    

# funzione che restituisce la matrice di tutte le occorrenze di ogni testo
# e la lunghezza totale di ogni testo misurata con il numero di parole
def corpus_analisys( corpus: list, words: list):

    barra = tqdm(total=len(corpus), desc='Analisi del corpus', position=0, leave=False)

    X = []
    lengths = []

    for text_elaborated in corpus:

        X.append(text_count_word(text_elaborated, words)) # aggiungo le occorrenze delle parole per il testo i-esimo
        lengths.append(len(text_elaborated)) # aggiungo la lunghezza del testo i-esimo
        barra.update(1)
    
    barra.close()

    X = np.array(X)
    length_texts = np.array(lengths)

    return X, length_texts    



# l'idea di base della creazione di questa classe è creare un'istanza allenabile e salvabile, così che anzichè dover 
# eseguire più codici è abbastanza eseguire il medesimo, con vari metodi. L'istanza può essere salvata in un file dat 
# così da poter conservare i risultati dell'addestramento
class logisticRegressionTrainer:

    def __init__(self, files_names:list, labels:list) -> None:

        print(">>> INIZIALIZZAZZIONE <<<")

        # nella fase di inizializzazione memorizzo gli articoli, estraggo le features 
        # e calcolo la lunghezza di ogni testo, ma anche la matrice X necessari poi per la fase 
        # di training

        self.files_names = files_names
        self.saved_corpus = corpus_read(files_names)
        self.labels = labels
        self.words = bow_elaboration(self.saved_corpus, labels)
        self.X, self.length_texts = corpus_analisys(self.saved_corpus, self.words)
        self.weights = np.zeros(len(self.words))
    
    # funzione che presi in input le occorrenze delle parole per ogni testo e i 
    # rispettivi pesi ritorna una bow con i pesi corretti
    def training( self ):

        print(">>> INIZIO TRAINING <<<")

        obbiettivo_ce = 0.1
        while True:

            input_utente = input(">> Inserire cross-entropy da raggiungere: ")

            if input_utente == "":
                break
            else:
                try:
                    obbiettivo_ce = float(input_utente)
                    break
                except ValueError:
                    print(">> Valore inserito non valido!!")


        percentage = 0

        while True:
            prediction_values = text_classification(self.X, self.weights, self.length_texts)
            cross_entropy_val = cross_entropy(prediction_values, self.labels)

            if cross_entropy_val > obbiettivo_ce:

                if percentage == 0:
                    diff = cross_entropy_val - obbiettivo_ce
                    barra = tqdm(total=diff, desc='Lettura testi', position=0, leave=False)
                    percentage = cross_entropy_val
                else:
                    barra.update(percentage-cross_entropy_val)
                    percentage = cross_entropy_val
            elif percentage == 0:
                print(">> Sistema presenta cross-entropy più bassa di quella inserita")
                print(">>> FINE TRAINING <<<")
                break
            else:
                barra.close()
                print(">>> FINE TRAINING <<<")
                break

            # in fase di addestramento viene utilizzato un algoritmo che implementa il gradient descent, dove dal peso attribuito ad ogni feature
            # viene sottrato il valore del gradiente calcolato per quella feature. In particolare si sta svolgendo un batch training che calcola 
            # il gradiente per l'intero dataset e poi esegue l'aggiornamento
            cost = (prediction_values - np.array(self.labels)).dot(self.X) / len(self.labels)
            self.weights = self.weights - 100 * cost
    

    # funzione che riceve in input una lista di path di test con i rispettivi labels, 
    # ne calcola un valore prevvisto e in base ai labels corretti stampa alcune informazioni statistiche
    def classification(self, test_texts: list, labels):

        dictionary = dict(zip(self.words, self.weights))
        prediction_values = []

        if np.all(self.weights == 0):
           print(">> Modello non addestrato!!")
           return
        
        print(">>> CLASSIFICAZIONE <<<")

        barra = tqdm(total=len(test_texts), desc='Preddizione test', position=0, leave=False)

        for file_name in test_texts:

            path = file_name

            if os.path.isfile(path):

                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    text_elaborated = text_elaboration(content) # viene fatta la medesima elaborazione fatta anche per il train
                    if test_text_classification(text_elaborated, dictionary) > 0.5:
                        prediction_values.append(1)
                    else:
                        prediction_values.append(0)
            
            barra.update(1)
        
        barra.close()

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(prediction_values)):
            pred = prediction_values[i]
            real = labels[i]

            if pred == 1:
                if pred == real:
                    true_positive += 1
                else:
                    false_positive +=1
            elif pred == 0:
                if pred == real:
                    true_negative += 1
                else:
                    false_negative += 1
        
        confusion_matrix = [ [true_positive, false_positive],[false_negative, true_negative]]

        print("\n===================== STATISTICHE ======================\n")
        confusion_matrix = np.array(confusion_matrix)
        
        print("CONFUSION MATRIX   |   Gold Positive   |   Gold negative")
        print(f"System positive    |        {confusion_matrix[0, 0]}        |         {confusion_matrix[0, 1]}")
        print(f"System negative    |         {confusion_matrix[1, 0]}        |        {confusion_matrix[1, 1]}")

        accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1]) / (confusion_matrix[0,0] + confusion_matrix[1,0] + confusion_matrix[0,1] + confusion_matrix[1,1])
        accuracy = round(accuracy * 100, 3)

        precision = confusion_matrix[0, 0] / (confusion_matrix[0,0] + confusion_matrix[0,1])
        precision = round(precision * 100, 3)

        recall = confusion_matrix[0, 0] / (confusion_matrix[0,0] + confusion_matrix[1,0])
        recall = round(recall * 100, 3)

        print()
        print(f"ACCURACY:     {accuracy} %")
        print(f"PRECISION:   {precision} %")
        print(f"RECALL:      {recall} %")
        print()

    def resetTrain( self ):
        self.weights = np.zeros(len(self.words))

    def getX( self ):
        return self.X
    
    def getBoW ( self ):

        bow = dict(zip(self.words, self.weights))
        return bow 
