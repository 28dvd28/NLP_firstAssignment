# Primo classificatore

Per lo sviluppo di questo progetto si è deciso di creare 4 file differenti ognuno dei quali da eseguire separatamente nell'ordine che verrà poi spiegato in seguito. 
Prima di procedere è importante però che vengano installate nel proprio sistema le seguenti librerie python:
1. math
2. nltk
3. os
4. json
5. numpy
6. wikipediaapi

Per la classificazione del testo si è deciso di adoperare la logistic regression, implementata come segue.

## Download del corpus
Prima fase è il download del corpus, che per essere ottenuto è necessario eseguire il file **corpus_read.py** all'interno del quale sono state sfruttate le api di wikipedia per scaricare un totale di circa 4000 documenti medici e 4000 documenti non medici, scegliendo delle categorie visualizzabili direttamente nel codice. Ovviamente sia per motivi di spazio che di tempo, scaricare l'intero corpus di wikipedia risulta impraticabile. Di questi documenti un 20% verrà destinato a costituire il test set, mentre il resto è l'effettivo dataset di addestramento.

## Training
Nella logistic regression la fase di addestramento prevede la definizione di un vettore di features con realtivi pesi da sfruttare per la classificazione del testo. 
Anziché partire con una definizione dei pesi casuale, procedendo quindi poi con la fase di training nella quale si cerca di ridurre al minimo la cross-entropy sfruttando l'algoritmo stochastic gradient descent. Per fare ciò si è deciso di procedere con la creazione di due Bag of Words contenenti una le parole che appaiono nei documenti medici, mentre l'altra nei documenti non medici, contando per ogni parola il numero di testi in cui appare. Ai BoW ottenuti viene operata una elaborazione così da eliminare le parole comuni. Il tutto implementato nel file **corpus_elaboration.py** che deve essere eseguito per secondo e che andrà a salvare in due documenti di testo le BoW ottenute.

## Cross-entropy
Come deto sulla base dei pesi ottenuti si dovrebbe operare una fase di training che tuttavia non è stato necessario implementare in quanto il valore della cross entropy già con i pesi calcolati nelle BoW è molto bassa. Per verificare ciò all'interno del file **precision_detection.py** è stata implementato il codice che calcola la cross-entropy media sui testi del dataset

## Test classfication
L'ultima fase, per la classificazione di un testo mai visto precedentemente, la si può trovare nel file **test_classification.py** il quale va lanciato per ultimo ed eseguirà la classificazione su circa 800 documenti salvati come test set. Al termine stamperà sul terminale i predicted labels, indicando in verde le previsioni corrette, mentre in rosso quelle erratte. Verranno inoltre calcolare anche i valori della accuratezza, della precisione e del recall. 

## Importante
Per la corretta esecuzione, i file python e le cartelle contenenti il corpus devono essere nella medesima directory, dalla quale eseguire anche i comandi per lanciare i vari file. **Eseguire i file da una directory differente da quelli in cui sono stati salvati darà errore**. 

# Secondo classificatore

Il secondo classificatore usa come il primo il medesimo corpus, quindi lo stesso codice per scaricarlo, ma anche alcune funzioni del file utils, mentre si differenzia per il tipo di addestramento. Se il primo otteneva direttamente i valori delle due bag_of_words dal corpus, questo secondo classificatore sfrutta un algoritmo che implementa il **Batch Training**, applicandolo sulla totalità del training corpus. 

## Implementazione

Libreria usate:
1. pickle 
2. tqdm

E' stata per prima cosa creata una classe, visualizzabile nel file **batch_training_utils.py**, la quale è stata creata per poter creare un'istanza della nostra AI addestrabile, così da poterla salvare in un file .dat e conservare i progressi raggiunti nell'addestramento. Per entrare nel dettaglio riguardo l'implementazione, basta guardare il codice che è stato commentato per spiegare ogni parte. In generale il funzionamento prevede:

1. L'inizializzazione. nella quale vengono estratte le features, ergo le parole importanti alla classificazione tra le classi medico e non medico, e viene calcolata e salvata anche la matrice X che contiene il conteggio delle riccorrenze per ogni testo di ogni features. Infine i pesi sono settatti tutti a 0
2. Un metodo richiamabile per l'addestramento, da eseguire in un secondo luogo. Questo per permettere eventualmente di poter rieseguire il training, affinandolo maggiormente
3. Un metodo per eseguire il test

Il file dove il tutto è stato implementato è **batch_training.py**, che una volta lanciato vi si presenterà come una console dalla quale inserire vari comandi, spiegati al lancio e comunque ottenibili con comando *help*, per poter eseguire le operazioni di cui sopra. La creazione di un'istanza di questa AI richiede un tempo di circa 5 minuti, mentre l'addestramento vero e proprio richiede circa 30 secondi per raggiungere una cross-entropy settata a 0.1, per questo è stata condivisa un'istanza già allenata nel file **logic_regression_trainer.dat**. È possibile eliminare il file, oppure digitare n quando appare a terminale la notifica *Istanza del trainer già presente, scaricare?* creando una nuova istanza e sovrascrivendo la precedente. Se si volesse invece resettare solo il training e rieseguirlo, caricare l'istanza presente e inserire il comando *reset-learning*
