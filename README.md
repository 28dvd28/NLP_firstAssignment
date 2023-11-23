# Text classification

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
