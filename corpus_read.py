import wikipediaapi
import os
import re

userAgent = "testNLPproject/1.0 (cordiolidavide1@gmail.com)"
wiki = wikipediaapi.Wikipedia(user_agent=userAgent, language = 'en')

titles = []

#funzione che rende un titolo compatibile per essere usato come nome di un file
def makeValidFilename(title):

    # Replace invalid characters with underscores
    validChars = re.sub(r'[^\w\s.-]', '_', title)
    # Remove leading and trailing whitespaces
    validChars = validChars.strip()
    return validChars

#funzione che scarica il testo di una pagina wikipedia, dato il titolo, salvandolo nella cartella indicata
def download_content_page(page_title, cartella):

    global page
    global titles

    page = wiki.page(page_title)

    if page.exists:
        page_text = page.text

    file_name = makeValidFilename(page_title)

    if not os.path.exists(cartella):
        os.makedirs(cartella)

    percorso_file = os.path.join(cartella, file_name + ".txt")

    try:
        with open(percorso_file, "w", encoding="utf-8") as file:
            file.write(page_text)
    except FileNotFoundError as e:
        print(e)


#la lettuara dei titoli appartenenti ad una categoria si svolge attraverso una funzione ricorsiva che itera
#attraverso ogni sottocategoria, salvando i titoli di ognuna fino al raggiungimento di una dimensione n
def category_members_read(categorymembers, n):

    global titles

    for c in categorymembers.values():
            
            if n > 0:

                if c.ns == wikipediaapi.Namespace.CATEGORY:
                    n = category_members_read(c.categorymembers, n)
                elif c.ns == wikipediaapi.Namespace.MAIN:
                    if not c.title in titles:
                        titles.append(c.title)
                        n -= 1
    
    return n
            



if __name__ == "__main__":

    print("Inizio lettura titoli")

    cat = wiki.page("Category:Medical diagnosis")
    category_members_read(cat.categorymembers, 1000)

    cat = wiki.page("Category:Medical terminology")
    category_members_read(cat.categorymembers, 1000)

    cat = wiki.page("Category:Medical specialties")
    category_members_read(cat.categorymembers, 1000)

    cat = wiki.page("Category:Medical treatments")
    category_members_read(cat.categorymembers, 1000)

    print(f"Titoli medici ottenuti: {len(titles)}")
    print("Inzio download")

    #variabile che contiene ogni quanti testi salvarne uno nel test set
    intervall_doc_testset = len(titles) // (len(titles) * 0.2)


    for doc in titles:
        if titles.index(doc) % intervall_doc_testset == 0:
            download_content_page(doc, "medical_test_set")
        else:
            download_content_page(doc, "medical_train_set")

    print("Download testi medici terminato")

    titles = []


    cat = wiki.page("Category:Political movements in Europe")
    category_members_read(cat.categorymembers, 1000)

    cat = wiki.page("Category:Economies")
    category_members_read(cat.categorymembers, 1000)

    cat = wiki.page("Category:Religion")
    category_members_read(cat.categorymembers, 1000)

    cat = wiki.page("Category:Mathematics")
    category_members_read(cat.categorymembers, 1000)

    print(f"Titoli non medicali ottenuti: {len(titles)}")
    print("Inzio download")

    intervall_doc_testset = len(titles) // (len(titles) * 0.2)


    for doc in titles:
        if titles.index(doc) % intervall_doc_testset == 0:
            download_content_page(doc, "non_medical_test_set")
        else:
            download_content_page(doc, "non_medical_train_set")

    print("Download terminato")
