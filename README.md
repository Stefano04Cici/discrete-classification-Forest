# ICON 25-26

### Diamond Price Prediction & Knowledge Base Reasoning


#### Esame di ingegneria della conoscenza, UniBa, realizzato da: 

- [Io](https://github.com/Hue-Jhan)

- [A.B.](https://github.com/Antob0906)

- [S.C.](https://github.com/Stefano04Cici)

***

### ⚙ Setup iniziale dell'ambiente di lavoro:

0. Requisiti iniziali:
    - Python 3.12.3;
    - [Swi prolog](https://www.swi-prolog.org) 10.0.0-1;

1. Clonare il repository eseguendo il seguente comando su terminale:  
    ```
    git clone https://github.com/Ingegneria-del-Software-xddd/Icon-2526-diamonds
    ```
    ```
    cd test-icon
    ```

2. Creare e attivare un nuovo ambiente virtuale
    ```py
    py -3.11 -m venv venv
    ```
    ```
    venv\Scripts\activate
    ```

3. Installare dipendenze necessarie:
    ```py
    pip install pandas numpy scikit-learn matplotlib seaborn scipy pyswip joblib rdflib
    ```

4. Avviare il programma
    ```py
    cd code/
    ```
    ```py
    python KB/ui_rdf.py
    ```

***

## Esecuzione del progetto

All’avvio, il sistema presenta un menù principale testuale che permette di guidare l’utente in base alle funzionalità disponibili:

<img align="center" src="docs/Screenshot_menu_principale.png" width=430>

Le operazioni possibili sono:

- Testare la previsione del Machine Learning sui diamanti inserendo o generando dati e ottenendo stime di prezzo con probabilità e livello di confidenza. 

- Esplorare e gestire soglie di valutazione tramite Knowledge Base, applicando regole esperte sulla qualità dei diamanti.

- Esportare della conoscenza in formato RDF/Turtle, con supporto a query SPARQL e generazione di report semantici. 

- Riaddestrare il modello AI;

- Analizzare i dati in modo esplorativo;

- Verificare le prestazioni del sistema di apprendimento.

L’esecuzione termina selezionando l’opzione di uscita dal menu.

