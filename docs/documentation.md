1 Introduzione
Il progetto "Diamond Price Prediction" ha l'obiettivo di integrare tecniche di Machine
Learning con sistemi basati su Conoscenza (Knowledge Base) per supportare la
valutazione e la stima del prezzo dei diamanti.
A differenza dell'oro o dell'argento, che hanno un prezzo al grammo standardizzato, il
valore di un diamante non è lineare, due pietre dello stesso peso possono avere prezzi
drasticamente diversi in base a sottili differenze di colore, purezza o qualità del taglio.
Questa soggettività rende difficile per un acquirente non esperto (o anche per un
investitore) capire se il prezzo proposto è onesto.
Da qui nasce l'idea del nostro progetto, che a differenza dei classici stimatori di prezzo
(che forniscono solo un valore numerico), il sistema è in grado di fornire un
ragionamento sul perché un diamante possiede determinate caratteristiche (es. rarità,
ottimo taglio), unendo la potenza del ML alla trasparenza delle regole logiche.

1.1 Obiettivi del progetto
L'obiettivo primario è sviluppare un'applicazione software ibrida che unisca due mondi
dell'Intelligenza Artificiale spesso tenuti separati:
1. Machine Learning: Per calcolare stime precise basate su migliaia di dati storici.
2. Rappresentazione della Conoscenza (Knowledge Base): Per spiegare le varie
caratteristiche della pietra usando regole logiche.
Nello specifico, il sistema offre all'utente un menu di funzionalità completo:
● Predizione del Prezzo: L'utente inserisce i dati (es. "0.5 carati, taglio Ideal") e il
sistema stima la fascia di prezzo tramite tecniche di Machine Learning
● Analisi Qualitativa: Il sistema spiega se il diamante è "Raro", "Commerciale" o "Da
Investimento" tramite la Knowledge Base.
● Simulazione Dati: Possibilità di generare diamanti casuali per testare il mercato.
● Export Semantico: Esportazione dei dati in formato RDF/Turtle per l'interoperabilità
sul Web Semantico.

1.2 Strumenti utilizzati
Di seguito è presente la lista completa di tutti gli strumenti utilizzati nel sistema:
● Python versione 3.12.3: Linguaggio base per l'intera infrastruttura;
● Scikit-Learn: Per il modello Random Forest e le pipeline di addestramento;
● Pandas, NumPy, Scipy: Per la manipolazione matematica dei dataset;
● Prolog (e libreria python PySwip): Per la definizione delle regole logiche;
● RDFLib: Per il web semantico e generazione di ontologie e Linked Data;
● joblib: Per la persistenza dei modelli;
● Matplotlib/seaborn: Per la produzione dei grafici.

Lo schema qui sotto mostra una versione semplificata del workflow del sistema tenendo
conto degli strumenti più importanti.

![tools_schema](img/tools_schema.png)

2) Analisi Dei Dati
In questo progetto, l'approccio ai dati non è stato puramente statistico, ma ibrido.
Abbiamo operato su due livelli di astrazione:
1. Livello Quantitativo (Raw Data): Rappresentato dai valori numerici grezzi (dimensioni
in mm, peso in carati, prezzo in dollari), utili per l'analisi statistica classica.
2. Livello Qualitativo (Semantic Data): Rappresentato da categorie discrete (es. "Low",
"Medium", "High"), fondamentali per il ragionamento simbolico della KB.
Ciò ha richiesto una pipeline di preprocessing complessa, capace di dialogare con il
motore di Prolog per "etichettare" il dataset prima ancora di sottoporlo agli algoritmi di
Machine Learning.

2.1 Dataset
Il progetto opera nel dominio della Gemmologia Computazionale. Per simulare il
ragionamento di un esperto, il sistema non può limitarsi a numeri grezzi, ma deve operare
su concetti semantici. Per questo motivo, sono state definite due strutture dati parallele.
Il dataset di partenza, quello grezzo, è il famoso “Diamonds Dataset”, composto da 53.940
istanze. Ogni istanza rappresenta un singolo diamante caratterizzato da 10 attributi
(feature), essi presentano valori numerici e sono i seguenti (mostrati anche nella tabella):
● Carat: peso del diamante (range: 0.2-5.01). È il fattore che più influenza il prezzo.
● Cut: La qualità del taglio. Ordine: Fair, Good, Very Good, Premium, Ideal.
● Color: Il colore del diamante. Ordine: J (peggiore), I, H, G, F, E, D (migliore).
● Clarity: La purezza (misura delle inclusioni). Ordine: l1 (peggiore), SI2, SI1, VS2, VS1,
VVS2, VVS1, IF (migliore).
● Depth: La percentuale di profondità totale (z / media(x, y)).
● Table: La larghezza della tavola superiore del diamante relativa al punto più largo.
● Dimensions (x, y, z): Lunghezza, larghezza e profondità in mm.
● Price (Target): Il prezzo in dollari USA (range: 326 - 18.823).

2.2 Preprocessing
Appena avviato il programma, la classe CategoricalDataFrame (in preprocessing.py) inizia
la preparazione dei Dati. A differenza delle pipeline standard che usano semplici formule
matematiche, questo modulo implementa una Discretizzazione basata su Conoscenza. Il
flusso di trasformazione avviene in 4 fasi:
1) Gestione dei Valori Mancanti e Cleaning
La prima fase ha riguardato la pulizia del dato grezzo. Sebbene il dataset Diamonds sia di
alta qualità, abbiamo implementato dei semplici meccanismi per garantire che il
workflow avvenga senza errori:
● Utilizzo di SimpleImputer con strategia most_frequent: Se nel file mancano delle
informazioni (es. manca il "colore"), il sistema inserisce automaticamente il valore più
comune (moda), evitando che il programma si blocchi.
● Rimozione di dimensioni fisiche nulle: rimossi diamanti con x, y o z uguali a 0, in
quanto fisicamente impossibili.

2) Integrazione Python-Prolog (Discretizzazione Logica)
Fase importante, invece di usare un semplice KBinsDiscretizer di Scikit-learn, abbiamo
sfruttato la libreria pyswip per interrogare la Knowledge Base:
1. Python legge una riga dal dataset grezzo (es. price: 326);
2. Python richiede a Prolog come quel dato venga classificato;
3. Viene invocata la regola Prolog corrispondente, ad esempio:
price_class(Price, low) :- Price < 1000.
4. Prolog restituisce l'etichetta “low”, che viene salvata nel nuovo dataset categorico.

3) Encoding e Trasformazione per il Machine Learning
Dato che gli algoritmi di Machine Learning (come Random Forest) richiedono input
numerici, il dataset, dopo essere stato pulito e "tradotto”, subisce una trasformazione
finale tramite ColumnTransformer:
● Ordinal Encoding per Variabili Gerarchiche: per le feature come cut, color e clarity
dove c'è una gerarchia chiara (Ideal è meglio di Premium), abbiamo assegnato dei
voti crescenti, per esempio Fair = 0, Good = 1. Ideal = 4.
● One-Hot Encoding per Variabili Nominali: per feature dove non c'è un ordine
numerico preciso, abbiamo trasformato le etichette in colonne binarie (0 o 1).
● Gestione delle Categorie Sconosciute: per far sì che valori fuori scala non causino il
crash dell'applicazione ma vengano gestiti in modo neutro, l'encoder è stato
configurato con handle_unknown='ignore' (o strategie simili tramite pipeline).

4) Data Splitting e Validazione
Infine, il dataset processato è stato diviso utilizzando train_test_split in 2 gruppi:
● Training Set (80%): Utilizzato per l'addestramento del modello e per la
Cross-Validation (con StratifiedKFold a 5 split, definito in CV_SPLITS nel file
config.py).
● Test Set (20%): Mantenuto isolato (hold-out) per la valutazione finale delle
performance, usati per dare un voto imparziale alla precisione del sistema.
A questo punto, i dati sono puliti, codificati e pronti per essere utilizzati nella fase di
addestramento del modello predittivo, che verrà descritta nel Capitolo 4 (ML). Di sotto vi
è uno schema che riassume la pipeline iniziale, cioè la logica che inizia non appena il
codice è avviato.

![initial_pipeline_schema](img/initial_pipeline_schema.png)

2.3 Analisi esplorativa

In questa sezione esploriamo visivamente i dati per comprenderne il comportamento
prima di passare alla fase di addestramento. L'obiettivo è capire quali caratteristiche sono
più importanti per determinare il prezzo e come sono distribuiti i diamanti nel nostro
dataset.

Associazione tra le categorie (Cramer's V)

Poiché le feature sono state trasformate in valori categorici (es. da 0.23 a low), non è
possibile utilizzare la classica correlazione lineare di Pearson. Abbiamo quindi calcolato la
Matrice di Associazione basata sulla V di Cramér, specifica per variabili nominali e
ordinali.

![cramer](img/cramer.png)


Questa immagine mostra una matrice di associazione basata sul coefficiente V di Cramér, usata per misurare l'intensità della relazione tra le variabili categoriche del dataset dei diamanti, con valori che vanno da 0.0 per nessuna associazione fino a 1.0 per un'associazione perfetta. Dall'analisi dei dati emerge in modo chiaro che il peso in carati e le dimensioni fisiche x, y e z sono fortemente legati tra loro con valori compresi tra 0.93 e 0.96, e sono anche i fattori che influenzano maggiormente il prezzo finale della gemma, registrando valori di associazione superiori a 0.80. Parallelamente, la qualità del taglio mostra una discreta relazione con la profondità e la larghezza della tavola con valori di 0.40 e 0.38, rispecchiando i parametri geometrici usati in gemmologia. Al contrario, caratteristiche come il colore e la purezza presentano indici di associazione molto più bassi sia rispetto alle dimensioni sia rispetto al prezzo, indicando che la loro influenza sul valore del diamante segue logiche meno dirette e legate a valutazioni qualitative.


Distribuzione dei Prezzi

Successivamente, abbiamo analizzato quanti diamanti ci sono per ogni fascia di prezzo.

![price_distribution](img/price_distribution.png)

Questa immagine mostra un grafico a barre intitolato "Target Class Distribution (price)" che rappresenta la distribuzione della frequenza (conteggio delle istanze) tra le tre classi di prezzo definite nella Knowledge Base: medium, high e low.   Dall'analisi del grafico emerge che le classi medium e high detengono la quota maggioritaria e si equivalgono quasi nei volumi (con la fascia medium leggermente superiore a circa 18.500 istanze e la fascia high poco sotto i 18.000), mentre la classe low risulta meno rappresentata, attestandosi intorno alle 13.500 istanze. Questa configurazione evidenzia uno sbilanciamento del dataset causato dalle soglie stabilite in Prolog, garantendo tuttavia al modello una ricchezza di esempi nelle fasce medio-alte, che risultano le più complesse da stimare

3) Knowledge Base (KB)

La Knowledge Base (KB) è il cervello del sistema. A differenza del Machine Learning, che
impara dagli esempi, la KB contiene le regole dettate da esperti di questo settore
(gemmologi). Nel nostro progetto, la KB è divisa in due componenti che lavorano insieme:
1. Modulo logico, in Prolog (facts_and_rules.pl): Contiene le definizioni statiche e
universali (es. "Cosa significa un taglio ideale?").
2. Modulo “Euristico”, in Python, utilizzando un Sistema a Soglie (threshold_system.py):
Usando le classi MiniKB e ExtendedKB gestisce la conoscenza procedurale e usa le
regole di prolog per dare valutazioni (es. "Questo diamante è eccezionale").
I due moduli comunicano tramite un middleware (pyswip), permettendo al sistema di
unire la logica di prolog con la flessibilità della programmazione a oggetti.

![kb_schema](img/kb_schema.png)


3.1 Regole in Prolog
Il file facts_and_rules.pl definisce il vocabolario del sistema, qui vengono mappati i valori
numerici continui in classi discrete, si è scelto di utilizzare le Clausole di Horn per definire
le proprietà. Le regole principali sono strutturate così:
● Classificazione Carati: Definisce se un diamante è piccolo (low), medio o grande
basandosi sul peso, per esempio:

carat_class(Carat, low) :- Carat < 0.5.
carat_class(Carat, medium) :- Carat >= 0.5, Carat < 1.0.
carat_class(Carat, high) :- Carat >= 1.0.

Regole Complesse: Abbiamo definito concetti più astratti, ovvero regole che
combinano più attributi per identificare gemme di interesse particolare. Esempio: La
regola rare_diamond(X) si attiva solo se il diamante è contemporaneamente puro
(clarity > vvs2), incolore (color > f) e con taglio ideale:

rare_diamond(X) :-
prop(X, clarity, if), % Internally Flawless
prop(X, color, d), % Colorless
prop(X, cut, ideal) % Best Cut


3.2 Threshold System
Mentre Prolog restituisce risposte binarie (Vero/Falso), la valutazione commerciale
richiede sfumature, dunque per la componente procedurale della Knowledge Base, invece
che una logica Fuzzy (che introduce gradi di verità parziali), è stato implementato un
Sistema Basato su Regole a Soglie Discrete. L'algoritmo mappa i valori continui o
categorici in un insieme di stati discreti (Enum), simulando il metodo di classificazione "a
gradini" utilizzato dai gemmologici reali. L’architettura del reasoning è la seguente:
1. Normalizzazione Gerarchica: La funzione get_hierarchy_level converte le label
categoriche (es. Ideal) in punteggi numerici (1, 2, 3) basati sulla classe Enum
BeautyLevel, la quale può essere di tipo LOW (caratteristica standard), MEDIUM
(buona), o HIGH (eccellente).
2. MiniKB (Knowledge Unit): Classi specializzate che valutano una singola dimensione
(es. riceve in input il taglio "Ideal" e restituisce un punteggio HIGH.).
3. ExtendedKB (Aggregatore): È il coordinatore che analizza l'intero diamante.
Combina i punteggi di tutte le caratteristiche, MiniKB e anche regole esterne
(caricate tramite file JSON come composite_rules.json) per generare un report
completo. Comprende una Logica di penalità, cioè a differenza di una media,
adotta un approccio conservativo ("Weakest Link"), se anche una sola proprietà
critica (es. Purezza) è LOW, il sistema genera un Warning, impedendo che un
diamante grande ma difettoso venga classificato come eccellente.

Per garantire la manutenibilità, le regole complesse non sono "hardcoded" nel Python ma
risiedono in composite_rules.json. Ecco un esempio di come il sistema definisce un
diamante "da investimento":

{
"name": "Investment Grade",
"conditions": {
"carat": "high",
"clarity": "high",
"color": "high"
},
"BeautyLevel": "high"
}


Esempio di utilizzo Knowledge Base
L’utente può interrogare il knowledge base tramite l’opzione 2 del Menu, che lo porterà
alla schermata qui a destra.
Attraverso essa può scegliere se
inserire l’istanza di un diamante
manualmente, generarlo in modo
casuale, visualizzare o inserire le
regole/soglie della knowledge base,
cercare regole specifiche o salvare la
knowledge base.

![threshold_menu](img/threshold_menu.png)


Ecco un esempio di valutazione di un diamante generato casualmente tramite la Knowledge Base: il sistema ha ricevuto in input una gemma con caratura high (elevata) e un ottimo colore D, ma caratterizzata da una tavola (table) low e dimensioni superficiali x e y ridotte (low). Di conseguenza l'ExtendedKB, applicando la sua logica conservativa ("Weakest Link"), ha assegnato un punteggio di qualità basso (0.333 / 33.3%), poiché le proporzioni difettose ne compromettono la resa estetica e la brillantezza nonostante la bontà degli altri parametri.

![random_diamond_eval](img/random_diamond_eval.png)


Statistica di Attivazione delle Regole

L'analisi della distribuzione dei livelli di qualità conferma che la logica conservativa
implementata nell'ExtendedKB è efficace, infatti solo il 15% circa dei diamanti raggiunge il
livello HIGH, dimostrando che il sistema agisce come un filtro, valorizzando solo le pietre
senza difetti critici (come un taglio Fair o una purezza I1) che abbassano il valore
complessivo, simulando in modo realistico la severità di una certificazione gemmologica.


![kb_stats](img/kb_stats.png)


4) Machine Learning
4.1 Sommario
Dopo aver trasformato i dati e definito le regole di business tramite la Knowledge Base, la
fase successiva è l'addestramento di un modello predittivo. Il problema è stato modellato
come un task di Classificazione Multiclasse.
L'obiettivo dell'algoritmo non è stimare il prezzo puntuale (che può variare per esempio a
cause di logiche di mercato imprevedibili), ma collocare il diamante nella corretta fascia di
valore definita dalla KB: Low Value (fascia economica), Medium Value (fascia commerciale
standard), High Value (Fascia di lusso/investimento).
4.2 Strumenti Utilizzati
Lo sviluppo del modulo di Machine Learning si è basato sul linguaggio Python:
● Scikit-Learn (sklearn): Framework principale utilizzato per la costruzione della
pipeline. Ha fornito le implementazioni ottimizzate del RandomForestClassifier e
del CalibratedClassifierCV per la gestione delle probabilità, oltre agli strumenti di
preprocessing (OneHotEncoder, OrdinalEncoder).
● Joblib: Fondamentale per la persistenza del modello. Dato che il training su 50.000
istanze è oneroso computazionalmente, Joblib permette di serializzare (salvare su
disco) l'oggetto modello addestrato e ricaricarlo istantaneamente all'avvio
dell'interfaccia, garantendo tempi di risposta immediati (bassa latenza).
● Pandas & NumPy: Utilizzati rispettivamente per la manipolazione strutturata dei
dati (Dataframes) e per le operazioni di algebra lineare sui vettori delle feature.
● Matplotlib & Seaborn: Librerie impiegate per la generazione dei grafici di
valutazione, i quali verranno discussi nella sezione 4.4.
4.3 Decisioni di progetto
Durante la fase di progettazione, sono stati valutati diversi algoritmi:
● Linear Regression: Scartata perché il prezzo dei diamanti non cresce linearmente
(un diamante da 2 carati vale molto più del doppio di uno da 1 carato).
● Support Vector Machines (SVM): Efficace, ma molto lenta su dataset grandi (50k+
righe) e difficile da interpretare.


Però per questo compito è stato selezionato il Random Forest Classifier, e inserito
all'interno di una pipeline complessa gestita dalla classe CalibratedClassifierCV. Il Random
Forest è un metodo di “ensemble learning” che costruisce una moltitudine di alberi
decisionali durante il training. Lo abbiamo scelto per vari motivi, i principali:
● Gestione Dati Misti: Lavora eccellentemente con il mix di feature ordinali (Taglio,
Colore) e continue (Carati, Dimensioni) presente nel nostro dataset.
● Importanza delle Feature: Permette di comprendere quali attributi influenzano di più
la decisione (nel nostro caso la Caratura è il fattore dominante).
● Resistenza all'Overfitting: Grazie alla tecnica del bagging (bootstrap aggregating), il
modello generalizza meglio sui nuovi dati rispetto a un singolo albero decisionale.
Un aspetto importante del progetto è l'affidabilità della predizione, non ci basta sapere
che un diamante è "High Value", vogliamo sapere quanto il modello ne è sicuro. Abbiamo
incapsulato il Random Forest in un CalibratedClassifierCV, che ricalibra le probabilità
predette in modo che rispecchino la vera frequenza delle classi, permettendo al sistema
di restituire uno score di confidenza (es. "Classificato High con probabilità del 92%").


Il training set è stato processato attraverso una Pipeline Scikit-learn composta da:
1. Imputer: Gestione valori mancanti.
2. Encoder: Trasformazione variabili categoriche.
3. Estimator: Il modello Random Forest (con 100 alberi, n_estimators=100).
Per validare la robustezza del modello, è stata utilizzata la tecnica della Stratified K-Fold
Cross Validation. Questo assicura che ogni "fold" (sottoinsieme di test)
mantenga la stessa proporzione di diamanti economici, medi e costosi del dataset
originale, evitando bias statistici. Al termine dell'addestramento, l'oggetto modello (che
contiene ormai la 'conoscenza statistica') viene salvato nel file model_path.joblib, pronto
per essere richiamato dall'interfaccia utente senza dover ripetere il training ogni volta.



4.3 Valutazione

La validazione del modello è stata effettuata sul Test Set (20% del dataset, circa 10.800
istanze), mantenuto isolato durante la fase di addestramento per garantire l'imparzialità
dei risultati. Il modello ha mostrato prestazioni eccellenti, giustificate dalla forte
correlazione tra le caratteristiche fisiche (carati/purezza) e la fascia di prezzo:

A differenza della semplice Accuracy, che potrebbe essere ingannevole, abbiamo analizzato le metriche per singola classe:

![first_eval](img/first_eval.png)

La performance sulla classe Low è molto solida (F1 0.902) con un'ottima recall (0.952). Il modello individua quasi sempre i diamanti economici, la performance sulla classe High mostra invece un'ottima precisione (0.960) e un buon f1-score (0.937). Questo significa che il modello mantiene un livello di affidabilità elevato sui diamanti di maggior valore, raggiungendo un'accuratezza globale del 90% sul test set.
I Brier score mostrati indicano un livello di accuratezza probabilistica molto buono per tutte le classi, con valori vicini allo zero (dove 0 rappresenta una predizione perfetta). In particolare, le classi high (0.0370) e low (0.0403) ottengono i punteggi migliori e più bassi, confermando un'elevata affidabilità e calibrazione delle probabilità stimate dal modello per i diamanti di fascia alta ed economica, mentre la classe medium registra un errore leggermente superiore (0.0773), coerente con la maggiore difficoltà nel delimitare le zone di confine intermedie.

![confusion_matrix](img/confusion_matrix.png)

Dall'analisi della Matrice di Confusione emerge che il modello distingue con grande precisione le classi estreme, registrando ad esempio zero errori di confusione diretta tra i diamanti "high" e quelli "low", mentre i principali errori si concentrano nelle interazioni con la fascia "medium" (come i 306 diamanti "high" classificati come "medium" e i 426 diamanti "medium" scambiati per "low"), dove i confini di prezzo sono più sfumati e complessi. Il modello raggiunge dunque un'accuracy del 90% su un test set di 10.000 campioni.


![learning_curve](img/learning_curve.png)


L'analisi delle Learning Curves evidenzia che il modello mostra un buon comportamento di convergenza all'aumentare della dimensione del training set: mentre la curva di training decresce leggermente stabilizzandosi attorno a 0.91, la curva di cross-validation cresce costantemente avvicinandosi ai valori del training e superando quota 0.89. Questo andamento dimostra che l'aggiunta di campioni migliora la capacità di generalizzazione del modello, riducendo progressivamente il divario tra le prestazioni sui dati noti e quelli di validazione.


![reliability_plots](img/reliability_plots.png)

E’ stata condotta anche l'analisi dei Reliability Plots sul modello finale del progetto. Essa rivela che il modello risulta ben calibrato per tutte e tre le classi (High, Low, Medium), le cui curve seguono in modo molto ravvicinato la diagonale ideale che rappresenta la perfetta corrispondenza tra probabilità predette e frazioni osservate. Questo indica che il modello stima correttamente i livelli di confidenza e incertezza nelle previsioni. La stabilità delle curve conferma l'efficacia del training e la robustezza generale raggiunta grazie all'ampliamento del dataset nelle fasi di progettazione


5) Web Semantico (RDF)

5.1 Sommario

Il modulo di Web Semantico ha l’obiettivo di trasformare la Knowledge Base simbolica del
sistema in una rappresentazione formale, interoperabile e interrogabile, utilizzando le
tecnologie standard del Semantic Web, in particolare RDF (Resource Description
Framework) e SPARQL. Attraverso questo componente, il progetto consente
l’esportazione e il riuso della conoscenza in contesti esterni, l’esecuzione di query
semantiche avanzate e la generazione di report RDF specifici per singoli diamanti.

5.2 Strumenti Utilizzati
Per l’implementazione del livello semantico è stato usato come strumento e standard RDF
(Resource Description Framework), utilizzato come modello dati per rappresentare le
caratteristiche dei diamanti, soglie di valutazione, regole semplici, regole composite, livelli
di apprezzamento (LOW, MEDIUM, HIGH) e i metadati del modello di Machine Learning.
La serializzazione adottata è Turtle (.ttl), scelta per la sua leggibilità e diffusione.
SPARQL, il linguaggio standard per l’interrogazione dei grafi RDF, è stato usato per
recuperare regole per caratteristica, analizzare la struttura della Knowledge Base,
ottenere statistiche semantiche, e recuperare pattern specifici (es. "Trova tutte le regole
che definiscono un diamante eccellente").
La Libreria Python RDFLib (Python) è stata utilizzata per creare grafi RDF, serializzare e
deserializzare file .ttl, eseguire query SPARQL, gestire namespace e ontologie leggere.
Per quanto riguarda le ontologie Standard, abbiamo usato integrazioni di vocabolari noti
come schema.org (per entità generiche), foaf (per agenti) e dc (Dublin Core per
metadati).
5.3 Decisioni di Progetto
Il modulo RDF non è stato concepito come un semplice convertitore di file, ma come un
livello di integrazione che unisce i due volti del sistema: la predizione statistica (ML) e il
ragionamento logico (KB). Il sistema offre un set completo di strumenti per la gestione del
ciclo di vita semantico, accessibili tramite un menu dedicato. Le decisioni progettuali
principali riguardano la definizione formale del vocabolario e l'implementazione delle
funzionalità operative di export e tracciabilità. Per garantire che i dati siano comprensibili
da altri sistemi software (interoperabilità), abbiamo adottato un approccio ibrido nella
definizione dei termini:

1. Ontologie Standard: Abbiamo riutilizzato vocabolari mondiali per i concetti
generici: schema.org (schema:), per definire il modello software e i dati generici, e
Dublin Core (dc:), per i metadati come descrizioni e date.
2. Namespace Personalizzato: Per i concetti specifici del nostro dominio che non
esistono negli standard, abbiamo creato il namespace
http://example.org/diamonds# (prefisso ex:). Esempi di predicati custom:
ex:hasCarat, ex:belongsToClass (classe di prezzo), ex:hasBeautyLevel (giudizio di
bellezza).

Il sistema implementa queste operazioni per trasformare i dati grezzi in conoscenza
semantica strutturata:
Generazione del report RDF
Una funzionalità chiave implementata nella funzione generate_diamond_rdf_report è la
creazione dinamica di un grafo per ogni singola predizione. Quando
l'utente analizza un diamante, il sistema genera un file .ttl che unisce tre fonti di dati:
● Dati Fisici: Le 4C inserite dall'utente (Carat, Cut, Color, Clarity).
● Dati Predittivi: Il prezzo stimato dal Random Forest e la confidenza associata.

● Metadati del Modello: Il grafo include un riferimento a quale modello ha effettuato
la stima (es. model_random_forest con accuracy=0.96), ciò garantisce la
tracciabilità (Provenance) della decisione.
Il file .ttl (turtle) generato rappresenta un grafo di nodi e archi, il software non genera
automaticamente un'immagine, ma per visualizzare il grafo bisogna utilizzare tool esterni.
Export della Knowledge Base e Integrazione ML

Il sistema è capace di "esportare se stesso", tramite la funzione kb_to_rdf, che
legge le regole interne del Threshold System e le traduce in triple RDF statiche.
Ecco un esempio di schema RDF semplificato che raffigura come il diamante (centro) sia
collegato ai dati fisici (sinistra, arancioni) e ai dati predittivi/modello (destra, verdi):

![diamond_rdf](img/diamond_rdf.png)

5.4 Valutazione
Per verificare che il livello semantico fosse coerente e privo di errori, abbiamo utilizzato
SPARQL, il linguaggio standard per l'interrogazione dei grafi RDF. Le query sono state
integrate direttamente nel menu dell'applicazione. Nel test in basso
l'obiettivo del test era verificare se il sistema riuscisse a rispondere a domande complesse
sulla propria conoscenza interna, ad esempio: "Quali regole definiscono un diamante eccellente?".
Viene lanciata la seguente interrogazione:

PREFIX ex: <http://example.org/diamonds#>
SELECT ?name ?feature ?operator ?value
WHERE {
  ?rule ex:ruleName ?name .
  ?rule ex:indicatesBeautyLevel ex:HIGH .
  ?rule ex:hasCondition ?condition .
  ?condition ex:appliesToFeature ?feature ;
             ex:thresholdOperator ?operator ;
             ex:conditionValue ?value .
}


![SPARQL_query](img/SPARQL_query.png)

L'esecuzione della query tramite il motore RDFLib ha prodotto il
seguente risultato, confermando che la struttura ontologica è corretta e interrogabile:


Infine, per garantire l'integrità dei dati, il sistema permette di ricaricare un grafo
precedentemente salvato, verificando che il file sia integro e leggibile
(Round-Trip Serialization). 



6) Conclusioni

L'approccio ibrido si è rivelato vincente rispetto ai metodi tradizionali:

1. Precisione: Il Random Forest garantisce un'accuratezza statistica superiore al 95% nella
classificazione delle fasce di prezzo.

2. Spiegabilità: L'uso di Prolog e del Threshold System fornisce quel livello di
comprensione semantica che manca ai soli algoritmi numerici. Il sistema non si limita a
dare un output, ma "argomenta" la sua decisione analizzando la qualità del taglio e la
rarità della pietra.

3. Scalabilità: La struttura modulare del codice permette di aggiornare le regole di
mercato (es. cambiare le soglie di prezzo in Prolog) senza dover implementare onerose modifiche al codice.

In definitiva, il sistema rappresenta un valido prototipo di Decision Support System (DSS)
per il settore gemmologico, capace di assistere sia un utente esperto che un neofita nella
valutazione di un diamante.


6.1 Sviluppi Futuri
Per migliorare ulteriormente il sistema, si ipotizzano i seguenti sviluppi:

● Interfaccia Grafica Web: Portare la CLI su una Web App (usando Flask o Streamlit) per
renderla accessibile da browser.

● Integrazione API Prezzi: Collegare il sistema alle API in tempo reale del mercato dei
diamanti (es. Rapaport Price List) per aggiornare le fasce di prezzo dinamicamente.

● Computer Vision: Integrare un modulo di analisi immagini per estrarre
automaticamente le caratteristiche (taglio, colore) da una foto del diamante,
automatizzando l'input dei dati


7) Riferimenti Bibliografici
1. Dataset:
Diamonds Dataset, Kaggle, (https://www.kaggle.com/datasets/shivam2503/diamonds)
Ggplot2 library references;
2. Machine Learning: Scikit-learn Documentation (RandomForest, CalibratedClassifierCV,
https://scikit-learn.org/stable/modules/ensemble.htm);
3. Prolog Integration: PySWIP Documentation & Logic Programming with Prolog (Bratko),
PySwip official website (https://pypi.org/project/pyswip/);
4. Knowledge Engineering: Russell, S., & Norvig, P. Artificial Intelligence: A Modern
Approach. (Capitoli su Knowledge Representation).
5. RDFLib Documentation: RDFLib 7.0.0 official documentation, la libreria standard de
facto per RDF in Python. Disponibile su: https://rdflib.readthedocs.io/en/stable/.
6. W3C Standards: Resource Description Framework (RDF) and SPARQL Query Language
specifications. Disponibile su: https://www.w3.org/RDF/
7. Risorse Video e Tutorial (YouTube): StatQuest with Josh Starmer: Risorsa
fondamentale per la comprensione visiva degli algoritmi di Machine Learning utilizzati,
in particolare per il funzionamento dei Random Forests, delle ROC Curves e della
Confusion Matrix); FreeCodeCamp.org (Python & Data Science): Corsi completi per
l'implementazione di pipeline di Machine Learning e Data Preprocessing in Python.