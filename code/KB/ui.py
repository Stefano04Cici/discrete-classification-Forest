from preprocessing import CategoricalDataFrame
from prediction import predict_diamond
from random_diamond import random_diamond
import json
from threshold_system import ExtendedKB, MiniKB, BeautyLevel, Threshold
import os
import pandas as pd

from rdf_exporter import ( 
    kb_to_rdf,
    save_kb_to_rdf,
    load_kb_from_rdf,
    generate_diamond_rdf_report,
    query_rdf_kb,
    SPARQL_QUERIES,
    export_kb_with_ml_integration
)
import os
import pandas as pd

last_tested_diamond = None





def prevision_menu():
    
    global last_tested_diamond
    
    while True:

        print("\n" + "="*60)
        print("MENU PREVISIONI - TEST DEL MODELLO AI".center(60))
        print("="*60)
        print("\nCosa vuoi fare?")
        print("1) Inserire MANUALMENTE le caratteristiche di un diamante")
        print("2) Generare un diamante CASUALE per il test")
        print("3) Caricare un diamante da file JSON")
        print("\n'salva' - Salva l'ultimo diamante testato")
        print("'esc'   - Torna al menu principale")
        print("\n" + "-"*60)
        
        choice = input(">>\t").strip().lower()
        
        if choice == "1":  
            print("\n" + "="*60)
            print("INSERIMENTO MANUALE DIAMANTE".center(60))
            print("="*60)
            
            diamond = {}
            
            features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
            
            for feature in features:
                while True:
                    print(f"\nCaratteristica: {feature}")
                    
                    if feature == 'carat':
                        print("   Valori possibili: low, medium, high")
                    elif feature == 'cut':
                        print("   Valori possibili: fair, good, very_good, premium, ideal")
                    elif feature == 'color':
                        print("   Valori possibili: d, e, f, g, h, i, j (d=migliore, j=peggiore)")
                    elif feature == 'clarity':
                        print("   Valori possibili: i1, si2, si1, vs2, vs1, vvs2, vvs1, if (if=migliore)")
                    elif feature in ['depth', 'table', 'x', 'y', 'z']:
                        print("   Valori possibili: low, medium, high")
                    
                    value = input(f"   Inserisci valore per {feature}: ").strip().lower()
                    
                    if value:  
                        diamond[feature] = value
                        break
                    else:
                        print("   ERRORE: Valore non valido. Riprova.")
            
            print("\n" + "-"*60)
            print("MODALITÀ DI PREDIZIONE")
            print("-"*60)
            print("Scegli come il modello deve decidere:")
            print("1) argmax (default per multiclasse - sceglie la classe con probabilità più alta)")
            print("2) Soglia fissa (specifica un valore tra 0 e 1)")
            
            mode_choice = input("Scelta (1 o 2): ").strip()
            
            if mode_choice == "2":
                while True:
                    try:
                        threshold = float(input("Inserisci soglia (0.0 - 1.0): "))
                        if 0 <= threshold <= 1:
                            thr_mode = "fixed"
                            thr_value = threshold
                            break
                        else:
                            print("ERRORE: La soglia deve essere tra 0 e 1")
                    except ValueError:
                        print("ERRORE: Inserisci un numero valido")
            else:
                thr_mode = "argmax"
                thr_value = None
            
            print("\n" + "="*60)
            print("RISULTATO DELLA PREDIZIONE".center(60))
            print("="*60)
            
            try:
                result = predict_diamond(diamond, thr_mode=thr_mode, thr_value=thr_value)
                
                if isinstance(result[0], str):  
                    predicted_class, probability, threshold_used, mode_used = result
                    print(f"\nCLASSE PREDETTA: {predicted_class}")
                    print(f"PROBABILITÀ: {probability:.2%}")
                    print(f"STRATEGIA: {mode_used}")
                    
                    if predicted_class == "low":
                        print("INTERPRETAZIONE: Diamante economico - buon rapporto qualità/prezzo")
                    elif predicted_class == "medium":
                        print("INTERPRETAZIONE: Diamante di medio valore - equilibrio qualità/prezzo")
                    else:
                        print("INTERPRETAZIONE: Diamante di alto valore - qualità premium")
                        
                else:  
                    predicted_label, probability, threshold_used, mode_used = result
                    class_name = "costoso" if predicted_label == 1 else "economico"
                    print(f"\nCLASSE PREDETTA: {class_name} ({predicted_label})")
                    print(f"PROBABILITÀ: {probability:.2%}")
                    print(f"SOGLIA USATA: {threshold_used:.3f}")
                    print(f"MODALITÀ: {mode_used}")
                
                last_tested_diamond = {
                    'diamond': diamond,
                    'result': result,
                    'mode': mode_used
                }
                
            except Exception as e:
                print(f"\nERRORE durante la predizione: {e}")
                print("Verifica che tutte le caratteristiche siano state inserite correttamente.")
        
        
        elif choice == "2":  
            
            print("\n" + "="*60)
            print("GENERAZIONE DIAMANTE CASUALE".center(60))
            print("="*60)
            
            while True:
                try:
                    num_diamonds = int(input("\nQuanti diamanti casuali vuoi generare? (1-10): "))
                    if 1 <= num_diamonds <= 10:
                        break
                    else:
                        print("ERRORE: Inserisci un numero tra 1 e 10")
                except ValueError:
                    print("ERRORE: Inserisci un numero valido")
            
            print("\n" + "-"*60)
            print("MODALITÀ DI PREDIZIONE")
            print("-"*60)
            print("Scegli come il modello deve decidere:")
            print("1) argmax (default per multiclasse)")
            print("2) Soglia fissa (specifica un valore tra 0 e 1)")
            
            mode_choice = input("Scelta (1 o 2): ").strip()
            
            if mode_choice == "2":
                while True:
                    try:
                        threshold = float(input("Inserisci soglia (0.0 - 1.0): "))
                        if 0 <= threshold <= 1:
                            thr_mode = "fixed"
                            thr_value = threshold
                            break
                        else:
                            print("ERRORE: La soglia deve essere tra 0 e 1")
                    except ValueError:
                        print("ERRORE: Inserisci un numero valido")
            else:
                thr_mode = "argmax"
                thr_value = None
            
            print("\n" + "="*60)
            print("DIAMANTI GENERATI E PREDIZIONI".center(60))
            print("="*60)
            
            for i in range(num_diamonds):
                print(f"\nDIAMANTE #{i+1}")
                print("-"*40)
                
                diamond = random_diamond(f"test_output/diamante_random_{i+1}.json")
                
                print("Caratteristiche:")
                for feature, value in diamond.items():
                    print(f"  {feature}: {value}")
                
                try:
                    result = predict_diamond(diamond, thr_mode=thr_mode, thr_value=thr_value)
                    
                    if isinstance(result[0], str):  
                        predicted_class, probability, _, mode_used = result
                        print(f"\n  Prezzo predetto: {predicted_class}")
                        print(f"  Probabilità: {probability:.2%}")
                    else:  # Binario
                        predicted_label, probability, threshold_used, mode_used = result
                        class_name = "costoso" if predicted_label == 1 else "economico"
                        print(f"\n  Prezzo predetto: {class_name}")
                        print(f"  Probabilità: {probability:.2%}")
                    
                    last_tested_diamond = {
                        'diamond': diamond,
                        'result': result,
                        'mode': mode_used
                    }
                
                except Exception as e:
                    print(f"\n  ERRORE nella predizione: {e}")
            
            print(f"\nSUCCESSO: Generati e analizzati {num_diamonds} diamanti casuali")
            print("NOTA: I diamanti sono stati salvati come 'diamante_random_X.json'")
        
        
        elif choice == "3":  
            print("\n" + "="*60)
            print("CARICA DIAMANTE DA FILE JSON".center(60))
            print("="*60)
            
            file_path = input("\nInserisci il nome del file JSON: ").strip()
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        diamond = json.load(f)
                    
                    print("\nSUCCESSO: File caricato correttamente!")
                    print("\nContenuto del file:")
                    for feature, value in diamond.items():
                        print(f"  {feature}: {value}")
                    
                    print("\n" + "-"*60)
                    print("MODALITÀ DI PREDIZIONE")
                    thr_mode = "argmax"
                    thr_value = None
                    
                    print("\n" + "="*60)
                    print("RISULTATO DELLA PREDIZIONE".center(60))
                    print("="*60)
                    
                    result = predict_diamond(diamond, thr_mode=thr_mode, thr_value=thr_value)
                    
                    if isinstance(result[0], str):
                        predicted_class, probability, _, _ = result
                        print(f"\nCLASSE PREDETTA: {predicted_class}")
                        print(f"PROBABILITÀ: {probability:.2%}")
                    else:
                        predicted_label, probability, _, _ = result
                        class_name = "costoso" if predicted_label == 1 else "economico"
                        print(f"\nCLASSE PREDETTA: {class_name}")
                        print(f"PROBABILITÀ: {probability:.2%}")
                    
                    last_tested_diamond = {
                        'diamond': diamond,
                        'result': result,
                        'mode': thr_mode
                    }
                    
                except Exception as e:
                    print(f"\nERRORE nel caricamento del file: {e}")
            else:
                print(f"\nERRORE: File non trovato: {file_path}")
        
        
        elif choice == "salva":  
            if last_tested_diamond is not None:
                filename = input("\nNome del file da salvare (senza estensione): ").strip()
                if not filename:
                    filename = "diamante_salvato"
                
                filename = "test_output/" + filename + ".json"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(last_tested_diamond['diamond'], f, indent=4, ensure_ascii=False)
                    
                    print(f"\nSUCCESSO: Diamante salvato in: {filename}")
                    print("NOTA: Puoi ricaricarlo con l'opzione 3 del menu")
                except Exception as e:
                    print(f"\nERRORE nel salvataggio: {e}")
            else:
                print("\nERRORE: Nessun diamante testato da salvare")
        
        
        elif choice == "esc":  
            print("\nTorno al menu principale...")
            break
        
        else:
            print("\nERRORE: Scelta non valida. Riprova.")


def threshold_menu():
    
    
    print("\nCaricamento knowledge base...")
    try:
        kb = ExtendedKB()
        kb.load_from_json()
        print("SUCCESSO: Knowledge base caricata da file")
    except Exception as e:
        print(f"NOTA: Creazione nuova knowledge base con valori default ({e})")
        kb = ExtendedKB()  
        
    while True:
        
        print("\n" + "="*60)
        print("MENU SOGLIE - VALUTAZIONE DIAMANTI".center(60))
        print("="*60)
        print("\nCosa vuoi fare?")
        print("1) Valutare un diamante inserito MANUALMENTE")
        print("2) Valutare un diamante CASUALE")
        print("3) Visualizzare tutte le regole/soglie")
        print("4) Aggiungere una nuova regola/soglia")
        print("5) Cercare regole specifiche")
        print("6) Salvare la knowledge base")
        print("\n'esc' - Torna al menu principale")
        print("\n" + "-"*60)
        
        choice = input(">>\t").strip().lower()
        
        if choice == "1":  
            print("\n" + "="*60)
            print("VALUTAZIONE DIAMANTE MANUALE".center(60))
            print("="*60)
            
            diamond = {}
            
            features_to_ask = ['carat', 'cut', 'color', 'clarity', 'depth', 'table']
            
            for feature in features_to_ask:
                while True:
                    print(f"\nCaratteristica: {feature}")
                    
                    if feature == 'carat':
                        print("   Esempio: low, medium, high")
                    elif feature == 'cut':
                        print("   Esempio: fair, good, very_good, premium, ideal")
                    elif feature == 'color':
                        print("   Esempio: d, e, f, g, h, i, j")
                    elif feature == 'clarity':
                        print("   Esempio: i1, si2, si1, vs2, vs1, vvs2, vvs1, if")
                    elif feature in ['depth', 'table']:
                        print("   Esempio: low, medium, high")
                    
                    value = input(f"   Valore per {feature}: ").strip().lower()
                    
                    if value:
                        diamond[feature] = value
                        break
                    else:
                        print("   ERRORE: Valore non valido")
            
            print("\n" + "="*60)
            print("RISULTATO VALUTAZIONE".center(60))
            print("="*60)
            
            try:
                score = kb.fuzzy_beauty_score(diamond)
                score_percent = score * 100
                
                print(f"\nPUNTEGGIO QUALITÀ: {score:.3f} ({score_percent:.1f}%)")
                print("-"*40)
                
                if score_percent >= 80:
                    print("ECCELLENTE - Diamante di altissima qualità")
                    print("   Tutte le caratteristiche soddisfano o superano le aspettative")
                elif score_percent >= 60:
                    print("BUONO - Diamante di buona qualità")
                    print("   La maggior parte delle caratteristiche è soddisfacente")
                elif score_percent >= 40:
                    print("MEDIO - Diamante accettabile")
                    print("   Alcune caratteristiche potrebbero essere migliorate")
                elif score_percent >= 20:
                    print("BASSO - Diamante di qualità inferiore")
                    print("   Molte caratteristiche non soddisfano gli standard")
                else:
                    print("MOLTO BASSO - Qualità insufficiente")
                    print("   Considera alternative migliori")
                
                print("\n" + "-"*60)
                print("DETTAGLIO PER CARATTERISTICA")
                print("-"*60)
                
                rules = kb.query()
                for _, rule in rules.iterrows():
                    feature = rule['feature']
                    if feature in diamond:
                        value = diamond[feature]
                        threshold = rule['value']
                        operator = rule['operator']
                        description = rule['description']
                        
                        print(f"\n{feature}: {value}")
                        print(f"  Regola: {operator} {threshold}")
                        print(f"  Descrizione: {description}")
            
            except Exception as e:
                print(f"\nERRORE nella valutazione: {e}")
        
        
        elif choice == "2":  
            print("\n" + "="*60)
            print("VALUTAZIONE DIAMANTE CASUALE".center(60))
            print("="*60)
            
           
            diamond = random_diamond("test_output/diamante_valutazione.json")
            
            print("\nDIAMANTE GENERATO:")
            print("-"*40)
            for feature, value in diamond.items():
                print(f"  {feature}: {value}")
            
            print("\n" + "-"*60)
            print("VALUTAZIONE KNOWLEDGE BASE")
            print("-"*60)
            
            try:
                score = kb.fuzzy_beauty_score(diamond)
                score_percent = score * 100
                
                print(f"\nPUNTEGGIO QUALITÀ: {score:.3f} ({score_percent:.1f}%)")
                
                if score_percent >= 80:
                    print("ECCELLENTE - Raro trovare un diamante così!")
                elif score_percent >= 60:
                    print("BUONO - Buon acquisto")
                elif score_percent >= 40:
                    print("MEDIO - Prezzo dovrebbe essere contenuto")
                elif score_percent >= 20:
                    print("BASSO - Valuta alternative")
                else:
                    print("MOLTO BASSO - Sconsigliato")
                    
            except Exception as e:
                print(f"\nERRORE nella valutazione: {e}")
        
        elif choice == "3":  
            
            print("\n" + "="*60)
            print("REGOLE DELLA KNOWLEDGE BASE".center(60))
            print("="*60)
            
            rules = kb.query()
            
            if len(rules) > 0:
                print(f"\nTrovate {len(rules)} regole:")
                print("-"*60)
                
                for idx, (_, rule) in enumerate(rules.iterrows(), 1):
                    print(f"\n{idx}. {rule['feature']}")
                    print(f"   Operatore: {rule['operator']}")
                    print(f"   Valore: {rule['value']}")
                    print(f"   Livello: {rule['level']}")
                    print(f"   Descrizione: {rule['description']}")
            else:
                print("\nNOTA: Nessuna regola trovata nella knowledge base")
        
        elif choice == "4":  
            
            print("\n" + "="*60)
            print("AGGIUNGI NUOVA REGOLA".center(60))
            print("="*60)
            
            print("\nScegli il tipo di regola:")
            print("1) Regola semplice (soglia per una caratteristica)")
            print("2) Regola composta (combinazione di più caratteristiche)")
            
            rule_type = input("\nScelta (1 o 2): ").strip()
            
            if rule_type == "1":
                print("\n" + "-"*60)
                print("NUOVA REGOLA SEMPLICE")
                print("-"*60)
                
                feature = input("\nCaratteristica (es: carat, cut, color): ").strip().lower()
                operator = input("Operatore (es: <=, >=, ==): ").strip()
                value = input("Valore (es: medium, ideal, h): ").strip().lower()
                
                print("\nLivello di apprezzamento:")
                print("1) LOW (basso)")
                print("2) MEDIUM (medio)")
                print("3) HIGH (alto)")
                
                level_choice = input("Scelta (1-3): ").strip()
                if level_choice == "1":
                    level = BeautyLevel.LOW
                elif level_choice == "2":
                    level = BeautyLevel.MEDIUM
                elif level_choice == "3":
                    level = BeautyLevel.HIGH
                else:
                    print("NOTA: Impostato livello MEDIUM di default")
                    level = BeautyLevel.MEDIUM
                
                description = input("\nDescrizione (spiega la regola): ").strip()
                
                try:
                    threshold = Threshold(
                        feature=feature,
                        operator=operator,
                        value=value,
                        level=level,
                        description=description
                    )
                    
                    kb.insert_threshold(threshold)
                    print(f"\nSUCCESSO: Regola aggiunta per {feature}")
                except Exception as e:
                    print(f"\nERRORE nella creazione della regola: {e}")
            
            elif rule_type == "2":
                print("\n" + "-"*60)
                print("NUOVA REGOLA COMPOSITA")
                print("-"*60)
                
                name = input("\nNome della regola (es: 'DiamantePerfetto'): ").strip()
                
                conditions = []
                print("\nAggiungi condizioni (lascia vuoto il nome per terminare):")
                
                while True:
                    feature = input("\nCaratteristica (lascia vuoto per finire): ").strip().lower()
                    if not feature:
                        break
                    
                    operator = input(f"Operatore per {feature} (es: ==, >=): ").strip()
                    value = input(f"Valore per {feature}: ").strip().lower()
                    
                    conditions.append((feature, operator, value))
                    print(f"SUCCESSO: Condizione aggiunta: {feature} {operator} {value}")
                
                if conditions:
                    print("\nLivello di apprezzamento:")
                    print("1) LOW (basso)")
                    print("2) MEDIUM (medio)")
                    print("3) HIGH (alto)")
                    
                    level_choice = input("Scelta (1-3): ").strip()
                    if level_choice == "1":
                        level = BeautyLevel.LOW
                    elif level_choice == "2":
                        level = BeautyLevel.MEDIUM
                    elif level_choice == "3":
                        level = BeautyLevel.HIGH
                    else:
                        print("NOTA: Impostato livello MEDIUM di default")
                        level = BeautyLevel.MEDIUM
                    
                    try:
                        kb.add_composite_rule(name, conditions, level)
                        print(f"\nSUCCESSO: Regola composita '{name}' aggiunta con {len(conditions)} condizioni")
                    except Exception as e:
                        print(f"\nERRORE nell'aggiunta della regola: {e}")
                else:
                    print("\nERRORE: Nessuna condizione aggiunta")
        
        elif choice == "5":  
            
            print("\n" + "="*60)
            print("CERCA REGOLE".center(60))
            print("="*60)
            
            print("\nCerca per:")
            print("1) Caratteristica")
            print("2) Livello di apprezzamento")
            print("3) Testo nella descrizione")
            
            search_type = input("\nScelta (1-3): ").strip()
            
            if search_type == "1":
                feature = input("\nNome caratteristica (es: cut, color): ").strip().lower()
                results = kb.query(feature=feature)
            elif search_type == "2":
                print("\nLivello:")
                print("1) LOW")
                print("2) MEDIUM")
                print("3) HIGH")
                level_choice = input("Scelta (1-3): ").strip()
                if level_choice == "1":
                    level = BeautyLevel.LOW
                elif level_choice == "2":
                    level = BeautyLevel.MEDIUM
                elif level_choice == "3":
                    level = BeautyLevel.HIGH
                else:
                    print("NOTA: Cerca a livello MEDIUM")
                    level = BeautyLevel.MEDIUM
                results = kb.query(level=level)
            elif search_type == "3":
                text = input("\nTesto da cercare nella descrizione: ").strip()
                results = kb.query(description_like=text)
            else:
                results = pd.DataFrame()
            
            if len(results) > 0:
                print(f"\nTrovate {len(results)} regole:")
                for _, rule in results.iterrows():
                    print(f"\n• {rule['feature']} {rule['operator']} {rule['value']}")
                    print(f"  Livello: {rule['level']}")
                    print(f"  Descrizione: {rule['description']}")
            else:
                print("\nNessuna regola trovata")
        
        elif choice == "6":  
            try:
                kb.save_to_json()
                print("\nSUCCESSO: Knowledge base salvata")
            except Exception as e:
                print(f"\nERRORE nel salvataggio: {e}")
        
        elif choice == "esc":  
            print("\nTorno al menu principale...")
            break
        
        else:
            print("\nERRORE: Scelta non valida. Riprova.")


def rdf_exporter_menu():
    
    print("\nCaricamento knowledge base per esportazione RDF...")
    try:
        kb = ExtendedKB()
        kb.load_from_json()
        print("SUCCESSO: Knowledge base caricata")
    except Exception as e:
        print(f"NOTA: Creazione nuova knowledge base ({e})")
        kb = ExtendedKB()
    
    while True:
        
        print("\n" + "="*60)
        print("MENU ESPORTAZIONE RDF - CONOSCENZA SEMANTICA".center(60))
        print("="*60)
        print("\nCosa vuoi fare?")
        print("1) Esportare la Knowledge Base in formato RDF/Turtle")
        print("2) Caricare una Knowledge Base da file RDF")
        print("3) Generare report RDF per un diamante specifico")
        print("4) Eseguire query SPARQL sulla KB")
        print("5) Esportare KB integrata con modello ML")
        print("6) Visualizzare statistiche della KB RDF")
        print("\n'esc' - Torna al menu principale")
        print("\n" + "-"*60)
        
        choice = input(">>\t").strip().lower()
        
        if choice == "1":  
            
            print("\n" + "="*60)
            print("ESPORTAZIONE KNOWLEDGE BASE IN RDF".center(60))
            print("="*60)
            
            default_name = "diamonds_kb.ttl"
            filename = input(f"\nNome file di output [{default_name}]: ").strip()
            if not filename:
                filename = default_name
            
            if not filename.endswith('.ttl'):
                filename += '.ttl'
            
            output_path = os.path.join("test_output", filename)
            
            try:
                result_path = save_kb_to_rdf(kb, output_path)
                
                print(f"\nSUCCESSO: Knowledge Base esportata!")
                print(f"File generato: {result_path}")
                
                from rdflib import Graph
                g = Graph()
                g.parse(result_path, format="turtle")
                print(f"Numero di triple RDF: {len(g)}")
                
                print("\nNamespace utilizzati:")
                for prefix, namespace in list(g.namespaces())[:5]:
                    print(f"  {prefix}: {namespace}")
                
                print("\nAnteprima del file:")
                with open(result_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:20]
                    for line in lines:
                        print(f"  {line.rstrip()}")
                
                if len(lines) >= 20:
                    print("  ... [file più lungo]")
                    
            except Exception as e:
                print(f"\nERRORE durante l'esportazione: {e}")
                import traceback
                traceback.print_exc()
        
        
        elif choice == "2":  
            print("\n" + "="*60)
            print("CARICA KNOWLEDGE BASE DA RDF".center(60))
            print("="*60)
            
            print("\nFile disponibili in test_output/:")
            try:
                files = [f for f in os.listdir("test_output") if f.endswith('.ttl')]
                for i, f in enumerate(files, 1):
                    print(f"  {i}) {f}")
            except:
                files = []
            
            if files:
                file_choice = input("\nNumero del file o percorso completo: ").strip()
                
                try:
                    if file_choice.isdigit():
                        idx = int(file_choice) - 1
                        if 0 <= idx < len(files):
                            rdf_path = os.path.join("test_output", files[idx])
                        else:
                            print("Numero non valido")
                            continue
                    else:
                        rdf_path = file_choice
                    
                    print(f"\nCaricamento da: {rdf_path}")
                    loaded_kb = load_kb_from_rdf(rdf_path)
                    
                    print("\nSUCCESSO: Knowledge Base caricata da RDF!")
                    print(f"Soglie caricate: {len(loaded_kb._store)}")
                    print(f"Regole composite: {len(loaded_kb.composite_rules)}")
                    
                    kb = loaded_kb
                    
                    if len(kb._store) > 0:
                        print("\nPrime 3 regole caricate:")
                        for i, (pos, thr) in enumerate(list(kb._store.items())[:3], 1):
                            print(f"  {i}. {thr.feature} {thr.operator} {thr.value}")
                    
                except Exception as e:
                    print(f"\nERRORE nel caricamento: {e}")
            else:
                print("\nNessun file RDF trovato nella cartella test_output/")
        
        
        elif choice == "3":  
            print("\n" + "="*60)
            print("REPORT RDF PER DIAMANTE".center(60))
            print("="*60)
            
            print("\nScegli come ottenere il diamante:")
            print("1) Inserire manualmente")
            print("2) Generare casualmente")
            print("3) Usare ultimo diamante testato")
            
            diamond_choice = input("\nScelta (1-3): ").strip()
            diamond = None
            
            if diamond_choice == "1":
                diamond = {}
                features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
                
                print("\nInserisci le caratteristiche:")
                for feature in features:
                    value = input(f"{feature}: ").strip().lower()
                    if value:
                        diamond[feature] = value
                    else:
                        diamond[feature] = "medium"  
            elif diamond_choice == "2":
                diamond = random_diamond("test_output/diamond_rdf_temp.json")
                print("\nDiamante generato casualmente")
                
            elif diamond_choice == "3":
                if last_tested_diamond:
                    diamond = last_tested_diamond['diamond']
                    print("\nUsando ultimo diamante testato")
                else:
                    print("\nNessun diamante testato disponibile")
                    continue
            else:
                print("Scelta non valida")
                continue
            
            if diamond:
                filename = input("\nNome file report [diamond_report.ttl]: ").strip()
                if not filename:
                    filename = "diamond_report.ttl"
                
                if not filename.endswith('.ttl'):
                    filename += '.ttl'
                
                output_path = os.path.join("test_output", filename)
                
                try:
                    result_path = generate_diamond_rdf_report(diamond, kb, output_path)
                    
                    print(f"\nSUCCESSO: Report RDF generato!")
                    print(f"File: {result_path}")
                    
                    fuzzy_score = kb.fuzzy_beauty_score(diamond)
                    print(f"Fuzzy score del diamante: {fuzzy_score:.3f}")
                    
                    print("\nAnteprima report:")
                    with open(result_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:15]
                        for line in lines:
                            print(f"  {line.rstrip()}")
                            
                except Exception as e:
                    print(f"\nERRORE nella generazione report: {e}")
        
        
        elif choice == "4":  
            print("\n" + "="*60)
            print("QUERY SPARQL SULLA KNOWLEDGE BASE".center(60))
            print("="*60)
            
            temp_file = "test_output/kb_temp_query.ttl"
            save_kb_to_rdf(kb, temp_file)
            
            print("\nQuery predefinite disponibili:")
            for i, (name, query) in enumerate(SPARQL_QUERIES.items(), 1):
                print(f"  {i}) {name}")
            
            print("  c) Query personalizzata")
            
            query_choice = input("\nScelta: ").strip().lower()
            sparql_query = ""
            
            if query_choice == "c":
                print("\nInserisci la tua query SPARQL (termina con linea vuota):")
                lines = []
                while True:
                    line = input("SPARQL> ")
                    if not line:
                        break
                    lines.append(line)
                sparql_query = "\n".join(lines)
            elif query_choice.isdigit():
                idx = int(query_choice) - 1
                query_names = list(SPARQL_QUERIES.keys())
                if 0 <= idx < len(query_names):
                    query_name = query_names[idx]
                    sparql_query = SPARQL_QUERIES[query_name]
                    print(f"\nQuery: {query_name}")
                else:
                    print("Numero non valido")
                    continue
            else:
                print("Scelta non valida")
                continue
            
            if sparql_query:
                try:
                    print("\nEsecuzione query...")
                    results = query_rdf_kb(temp_file, sparql_query)
                    
                    print(f"\nRISULTATI: {len(results)} righe trovate")
                    print("-"*60)
                    
                    if results:
                        for i, row in enumerate(results[:5], 1):
                            print(f"\nRiga {i}:")
                            for key, value in row.items():
                                print(f"  {key}: {value}")
                        
                        if len(results) > 5:
                            print(f"\n... e altre {len(results) - 5} righe")
                            
                        save_res = input("\nSalvare i risultati in JSON? (s/n): ").strip().lower()
                        if save_res == 's':
                            json_file = input("Nome file [query_results.json]: ").strip()
                            if not json_file:
                                json_file = "query_results.json"
                            
                            output_path = os.path.join("test_output", json_file)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)
                            print(f"Risultati salvati in: {output_path}")
                            
                    else:
                        print("Nessun risultato trovato")
                        
                except Exception as e:
                    print(f"\nERRORE nell'esecuzione query: {e}")
            
            try:
                os.remove(temp_file)
            except:
                pass
        
        
        elif choice == "5":  
            print("\n" + "="*60)
            print("ESPORTAZIONE INTEGRATA ML + KB".center(60))
            print("="*60)
            
            model_info = {}
            
            print("\nInserisci informazioni del modello ML:")
            model_info['name'] = input("Nome modello [RandomForest]: ").strip() or "RandomForest"
            model_info['description'] = input("Descrizione: ").strip() or "Modello Random Forest per classificazione diamanti"
            
            try:
                from prediction import load_payload
                payload = load_payload()
                if 'features' in payload:
                    model_info['features'] = payload['features']
                    print(f"Features caricate automaticamente: {len(model_info['features'])}")
                else:
                    # Chiedi manualmente
                    features_str = input("Features (separate da virgola): ").strip()
                    model_info['features'] = [f.strip() for f in features_str.split(',')] if features_str else []
            except:
                features_str = input("Features (separate da virgola): ").strip()
                model_info['features'] = [f.strip() for f in features_str.split(',')] if features_str else []
            
            try:
                from preprocessing import CategoricalDataFrame
                df = CategoricalDataFrame()
                metrics = df.evaluate_model_performance(plot_confusion_matrix=False)
                model_info['accuracy'] = metrics.get('accuracy', 0.0)
                print(f"Accuracy rilevata: {model_info['accuracy']:.3f}")
            except:
                acc_input = input("Accuracy modello (0.0-1.0) [0.85]: ").strip()
                model_info['accuracy'] = float(acc_input) if acc_input else 0.85
            
            base_name = input("\nBase nome file [diamonds_ai_system]: ").strip() or "diamonds_ai_system"
            
            try:
                files = export_kb_with_ml_integration(kb, model_info, base_name)
                
                print("\nSUCCESSO: Sistema integrato esportato!")
                for key, path in files.items():
                    print(f"  {key}: {path}")
                    
            except Exception as e:
                print(f"\nERRORE nell'esportazione integrata: {e}")
        
        
        elif choice == "6":  
            print("\n" + "="*60)
            print("STATISTICHE KNOWLEDGE BASE RDF".center(60))
            print("="*60)
            
            temp_file = "test_output/kb_stats_temp.ttl"
            save_kb_to_rdf(kb, temp_file)
            
            try:
                from rdflib import Graph
                g = Graph()
                g.parse(temp_file, format="turtle")
                
                print(f"\nStatistiche della Knowledge Base:")
                print(f"Triple totali: {len(g)}")
                print(f"Namespace definiti: {len(list(g.namespaces()))}")
                print(f"Soglie nella KB: {len(kb._store)}")
                print(f"Regole composite: {len(kb.composite_rules)}")
                
                # Conta tipi di triple
                subject_counts = {}
                for s, p, o in g:
                    pred_name = str(p).split('#')[-1] if '#' in str(p) else str(p)
                    subject_counts[pred_name] = subject_counts.get(pred_name, 0) + 1
                
                print("\nTriple per predicato (top 10):")
                sorted_preds = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                for pred, count in sorted_preds:
                    print(f"  {pred}: {count}")
                
                query = """
                PREFIX ex: <http://example.org/diamonds#>
                SELECT (COUNT(?feature) as ?count)
                WHERE {
                    ?feature a ex:DiamondFeature .
                }
                """
                results = query_rdf_kb(temp_file, query)
                if results and 'count' in results[0]:
                    print(f"\nFeatures definite in RDF: {results[0]['count']}")
                
            except Exception as e:
                print(f"\nERRORE nell'analisi: {e}")
            
            try:
                os.remove(temp_file)
            except:
                pass
        
        
        elif choice == "esc":  
            print("\nTorno al menu principale...")
            break
        
        else:
            print("\nERRORE: Scelta non valida. Riprova.")


def ui():
    
    print("\n" + "="*70)
    print("BENVENUTO NEL SISTEMA DI INTELLIGENZA ARTIFICIALE".center(70))
    print("PREDIZIONE E VALUTAZIONE DIAMANTI".center(70))
    print("="*70)
    
    print("\nQuesto sistema permette di:")
    print("   1) Prevedere il prezzo di un diamante usando AI")
    print("   2) Valutare la qualità di un diamante con regole esperte")
    print("   3) Esportare conoscenza in formato semantico (RDF)")
    
    print("\n" + "-"*70)
    print("INIZIALIZZAZIONE DEL MODELLO DI APPRENDIMENTO".center(70))
    print("-"*70)
    print("\nSto caricando e preparando i dati dei diamanti...")
    
    df = CategoricalDataFrame()
    
    print("\nSUCCESSO: DATI CARICATI CORRETTAMENTE!")
    print(f"   Diamanti nel dataset: {len(df)}")
    print(f"   Colonne disponibili: {', '.join(df.columns)}")
    
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPALE".center(60))
        print("="*60)
        print("\nCosa vuoi fare?")
        print("1) TESTARE LA PREVISIONE AI")
        print("   • Inserisci o genera diamanti")
        print("   • Ottieni previsioni di prezzo (low/medium/high)")
        print("   • Vedi le probabilità e la confidenza")
        
        print("\n2) ESPLORARE SOGLIE DI VALUTAZIONE")
        print("   • Valuta la qualità dei diamanti")
        print("   • Gestisci regole di valutazione")
        print("   • Aggiungi nuove regole esperte")
        
        print("\n3) ESPORTAZIONE RDF - CONOSCENZA SEMANTICA")
        print("   • Esporta regole in formato RDF/Turtle")
        print("   • Esegui query SPARQL sulla knowledge base")
        print("   • Genera report semantici per diamanti")
        
        print("\n4) ADDESTRARE IL MODELLO AI")
        print("   • Rigenera il modello con i dati attuali")
        print("   • Ottieni nuove metriche di performance")
        
        print("\n5) ANALISI ESPLORATIVA DEI DATI")
        
        print("\n6) VERIFICA PRESTAZIONI DEL SISTEMA DI APPRENDIMENTO")
        
        print("\n7) ESCI")
        print("\n" + "-"*60)
        
        choice = input("\nSeleziona un'opzione (1-7): ").strip()
        
        if choice == "1":
            prevision_menu()
        elif choice == "2":
            threshold_menu()
        elif choice == "3":
            rdf_exporter_menu()
        elif choice == "4":
            print("\n" + "="*60)
            print("ADDESTRAMENTO MODELLO AI".center(60))
            print("="*60)
            print("\nATTENZIONE: questa operazione potrebbe richiedere alcuni minuti")
            confirm = input("\nProcedere con l'addestramento? (s/n): ").strip().lower()
            if confirm == 's':
                print("\nSto addestrando il modello...")
                df.train_model()
                print("\nSUCCESSO: Modello addestrato e salvato!")
            else:
                print("\nAddestramento annullato")
        elif choice == "5":
            df.eda()
            input("\nPremi Invio per continuare...")
        elif choice == "6":
            df.plot_learning_curve_single_run()
            df.evaluate_model_performance()
            input("\nPremi Invio per continuare...")
        elif choice == "7":
            print("\n" + "="*60)
            print("GRAZIE PER AVER USATO IL SISTEMA!".center(60))
            print("="*60)
            break
        else:
            print("\nERRORE: Scelta non valida. Inserisci un numero da 1 a 7.")


if __name__ == "__main__":
    ui()