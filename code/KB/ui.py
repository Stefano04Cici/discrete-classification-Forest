from preprocessing import CategoricalDataFrame
from prediction import predict_diamond
from random_diamond import random_diamond
import json
from threshold_system import ExtendedKB, MiniKB, BeautyLevel, Threshold
import os
import pandas as pd
import config
from examples_csv_to_prolog import execute_insert_facts

from rdf_exporter import ( 
    kb_to_rdf,
    save_kb_to_rdf,
    load_kb_from_rdf,
    generate_diamond_rdf_report,
    is_diamond_report,
    load_diamond_report_from_rdf,
    query_rdf_kb,
    SPARQL_QUERIES,
    export_kb_rdf,
    describe_rdf
)
import os
import pandas as pd

last_tested_diamond = None


def print_rdf_summary(info):
    
    tipo_map = {
        "diamond": "REPORT DIAMANTE",
        "integrated": "KNOWLEDGE BASE INTEGRATA CON MODELLO ML",
        "kb": "KNOWLEDGE BASE"
    }
    
    print("\n" + "="*60)
    print("RESOCONTO FILE RDF".center(60))
    print("="*60)
    print(f"  Percorso: {info.get('path')}")
    print(f"  Triplette: {info.get('triples')}")
    print(f"  Tipo: {tipo_map.get(info.get('type'), info.get('type'))}")
    
    if info.get('beauty_levels'):
        print("\n== LIVELLI DI BELLEZZA ==")
        for lv in info['beauty_levels']:
            print(f"  * {lv['label']}" + (f" ({lv['appreciation']})" if lv.get('appreciation') else ""))
    
    if info.get('models'):
        for m in info['models']:
            print("\n== MODELLO ML ==")
            print(f"  Nome: {m['name']}")
            if m['title']:
                print(f"  Titolo: {m['title']}")
            if m['description']:
                print(f"  Descrizione: {m['description']}")
            acc = m['accuracy']
            if isinstance(acc, float):
                print(f"  Accuracy: {acc:.3f}")
            elif acc is not None:
                print(f"  Accuracy: {acc}")
            if m['features']:
                print(f"  Feature usate ({len(m['features'])}): {', '.join(m['features'])}")
            if m['kb']:
                print(f"  Collegata a KB: {', '.join(m['kb'])}")
    elif info.get('type') == 'integrated':
        print("\n== MODELLO ML == (nessun modello rilevato)")
    
    kb = info.get('kb')
    if kb:
        print("\n== KNOWLEDGE BASE ==")
        print(f"  Titolo: {kb.get('title')}")
        if kb.get('creator'):
            print(f"  Creatore: {kb['creator']}")
        if kb.get('date'):
            print(f"  Data: {kb['date']}")
        if kb.get('description'):
            print(f"  Descrizione: {kb['description']}")
        if kb.get('num_thresholds') is not None:
            print(f"  Soglie dichiarate: {kb['num_thresholds']}")
        if kb.get('num_composite_rules') is not None:
            print(f"  Regole composite dichiarate: {kb['num_composite_rules']}")
        if kb.get('models'):
            print(f"  Completata da modello/i: {', '.join(kb['models'])}")
    
    if info.get('features'):
        print("\n== CARATTERISTICHE ==")
        for f in info['features']:
            dettagli = []
            if f['types']:
                dettagli.append("tipo " + ", ".join(f['types']))
            if f['category']:
                dettagli.append(f"categoria {f['category']}")
            if f['unit']:
                dettagli.append(f"unità {f['unit']}")
            
            header = f"  - {f['name']}"
            if dettagli:
                header += "  (" + ", ".join(dettagli) + ")"
            print(header)
            
            if f['thresholds']:
                for t in f['thresholds']:
                    level = f" [{t['level']}]" if t['level'] else ""
                    riga = f"      · {t['operator']} {t['value']}{level}"
                    if t['description']:
                        riga += f" — {t['description']}"
                    print(riga)
            else:
                print("      · nessuna soglia definita")
    
    if info.get('rules'):
        print("\n== REGOLE COMPOSITE ==")
        for r in info['rules']:
            print(f"  * {r['name']} -> {r['level']}")
            for c in r['conditions']:
                print(f"      se {c['feature']} {c['operator']} {c['value']}")
    else:
        print("\n== REGOLE COMPOSITE == (nessuna)")
    
    print("\n" + "="*60)


def print_diamond_report(report):
    
    print("\n" + "="*60)
    print("REPORT DIAMANTE".center(60))
    print("="*60)
    print(f"Diamante: {report['label']}")
    print(f"URI: {report['uri']}")
    
    if report['features']:
        print("\nCaratteristiche:")
        for feature, value in report['features'].items():
            print(f"  {feature}: {value}")
    
    fuzzy = report['fuzzy_beauty_score']
    if fuzzy is not None:
        print(f"\nFuzzy score: {fuzzy:.3f}")
    if report['beauty_category']:
        print(f"Categoria bellezza: {report['beauty_category']}")
    
    if report['evaluations']:
        print("\nValutazione soglie:")
        for eval_item in report['evaluations']:
            respected = eval_item['respected']
            status = "OK" if str(respected).lower() == "true" else "NON rispettata"
            print(f"  - {eval_item['feature']}: "
                  f"osservato={eval_item['observed']}, "
                  f"atteso {eval_item['expected_operator']} {eval_item['expected_value']} "
                  f"[{status}]")
    else:
        print("\nNessuna valutazione soglie nel report")


def print_kb_recap(kb):
    
    print(f"Soglie caricate: {len(kb._store)}")
    for i, (pos, thr) in enumerate(kb._store.items(), 1):
        print(f"  {i}. {thr.feature} {thr.operator} {thr.value}")
    
    print(f"Regole composite caricate: {len(kb.composite_rules)}")
    for i, rule in enumerate(kb.composite_rules, 1):
        conds = ", ".join(f"{c[0]} {c[1]} {c[2]}" for c in rule["conditions"])
        print(f"  {i}. {rule['name']} -> {rule['BeautyLevel'].value}"
              f"  (se: {conds})")


def print_rdf_file_stats(rdf_path):
    
    from rdflib import Graph
    
    g = Graph()
    g.parse(rdf_path, format="turtle")
    
    print(f"\nStatistiche della KB RDF ({rdf_path}):")
    print(f"Triple totali: {len(g)}")
    print(f"Namespace definiti: {len(list(g.namespaces()))}")
    
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
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT (COUNT(DISTINCT ?feature) AS ?count)
    WHERE {
        { ?feature a ex:DiamondFeature . }
        UNION
        { ?feature a ?type . ?type rdfs:subClassOf ex:DiamondFeature . }
    }
    """
    results = query_rdf_kb(rdf_path, query)
    if results and 'count' in results[0]:
        print(f"\nFeatures definite in RDF: {results[0]['count']}")




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
            
            print("\n" + "="*60)
            print("RISULTATO DELLA PREDIZIONE".center(60))
            print("="*60)
            
            try:
                result = predict_diamond(diamond, thr_mode="argmax")
                
                if isinstance(result[0], str):  
                    predicted_class, probability, _, _ = result
                    print(f"\nCLASSE PREDETTA: {predicted_class}")
                    print(f"PROBABILITÀ: {probability:.2%}")
                    
                    if predicted_class == "low":
                        print("INTERPRETAZIONE: Diamante economico - buon rapporto qualità/prezzo")
                    elif predicted_class == "medium":
                        print("INTERPRETAZIONE: Diamante di medio valore - equilibrio qualità/prezzo")
                    else:
                        print("INTERPRETAZIONE: Diamante di alto valore - qualità premium")
                        
                else:  
                    predicted_label, probability, _, _ = result
                    class_name = "costoso" if predicted_label == 1 else "economico"
                    print(f"\nCLASSE PREDETTA: {class_name} ({predicted_label})")
                    print(f"PROBABILITÀ: {probability:.2%}")
                
                last_tested_diamond = {
                    'diamond': diamond,
                    'result': result,
                    'mode': "argmax"
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
                    result = predict_diamond(diamond, thr_mode="argmax")
                    
                    if isinstance(result[0], str):  
                        predicted_class, probability, _, _ = result
                        print(f"\n  Prezzo predetto: {predicted_class}")
                        print(f"  Probabilità: {probability:.2%}")
                    else:  # Binario
                        predicted_label, probability, _, _ = result
                        class_name = "costoso" if predicted_label == 1 else "economico"
                        print(f"\n  Prezzo predetto: {class_name}")
                        print(f"  Probabilità: {probability:.2%}")
                    
                    last_tested_diamond = {
                        'diamond': diamond,
                        'result': result,
                        'mode': "argmax"
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
            
            if file_path and not os.path.isabs(file_path) and os.path.dirname(file_path) == "":
                if not file_path.endswith(".json"):
                    file_path += ".json"
                file_path = os.path.join("test_output", file_path)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        diamond = json.load(f)
                    
                    print("\nSUCCESSO: File caricato correttamente!")
                    print("\nContenuto del file:")
                    for feature, value in diamond.items():
                        print(f"  {feature}: {value}")
                    
                    print("\n" + "="*60)
                    print("RISULTATO DELLA PREDIZIONE".center(60))
                    print("="*60)
                    
                    result = predict_diamond(diamond, thr_mode="argmax")
                    
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
                        'mode': "argmax"
                    }
                    
                except Exception as e:
                    print(f"\nERRORE nel caricamento del file: {e}")
            else:
                print(f"\nERRORE: File non trovato: {file_path}")
                if os.path.isdir("test_output"):
                    names = sorted(f for f in os.listdir("test_output") if f.endswith('.json'))
                    if names:
                        print("\nFile .json disponibili in test_output/:")
                        for i, name in enumerate(names, 1):
                            print(f"  {i}) {name}")
        
        
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
                print(f"\nTrovate {len(rules)} regole semplici:")
                print("-"*60)
                
                for idx, (_, rule) in enumerate(rules.iterrows(), 1):
                    print(f"\n{idx}. {rule['feature']}")
                    print(f"   Operatore: {rule['operator']}")
                    print(f"   Valore: {rule['value']}")
                    print(f"   Livello: {rule['level']}")
                    print(f"   Descrizione: {rule['description']}")
            else:
                print("\nNOTA: Nessuna regola semplice trovata nella knowledge base")

            if hasattr(kb, 'composite_rules') and len(kb.composite_rules) > 0:
                print(f"\nTrovate {len(kb.composite_rules)} regole composite:")
                print("-"*60)
                
                for idx, rule in enumerate(kb.composite_rules, 1):
                    conds = " AND ".join(
                        f"{c[0]} {c[1]} {c[2]}" for c in rule["conditions"]
                    )
                    print(f"\n{idx}. {rule['name']}")
                    print(f"   Condizioni: {conds}")
                    print(f"   BeautyLevel: {rule['BeautyLevel'].value}")
            elif hasattr(kb, 'composite_rules'):
                print("\nNOTA: Nessuna regola composta trovata nella knowledge base")
        
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
    
    loaded_rdf_path = None
    loaded_kb_path = None
    loaded_diamond_report = None
    
    while True:
        
        print("\n" + "="*60)
        print("MENU ESPORTAZIONE RDF - CONOSCENZA SEMANTICA".center(60))
        print("="*60)
        print("\nCosa vuoi fare?")
        print("1) Esportare la Knowledge Base in RDF")
        print("2) Caricare una Knowledge Base da file RDF, o un diamante")
        print("3) Generare report RDF per un diamante specifico")
        print("4) Eseguire query SPARQL sulla KB")
        print("5) Visualizzare statistiche della KB RDF, o del diamante")
        print("\n'esc' - Torna al menu principale")
        print("\n" + "-"*60)
        
        choice = input(">>\t").strip().lower()
        
        if choice == "1":  
            
            print("\n" + "="*60)
            print("ESPORTAZIONE KNOWLEDGE BASE RDF".center(60))
            print("="*60)
            
            base_name = input("\nBase nome file [diamonds_ai_system]: ").strip() or "diamonds_ai_system"
            
            print("\nInformazioni della Knowledge Base:")
            kb_metadata = {}
            kb_metadata['title'] = input("Titolo KB [Knowledge Base per Valutazione Diamanti]: ").strip() or "Knowledge Base per Valutazione Diamanti"
            kb_metadata['creator'] = input("Creatore KB [Sistema di Intelligenza Artificiale]: ").strip() or "Sistema di Intelligenza Artificiale"
            kb_metadata['date'] = input("Data KB [2024]: ").strip() or "2024"
            kb_metadata['description'] = input("Descrizione KB [Base di conoscenza per la valutazione della qualità dei diamanti basata su caratteristiche delle 4C]: ").strip() or "Base di conoscenza per la valutazione della qualità dei diamanti basata su caratteristiche delle 4C"
            
            try:
                result_path = export_kb_rdf(kb, base_name, kb_metadata=kb_metadata)
                
                print("\nSUCCESSO: Knowledge Base esportata!")
                print(f"  File: {result_path}")
                    
            except Exception as e:
                print(f"\nERRORE nell'esportazione: {e}")
        
        
        elif choice == "2":  
            print("\n" + "="*60)
            print("CARICA KNOWLEDGE BASE O DIAMANTE DA RDF".center(60))
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
                    
                    if is_diamond_report(rdf_path):
                        print("\nFile riconosciuto: REPORT DIAMANTE")
                        loaded_diamond_report = load_diamond_report_from_rdf(rdf_path)
                        loaded_kb_path = None
                        
                        if not loaded_diamond_report:
                            print("\nNessun diamante trovato nel report")
                        else:
                            print(f"\nSUCCESSO: {len(loaded_diamond_report)} diamante/i caricato/i in memoria.")
                            print("Informazioni disponibili con l'opzione 5.")
                        loaded_rdf_path = rdf_path
                    
                    else:
                        loaded_kb = load_kb_from_rdf(rdf_path)
                        loaded_diamond_report = None
                        
                        loaded_kb_path = rdf_path
                        loaded_rdf_path = rdf_path
                        
                        kb = loaded_kb
                        
                        print("\nSUCCESSO: Knowledge Base caricata in memoria da RDF!")
                        print(f"Soglie caricate: {len(kb._store)}")
                        print(f"Regole composite: {len(kb.composite_rules)}")
                        print("Resoconto e statistiche disponibili con l'opzione 5.")
                        print("KB pronta: query SPARQL (opzione 4) e statistiche (opzione 5) abilitate.")
                
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
                diamond = random_diamond()
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
            
            if not loaded_kb_path:
                print("\nERRORE: Nessuna Knowledge Base caricata in memoria.")
                print("Usa prima l'opzione 2 per caricare una KB in memoria.")
                continue
            
            print(f"\nFile interrogato: {loaded_kb_path}")
            
            while True:
                print("\n" + "-"*60)
                print("Query predefinite disponibili:")
                for i, (name, query) in enumerate(SPARQL_QUERIES.items(), 1):
                    print(f"  {i}) {name}")
                
                print("  c) Query personalizzata")
                print("  esc) Torna al menu Esportazione RDF")
                
                query_choice = input("\nScelta: ").strip().lower()
                
                if query_choice in ("esc", "q", "0", "exit"):
                    break
                
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
                        results = query_rdf_kb(loaded_kb_path, sparql_query)
                        
                        print(f"\nRISULTATI: {len(results)} righe trovate")
                        print("-"*60)
                        
                        if results:
                            for i, row in enumerate(results, 1):
                                print(f"\nRiga {i}:")
                                for key, value in row.items():
                                    print(f"  {key}: {value}")
                            
                        else:
                            print("Nessun risultato trovato")
                            
                    except Exception as e:
                        print(f"\nERRORE nell'esecuzione query: {e}")
                
                print("\n[INVIO = nuova query | 'esc' = torna al menu Esportazione RDF]")
                back = input("> ").strip().lower()
                if back in ("esc", "q", "0", "exit"):
                    break
        
        
        elif choice == "5":  
            print("\n" + "="*60)
            print("STATISTICHE KNOWLEDGE BASE RDF".center(60))
            print("="*60)
            
            if loaded_kb_path:
                try:
                    print_rdf_summary(describe_rdf(loaded_kb_path))
                except Exception as e:
                    print(f"\n[AVVISO] Analisi file non riuscita: {e}")
                
                print("\n== CONTENUTO CARICATO IN MEMORIA ==")
                print_kb_recap(kb)
                print_rdf_file_stats(loaded_kb_path)
            
            elif loaded_diamond_report:
                print(f"\nPercorso: {loaded_rdf_path}")
                for report in loaded_diamond_report:
                    print_diamond_report(report)
            
            else:
                print("\nNESSUN FILE RDF CARICATO IN MEMORIA")
                print("Usa l'opzione 2 per caricare una KB o un diamante.")
        
        
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
                while True:
                    try:
                        num_examples = int(input("\nSu quanti esempi deve essere effettuato l'addestramento? (da 25 a 53940): ").strip())
                        if 25 <= num_examples <= 53940:
                            break
                        else:
                            print("ERRORE: Il numero deve essere tra 25 e 53940")
                    except ValueError:
                        print("ERRORE: Inserisci un numero valido")

                config.NUM_TRAINING_EXAMPLES = num_examples
                print(f"\nSto rigenerando i dati con {num_examples} diamanti...")
                df = CategoricalDataFrame(num_diamonds=num_examples)   # type: ignore
                print("\nSUCCESSO: Modello addestrato e salvato!")
            else:
                print("\nAddestramento annullato")
        elif choice == "5":
            df.eda()
            input("\nPremi Invio per continuare...")
        elif choice == "6":
            df.plot_reliability_diagram()
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