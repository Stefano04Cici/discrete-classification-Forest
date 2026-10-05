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
    
    type_map = {
        "diamond": "DIAMOND REPORT",
        "integrated": "KNOWLEDGE BASE INTEGRATED WITH ML MODEL",
        "kb": "KNOWLEDGE BASE"
    }
    
    print("\n" + "="*60)
    print("RDF FILE SUMMARY".center(60))
    print("="*60)
    print(f"  Path: {info.get('path')}")
    print(f"  Triples: {info.get('triples')}")
    print(f"  Type: {type_map.get(info.get('type'), info.get('type'))}")
    
    if info.get('beauty_levels'):
        print("\n== BEAUTY LEVELS ==")
        for lv in info['beauty_levels']:
            print(f"  * {lv['label']}" + (f" ({lv['appreciation']})" if lv.get('appreciation') else ""))
    
    if info.get('models'):
        for m in info['models']:
            print("\n== ML MODEL ==")
            print(f"  Name: {m['name']}")
            if m['title']:
                print(f"  Title: {m['title']}")
            if m['description']:
                print(f"  Description: {m['description']}")
            acc = m['accuracy']
            if isinstance(acc, float):
                print(f"  Accuracy: {acc:.3f}")
            elif acc is not None:
                print(f"  Accuracy: {acc}")
            if m['features']:
                print(f"  Features used ({len(m['features'])}): {', '.join(m['features'])}")
            if m['kb']:
                print(f"  Linked to KB: {', '.join(m['kb'])}")
    elif info.get('type') == 'integrated':
        print("\n== ML MODEL == (no model detected)")
    
    kb = info.get('kb')
    if kb:
        print("\n== KNOWLEDGE BASE ==")
        print(f"  Title: {kb.get('title')}")
        if kb.get('creator'):
            print(f"  Creator: {kb['creator']}")
        if kb.get('date'):
            print(f"  Date: {kb['date']}")
        if kb.get('description'):
            print(f"  Description: {kb['description']}")
        if kb.get('num_thresholds') is not None:
            print(f"  Declared thresholds: {kb['num_thresholds']}")
        if kb.get('num_composite_rules') is not None:
            print(f"  Declared composite rules: {kb['num_composite_rules']}")
        if kb.get('models'):
            print(f"  Completed by model/s: {', '.join(kb['models'])}")
    
    if info.get('features'):
        print("\n== FEATURES ==")
        for f in info['features']:
            details = []
            if f['types']:
                details.append("type " + ", ".join(f['types']))
            if f['category']:
                details.append(f"category {f['category']}")
            if f['unit']:
                details.append(f"unit {f['unit']}")
            
            header = f"  - {f['name']}"
            if details:
                header += "  (" + ", ".join(details) + ")"
            print(header)
            
            if f['thresholds']:
                for t in f['thresholds']:
                    level = f" [{t['level']}]" if t['level'] else ""
                    row = f"      · {t['operator']} {t['value']}{level}"
                    if t['description']:
                        row += f" — {t['description']}"
                    print(row)
            else:
                print("      · no threshold defined")
    
    if info.get('rules'):
        print("\n== COMPOSITE RULES ==")
        for r in info['rules']:
            print(f"  * {r['name']} -> {r['level']}")
            for c in r['conditions']:
                print(f"      if {c['feature']} {c['operator']} {c['value']}")
    else:
        print("\n== COMPOSITE RULES == (none)")
    
    print("\n" + "="*60)


def print_diamond_report(report):
    
    print("\n" + "="*60)
    print("DIAMOND REPORT".center(60))
    print("="*60)
    print(f"Diamond: {report['label']}")
    print(f"URI: {report['uri']}")
    
    if report['features']:
        print("\nFeatures:")
        for feature, value in report['features'].items():
            print(f"  {feature}: {value}")
    
    fuzzy = report['fuzzy_beauty_score']
    if fuzzy is not None:
        print(f"\nFuzzy score: {fuzzy:.3f}")
    if report['beauty_category']:
        print(f"Beauty category: {report['beauty_category']}")
    
    if report['evaluations']:
        print("\nThreshold evaluation:")
        for eval_item in report['evaluations']:
            respected = eval_item['respected']
            status = "OK" if str(respected).lower() == "true" else "NOT respected"
            print(f"  - {eval_item['feature']}: "
                  f"observed={eval_item['observed']}, "
                  f"expected {eval_item['expected_operator']} {eval_item['expected_value']} "
                  f"[{status}]")
    else:
        print("\nNo threshold evaluation in the report")


def print_kb_recap(kb):
    
    print(f"Thresholds loaded: {len(kb._store)}")
    for i, (pos, thr) in enumerate(kb._store.items(), 1):
        print(f"  {i}. {thr.feature} {thr.operator} {thr.value}")
    
    print(f"Composite rules loaded: {len(kb.composite_rules)}")
    for i, rule in enumerate(kb.composite_rules, 1):
        conds = ", ".join(f"{c[0]} {c[1]} {c[2]}" for c in rule["conditions"])
        print(f"  {i}. {rule['name']} -> {rule['BeautyLevel'].value}"
              f"  (if: {conds})")


def print_rdf_file_stats(rdf_path):
    
    from rdflib import Graph
    
    g = Graph()
    g.parse(rdf_path, format="turtle")
    
    print(f"\nRDF KB statistics ({rdf_path}):")
    print(f"Total triples: {len(g)}")
    print(f"Defined namespaces: {len(list(g.namespaces()))}")
    
    subject_counts = {}
    for s, p, o in g:
        pred_name = str(p).split('#')[-1] if '#' in str(p) else str(p)
        subject_counts[pred_name] = subject_counts.get(pred_name, 0) + 1
    
    print("\nTriples per predicate (top 10):")
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
        print(f"\nFeatures defined in RDF: {results[0]['count']}")




def prediction_menu():
    
    global last_tested_diamond
    
    while True:

        print("\n" + "="*60)
        print("PREDICTIONS MENU - AI MODEL TEST".center(60))
        print("="*60)
        print("\nWhat do you want to do?")
        print("1) Enter MANUALLY the features of a diamond")
        print("2) Generate a RANDOM diamond for the test")
        print("3) Load a diamond from a JSON file")
        print("\n'save' - Saves the last tested diamond")
        print("'esc'   - Returns to the main menu")
        print("\n" + "-"*60)
        
        choice = input(">>\t").strip().lower()
        
        if choice == "1":  
            print("\n" + "="*60)
            print("MANUAL DIAMOND ENTRY".center(60))
            print("="*60)
            
            diamond = {}
            
            features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
            
            for feature in features:
                while True:
                    print(f"\nFeature: {feature}")
                    
                    if feature == 'carat':
                        print("   Possible values: low, medium, high")
                    elif feature == 'cut':
                        print("   Possible values: fair, good, very_good, premium, ideal")
                    elif feature == 'color':
                        print("   Possible values: d, e, f, g, h, i, j (d=best, j=worst)")
                    elif feature == 'clarity':
                        print("   Possible values: i1, si2, si1, vs2, vs1, vvs2, vvs1, if (if=best)")
                    elif feature in ['depth', 'table', 'x', 'y', 'z']:
                        print("   Possible values: low, medium, high")
                    
                    value = input(f"   Enter value for {feature}: ").strip().lower()
                    
                    if value:  
                        diamond[feature] = value
                        break
                    else:
                        print("   ERROR: Invalid value. Try again.")
            
            print("\n" + "="*60)
            print("PREDICTION RESULT".center(60))
            print("="*60)
            
            try:
                result = predict_diamond(diamond, thr_mode="argmax")
                
                if isinstance(result[0], str):  
                    predicted_class, probability, _, _ = result
                    print(f"\nPREDICTED CLASS: {predicted_class}")
                    print(f"PROBABILITY: {probability:.2%}")
                    
                    if predicted_class == "low":
                        print("INTERPRETATION: Economic diamond - good quality/price ratio")
                    elif predicted_class == "medium":
                        print("INTERPRETATION: Medium value diamond - quality/price balance")
                    else:
                        print("INTERPRETATION: High value diamond - premium quality")
                        
                else:  
                    predicted_label, probability, _, _ = result
                    class_name = "expensive" if predicted_label == 1 else "economic"
                    print(f"\nPREDICTED CLASS: {class_name} ({predicted_label})")
                    print(f"PROBABILITY: {probability:.2%}")
                
                last_tested_diamond = {
                    'diamond': diamond,
                    'result': result,
                    'mode': "argmax"
                }
                
            except Exception as e:
                print(f"\nERROR during the prediction: {e}")
                print("Check that all the features have been entered correctly.")
        
        
        elif choice == "2":  
            
            print("\n" + "="*60)
            print("RANDOM DIAMOND GENERATION".center(60))
            print("="*60)
            
            while True:
                try:
                    num_diamonds = int(input("\nHow many random diamonds do you want to generate? (1-10): "))
                    if 1 <= num_diamonds <= 10:
                        break
                    else:
                        print("ERROR: Enter a number between 1 and 10")
                except ValueError:
                    print("ERROR: Enter a valid number")
            
            print("\n" + "="*60)
            print("GENERATED DIAMONDS AND PREDICTIONS".center(60))
            print("="*60)
            
            for i in range(num_diamonds):
                print(f"\nDIAMOND #{i+1}")
                print("-"*40)
                
                diamond = random_diamond(f"test_output/diamond_random_{i+1}.json")
                
                print("Features:")
                for feature, value in diamond.items():
                    print(f"  {feature}: {value}")
                
                try:
                    result = predict_diamond(diamond, thr_mode="argmax")
                    
                    if isinstance(result[0], str):  
                        predicted_class, probability, _, _ = result
                        print(f"\n  Predicted price: {predicted_class}")
                        print(f"  Probability: {probability:.2%}")
                    else:  # Binary
                        predicted_label, probability, _, _ = result
                        class_name = "expensive" if predicted_label == 1 else "economic"
                        print(f"\n  Predicted price: {class_name}")
                        print(f"  Probability: {probability:.2%}")
                    
                    last_tested_diamond = {
                        'diamond': diamond,
                        'result': result,
                        'mode': "argmax"
                    }
                
                except Exception as e:
                    print(f"\n  ERROR in the prediction: {e}")
            
            print(f"\nSUCCESS: Generated and analyzed {num_diamonds} random diamonds")
            print("NOTE: The diamonds have been saved as 'diamond_random_X.json'")
        
        
        elif choice == "3":  
            print("\n" + "="*60)
            print("LOAD DIAMOND FROM JSON FILE".center(60))
            print("="*60)
            
            file_path = input("\nEnter the name of the JSON file: ").strip()
            
            if file_path and not os.path.isabs(file_path) and os.path.dirname(file_path) == "":
                if not file_path.endswith(".json"):
                    file_path += ".json"
                file_path = os.path.join("test_output", file_path)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        diamond = json.load(f)
                    
                    print("\nSUCCESS: File loaded correctly!")
                    print("\nFile content:")
                    for feature, value in diamond.items():
                        print(f"  {feature}: {value}")
                    
                    print("\n" + "="*60)
                    print("PREDICTION RESULT".center(60))
                    print("="*60)
                    
                    result = predict_diamond(diamond, thr_mode="argmax")
                    
                    if isinstance(result[0], str):
                        predicted_class, probability, _, _ = result
                        print(f"\nPREDICTED CLASS: {predicted_class}")
                        print(f"PROBABILITY: {probability:.2%}")
                    else:
                        predicted_label, probability, _, _ = result
                        class_name = "expensive" if predicted_label == 1 else "economic"
                        print(f"\nPREDICTED CLASS: {class_name}")
                        print(f"PROBABILITY: {probability:.2%}")
                    
                    last_tested_diamond = {
                        'diamond': diamond,
                        'result': result,
                        'mode': "argmax"
                    }
                    
                except Exception as e:
                    print(f"\nERROR while loading the file: {e}")
            else:
                print(f"\nERROR: File not found: {file_path}")
                if os.path.isdir("test_output"):
                    names = sorted(f for f in os.listdir("test_output") if f.endswith('.json'))
                    if names:
                        print("\nAvailable .json files in test_output/:")
                        for i, name in enumerate(names, 1):
                            print(f"  {i}) {name}")
        
        
        elif choice == "save":  
            if last_tested_diamond is not None:
                filename = input("\nName of the file to save (without extension): ").strip()
                if not filename:
                    filename = "diamond_saved"
                
                filename = "test_output/" + filename + ".json"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(last_tested_diamond['diamond'], f, indent=4, ensure_ascii=False)
                    
                    print(f"\nSUCCESS: Diamond saved in: {filename}")
                    print("NOTE: You can reload it with option 3 of the menu")
                except Exception as e:
                    print(f"\nERROR while saving: {e}")
            else:
                print("\nERROR: No tested diamond to save")
        
        
        elif choice == "esc":  
            print("\nReturning to the main menu...")
            break
        
        else:
            print("\nERROR: Invalid choice. Try again.")


def threshold_menu():
    
    
    print("\nLoading knowledge base...")
    try:
        kb = ExtendedKB()
        kb.load_from_json()
        print("SUCCESS: Knowledge base loaded from file")
    except Exception as e:
        print(f"NOTE: Creating new knowledge base with default values ({e})")
        kb = ExtendedKB()  
        
    while True:
        
        print("\n" + "="*60)
        print("THRESHOLD MENU - DIAMOND EVALUATION".center(60))
        print("="*60)
        print("\nWhat do you want to do?")
        print("1) Evaluate a diamond entered MANUALLY")
        print("2) Evaluate a RANDOM diamond")
        print("3) View all the rules/thresholds")
        print("4) Add a new rule/threshold")
        print("5) Search for specific rules")
        print("6) Save the knowledge base")
        print("\n'esc' - Returns to the main menu")
        print("\n" + "-"*60)
        
        choice = input(">>\t").strip().lower()
        
        if choice == "1":  
            print("\n" + "="*60)
            print("MANUAL DIAMOND EVALUATION".center(60))
            print("="*60)
            
            diamond = {}
            
            features_to_ask = ['carat', 'cut', 'color', 'clarity', 'depth', 'table']
            
            for feature in features_to_ask:
                while True:
                    print(f"\nFeature: {feature}")
                    
                    if feature == 'carat':
                        print("   Example: low, medium, high")
                    elif feature == 'cut':
                        print("   Example: fair, good, very_good, premium, ideal")
                    elif feature == 'color':
                        print("   Example: d, e, f, g, h, i, j")
                    elif feature == 'clarity':
                        print("   Example: i1, si2, si1, vs2, vs1, vvs2, vvs1, if")
                    elif feature in ['depth', 'table']:
                        print("   Example: low, medium, high")
                    
                    value = input(f"   Value for {feature}: ").strip().lower()
                    
                    if value:
                        diamond[feature] = value
                        break
                    else:
                        print("   ERROR: Invalid value")
            
            print("\n" + "="*60)
            print("EVALUATION RESULT".center(60))
            print("="*60)
            
            try:
                score = kb.fuzzy_beauty_score(diamond)
                score_percent = score * 100
                
                print(f"\nQUALITY SCORE: {score:.3f} ({score_percent:.1f}%)")
                print("-"*40)
                
                if score_percent >= 80:
                    print("EXCELLENT - Diamond of very high quality")
                    print("   All the features meet or exceed the expectations")
                elif score_percent >= 60:
                    print("GOOD - Diamond of good quality")
                    print("   Most of the features are satisfactory")
                elif score_percent >= 40:
                    print("MEDIUM - Acceptable diamond")
                    print("   Some features could be improved")
                elif score_percent >= 20:
                    print("LOW - Diamond of lower quality")
                    print("   Many features do not meet the standards")
                else:
                    print("VERY LOW - Insufficient quality")
                    print("   Consider better alternatives")
                
                print("\n" + "-"*60)
                print("DETAIL PER FEATURE")
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
                        print(f"  Rule: {operator} {threshold}")
                        print(f"  Description: {description}")
            
            except Exception as e:
                print(f"\nERROR in the evaluation: {e}")
        
        
        elif choice == "2":  
            print("\n" + "="*60)
            print("RANDOM DIAMOND EVALUATION".center(60))
            print("="*60)
            
           
            diamond = random_diamond("test_output/diamond_evaluation.json")
            
            print("\nGENERATED DIAMOND:")
            print("-"*40)
            for feature, value in diamond.items():
                print(f"  {feature}: {value}")
            
            print("\n" + "-"*60)
            print("KNOWLEDGE BASE EVALUATION")
            print("-"*60)
            
            try:
                score = kb.fuzzy_beauty_score(diamond)
                score_percent = score * 100
                
                print(f"\nQUALITY SCORE: {score:.3f} ({score_percent:.1f}%)")
                
                if score_percent >= 80:
                    print("EXCELLENT - Rare to find a diamond like this!")
                elif score_percent >= 60:
                    print("GOOD - Good purchase")
                elif score_percent >= 40:
                    print("MEDIUM - Price should be low")
                elif score_percent >= 20:
                    print("LOW - Evaluate alternatives")
                else:
                    print("VERY LOW - Not recommended")
                    
            except Exception as e:
                print(f"\nERROR in the evaluation: {e}")
        
        elif choice == "3":  
            
            print("\n" + "="*60)
            print("RULES OF THE KNOWLEDGE BASE".center(60))
            print("="*60)
            
            rules = kb.query()
            
            if len(rules) > 0:
                print(f"\nFound {len(rules)} simple rules:")
                print("-"*60)
                
                for idx, (_, rule) in enumerate(rules.iterrows(), 1):
                    print(f"\n{idx}. {rule['feature']}")
                    print(f"   Operator: {rule['operator']}")
                    print(f"   Value: {rule['value']}")
                    print(f"   Level: {rule['level']}")
                    print(f"   Description: {rule['description']}")
            else:
                print("\nNOTE: No simple rule found in the knowledge base")

            if hasattr(kb, 'composite_rules') and len(kb.composite_rules) > 0:
                print(f"\nFound {len(kb.composite_rules)} composite rules:")
                print("-"*60)
                
                for idx, rule in enumerate(kb.composite_rules, 1):
                    conds = " AND ".join(
                        f"{c[0]} {c[1]} {c[2]}" for c in rule["conditions"]
                    )
                    print(f"\n{idx}. {rule['name']}")
                    print(f"   Conditions: {conds}")
                    print(f"   BeautyLevel: {rule['BeautyLevel'].value}")
            elif hasattr(kb, 'composite_rules'):
                print("\nNOTE: No composite rule found in the knowledge base")
        
        elif choice == "4":  
            
            print("\n" + "="*60)
            print("ADD NEW RULE".center(60))
            print("="*60)
            
            print("\nChoose the type of rule:")
            print("1) Simple rule (threshold for a feature)")
            print("2) Composite rule (combination of multiple features)")
            
            rule_type = input("\nChoice (1 or 2): ").strip()
            
            if rule_type == "1":
                print("\n" + "-"*60)
                print("NEW SIMPLE RULE")
                print("-"*60)
                
                feature = input("\nFeature (e.g.: carat, cut, color): ").strip().lower()
                operator = input("Operator (e.g.: <=, >=, ==): ").strip()
                value = input("Value (e.g.: medium, ideal, h): ").strip().lower()
                
                print("\nAppreciation level:")
                print("1) LOW (low)")
                print("2) MEDIUM (medium)")
                print("3) HIGH (high)")
                
                level_choice = input("Choice (1-3): ").strip()
                if level_choice == "1":
                    level = BeautyLevel.LOW
                elif level_choice == "2":
                    level = BeautyLevel.MEDIUM
                elif level_choice == "3":
                    level = BeautyLevel.HIGH
                else:
                    print("NOTE: Set MEDIUM level by default")
                    level = BeautyLevel.MEDIUM
                
                description = input("\nDescription (explain the rule): ").strip()
                
                try:
                    threshold = Threshold(
                        feature=feature,
                        operator=operator,
                        value=value,
                        level=level,
                        description=description
                    )
                    
                    kb.insert_threshold(threshold)
                    print(f"\nSUCCESS: Rule added for {feature}")
                except Exception as e:
                    print(f"\nERROR while creating the rule: {e}")
            
            elif rule_type == "2":
                print("\n" + "-"*60)
                print("NEW COMPOSITE RULE")
                print("-"*60)
                
                name = input("\nRule name (e.g.: 'PerfectDiamond'): ").strip()
                
                conditions = []
                print("\nAdd conditions (leave the name empty to finish):")
                
                while True:
                    feature = input("\nFeature (leave empty to finish): ").strip().lower()
                    if not feature:
                        break
                    
                    operator = input(f"Operator for {feature} (e.g.: ==, >=): ").strip()
                    value = input(f"Value for {feature}: ").strip().lower()
                    
                    conditions.append((feature, operator, value))
                    print(f"SUCCESS: Condition added: {feature} {operator} {value}")
                
                if conditions:
                    print("\nAppreciation level:")
                    print("1) LOW (low)")
                    print("2) MEDIUM (medium)")
                    print("3) HIGH (high)")
                    
                    level_choice = input("Choice (1-3): ").strip()
                    if level_choice == "1":
                        level = BeautyLevel.LOW
                    elif level_choice == "2":
                        level = BeautyLevel.MEDIUM
                    elif level_choice == "3":
                        level = BeautyLevel.HIGH
                    else:
                        print("NOTE: Set MEDIUM level by default")
                        level = BeautyLevel.MEDIUM
                    
                    try:
                        kb.add_composite_rule(name, conditions, level)
                        print(f"\nSUCCESS: Composite rule '{name}' added with {len(conditions)} conditions")
                    except Exception as e:
                        print(f"\nERROR while adding the rule: {e}")
                else:
                    print("\nERROR: No condition added")
        
        elif choice == "5":  
            
            print("\n" + "="*60)
            print("SEARCH RULES".center(60))
            print("="*60)
            
            print("\nSearch by:")
            print("1) Feature")
            print("2) Appreciation level")
            print("3) Text in the description")
            
            search_type = input("\nChoice (1-3): ").strip()
            
            if search_type == "1":
                feature = input("\nFeature name (e.g.: cut, color): ").strip().lower()
                results = kb.query(feature=feature)
            elif search_type == "2":
                print("\nLevel:")
                print("1) LOW")
                print("2) MEDIUM")
                print("3) HIGH")
                level_choice = input("Choice (1-3): ").strip()
                if level_choice == "1":
                    level = BeautyLevel.LOW
                elif level_choice == "2":
                    level = BeautyLevel.MEDIUM
                elif level_choice == "3":
                    level = BeautyLevel.HIGH
                else:
                    print("NOTE: Searching at MEDIUM level")
                    level = BeautyLevel.MEDIUM
                results = kb.query(level=level)
            elif search_type == "3":
                text = input("\nText to search for in the description: ").strip()
                results = kb.query(description_like=text)
            else:
                results = pd.DataFrame()
            
            if len(results) > 0:
                print(f"\nFound {len(results)} rules:")
                for _, rule in results.iterrows():
                    print(f"\n• {rule['feature']} {rule['operator']} {rule['value']}")
                    print(f"  Level: {rule['level']}")
                    print(f"  Description: {rule['description']}")
            else:
                print("\nNo rule found")
        
        elif choice == "6":  
            try:
                kb.save_to_json()
                print("\nSUCCESS: Knowledge base saved")
            except Exception as e:
                print(f"\nERROR while saving: {e}")
        
        elif choice == "esc":  
            print("\nReturning to the main menu...")
            break
        
        else:
            print("\nERROR: Invalid choice. Try again.")


def rdf_exporter_menu():
    
    print("\nLoading knowledge base for RDF export...")
    try:
        kb = ExtendedKB()
        kb.load_from_json()
        print("SUCCESS: Knowledge base loaded")
    except Exception as e:
        print(f"NOTE: Creating new knowledge base ({e})")
        kb = ExtendedKB()
    
    loaded_rdf_path = None
    loaded_kb_path = None
    loaded_diamond_report = None
    
    while True:
        
        print("\n" + "="*60)
        print("RDF EXPORT MENU - SEMANTIC KNOWLEDGE".center(60))
        print("="*60)
        print("\nWhat do you want to do?")
        print("1) Export the Knowledge Base in RDF")
        print("2) Load a Knowledge Base from an RDF file, or a diamond")
        print("3) Generate RDF reports for a specific diamond")
        print("4) Run SPARQL queries on the KB")
        print("5) Display statistics of the RDF KB, or of the diamond")
        print("\n'esc' - Returns to the main menu")
        print("\n" + "-"*60)
        
        choice = input(">>\t").strip().lower()
        
        if choice == "1":  
            
            print("\n" + "="*60)
            print("KNOWLEDGE BASE RDF EXPORT".center(60))
            print("="*60)
            
            base_name = input("\nBase file name [diamonds_ai_system]: ").strip() or "diamonds_ai_system"
            
            print("\nKnowledge Base information:")
            kb_metadata = {}
            kb_metadata['title'] = input("KB Title [Knowledge Base for Diamond Evaluation]: ").strip() or "Knowledge Base for Diamond Evaluation"
            kb_metadata['creator'] = input("KB Creator [Artificial Intelligence System]: ").strip() or "Artificial Intelligence System"
            kb_metadata['date'] = input("KB Date [2024]: ").strip() or "2024"
            kb_metadata['description'] = input("KB Description [Knowledge base for evaluating the quality of diamonds based on the characteristics of the 4C]: ").strip() or "Knowledge base for evaluating the quality of diamonds based on the characteristics of the 4C"
            
            try:
                result_path = export_kb_rdf(kb, base_name, kb_metadata=kb_metadata)
                
                print("\nSUCCESS: Knowledge Base exported!")
                print(f"  File: {result_path}")
                    
            except Exception as e:
                print(f"\nERROR while exporting: {e}")
        
        
        elif choice == "2":  
            print("\n" + "="*60)
            print("LOAD KNOWLEDGE BASE OR DIAMOND FROM RDF".center(60))
            print("="*60)
            
            print("\nFiles available in test_output/:")
            try:
                files = [f for f in os.listdir("test_output") if f.endswith('.ttl')]
                for i, f in enumerate(files, 1):
                    print(f"  {i}) {f}")
            except:
                files = []
            
            if files:
                file_choice = input("\nFile number or full path: ").strip()
                
                try:
                    if file_choice.isdigit():
                        idx = int(file_choice) - 1
                        if 0 <= idx < len(files):
                            rdf_path = os.path.join("test_output", files[idx])
                        else:
                            print("Invalid number")
                            continue
                    else:
                        rdf_path = file_choice
                    
                    print(f"\nLoading from: {rdf_path}")
                    
                    if is_diamond_report(rdf_path):
                        print("\nFile recognized: DIAMOND REPORT")
                        loaded_diamond_report = load_diamond_report_from_rdf(rdf_path)
                        loaded_kb_path = None
                        
                        if not loaded_diamond_report:
                            print("\nNo diamond found in the report")
                        else:
                            print(f"\nSUCCESS: {len(loaded_diamond_report)} diamond/s loaded in memory.")
                            print("Information available with option 5.")
                        loaded_rdf_path = rdf_path
                    
                    else:
                        loaded_kb = load_kb_from_rdf(rdf_path)
                        loaded_diamond_report = None
                        
                        loaded_kb_path = rdf_path
                        loaded_rdf_path = rdf_path
                        
                        kb = loaded_kb
                        
                        print("\nSUCCESS: Knowledge Base loaded in memory from RDF!")
                        print(f"Thresholds loaded: {len(kb._store)}")
                        print(f"Composite rules: {len(kb.composite_rules)}")
                        print("Summary and statistics available with option 5.")
                        print("KB ready: SPARQL queries (option 4) and statistics (option 5) enabled.")
                
                except Exception as e:
                    print(f"\nERROR while loading: {e}")
            else:
                print("\nNo RDF file found in the test_output/ folder")
        
        
        elif choice == "3":  
            print("\n" + "="*60)
            print("RDF REPORT FOR DIAMOND".center(60))
            print("="*60)
            
            print("\nChoose how to obtain the diamond:")
            print("1) Enter manually")
            print("2) Generate randomly")
            print("3) Use last tested diamond")
            
            diamond_choice = input("\nChoice (1-3): ").strip()
            diamond = None
            
            if diamond_choice == "1":
                diamond = {}
                features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
                
                print("\nEnter the features:")
                for feature in features:
                    value = input(f"{feature}: ").strip().lower()
                    if value:
                        diamond[feature] = value
                    else:
                        diamond[feature] = "medium"  
            elif diamond_choice == "2":
                diamond = random_diamond()
                print("\nDiamond generated randomly")
                
            elif diamond_choice == "3":
                if last_tested_diamond:
                    diamond = last_tested_diamond['diamond']
                    print("\nUsing last tested diamond")
                else:
                    print("\nNo tested diamond available")
                    continue
            else:
                print("Invalid choice")
                continue
            
            if diamond:
                filename = input("\nReport file name [diamond_report.ttl]: ").strip()
                if not filename:
                    filename = "diamond_report.ttl"
                
                if not filename.endswith('.ttl'):
                    filename += '.ttl'
                
                output_path = os.path.join("test_output", filename)
                
                try:
                    result_path = generate_diamond_rdf_report(diamond, kb, output_path)
                    
                    print(f"\nSUCCESS: RDF report generated!")
                    print(f"File: {result_path}")
                    
                    fuzzy_score = kb.fuzzy_beauty_score(diamond)
                    print(f"Fuzzy score of the diamond: {fuzzy_score:.3f}")
                    
                    print("\nReport preview:")
                    with open(result_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:15]
                        for line in lines:
                            print(f"  {line.rstrip()}")
                            
                except Exception as e:
                    print(f"\nERROR while generating the report: {e}")
        
        
        elif choice == "4":  
            print("\n" + "="*60)
            print("SPARQL QUERY ON THE KNOWLEDGE BASE".center(60))
            print("="*60)
            
            if not loaded_kb_path:
                print("\nERROR: No Knowledge Base loaded in memory.")
                print("Use option 2 first to load a KB in memory.")
                continue
            
            print(f"\nQueried file: {loaded_kb_path}")
            
            while True:
                print("\n" + "-"*60)
                print("Available predefined queries:")
                for i, (name, query) in enumerate(SPARQL_QUERIES.items(), 1):
                    print(f"  {i}) {name}")
                
                print("  c) Custom query")
                print("  esc) Returns to the RDF Export menu")
                
                query_choice = input("\nChoice: ").strip().lower()
                
                if query_choice in ("esc", "q", "0", "exit"):
                    break
                
                sparql_query = ""
                
                if query_choice == "c":
                    print("\nEnter your SPARQL query (end with an empty line):")
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
                        print("Invalid number")
                        continue
                else:
                    print("Invalid choice")
                    continue
                
                if sparql_query:
                    try:
                        print("\nRunning query...")
                        results = query_rdf_kb(loaded_kb_path, sparql_query)
                        
                        print(f"\nRESULTS: {len(results)} rows found")
                        print("-"*60)
                        
                        if results:
                            for i, row in enumerate(results, 1):
                                print(f"\nRow {i}:")
                                for key, value in row.items():
                                    print(f"  {key}: {value}")
                            
                        else:
                            print("No result found")
                            
                    except Exception as e:
                        print(f"\nERROR while executing the query: {e}")
                
                print("\n[ENTER = new query | 'esc' = returns to the RDF Export menu]")
                back = input("> ").strip().lower()
                if back in ("esc", "q", "0", "exit"):
                    break
        
        
        elif choice == "5":  
            print("\n" + "="*60)
            print("KNOWLEDGE BASE RDF STATISTICS".center(60))
            print("="*60)
            
            if loaded_kb_path:
                try:
                    print_rdf_summary(describe_rdf(loaded_kb_path))
                except Exception as e:
                    print(f"\n[WARNING] File analysis failed: {e}")
                
                print("\n== CONTENT LOADED IN MEMORY ==")
                print_kb_recap(kb)
                print_rdf_file_stats(loaded_kb_path)
            
            elif loaded_diamond_report:
                print(f"\nPath: {loaded_rdf_path}")
                for report in loaded_diamond_report:
                    print_diamond_report(report)
            
            else:
                print("\nNO RDF FILE LOADED IN MEMORY")
                print("Use option 2 to load a KB or a diamond.")
        
        
        elif choice == "esc":  
            print("\nReturning to the main menu...")
            break
        
        else:
            print("\nERROR: Invalid choice. Try again.")


def ui():
    
    print("\n" + "="*70)
    print("WELCOME TO THE ARTIFICIAL INTELLIGENCE SYSTEM".center(70))
    print("DIAMOND PREDICTION AND EVALUATION".center(70))
    print("="*70)
    
    print("\nThis system allows you to:")
    print("   1) Predict the price of a diamond using AI")
    print("   2) Evaluate the quality of a diamond with expert rules")
    print("   3) Export knowledge in semantic format (RDF)")
    
    print("\n" + "-"*70)
    print("INITIALIZATION OF THE LEARNING MODEL".center(70))
    print("-"*70)
    print("\nI am loading and preparing the diamond data...")
    
    df = CategoricalDataFrame()
    
    print("\nSUCCESS: DATA LOADED CORRECTLY!")
    print(f"   Diamonds in the dataset: {len(df)}")
    print(f"   Available columns: {', '.join(df.columns)}")
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU".center(60))
        print("="*60)
        print("\nWhat do you want to do?")
        print("1) TEST THE AI PREDICTION")
        print("   • Enter or generate diamonds")
        print("   • Get price predictions (low/medium/high)")
        print("   • See the probabilities and the confidence")
        
        print("\n2) EXPLORE EVALUATION THRESHOLDS")
        print("   • Evaluate the quality of diamonds")
        print("   • Manage evaluation rules")
        print("   • Add new expert rules")
        
        print("\n3) RDF EXPORT - SEMANTIC KNOWLEDGE")
        print("   • Export rules in RDF/Turtle format")
        print("   • Run SPARQL queries on the knowledge base")
        print("   • Generate semantic reports for diamonds")
        
        print("\n4) TRAIN THE AI MODEL")
        print("   • Regenerate the model with the current data")
        print("   • Get new performance metrics")
        
        print("\n5) EXPLORATORY DATA ANALYSIS")
        print("   • Visualize the dataset with graphs and statistics")
        print("   • Analyze correlations and patterns in the data")
        
        print("\n6) CHECK THE PERFORMANCE OF THE LEARNING SYSTEM")
        print("   • Evaluate the model with learning curves and reliability diagrams")
        print("   • Get performance metrics and insights on the model's behavior")
        
        print("\n7) EXIT")
        print("\n" + "-"*60)
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == "1":
            prediction_menu()
        elif choice == "2":
            threshold_menu()
        elif choice == "3":
            rdf_exporter_menu()
        elif choice == "4":
            print("\n" + "="*60)
            print("AI MODEL TRAINING".center(60))
            print("="*60)
            print("\nWARNING: this operation could take a few minutes")
            confirm = input("\nProceed with the training? (y/n): ").strip().lower()
            if confirm == 'y':
                while True:
                    try:
                        num_examples = int(input("\nOn how many examples should the training be performed? (from 25 to 53940): ").strip())
                        if 25 <= num_examples <= 53940:
                            break
                        else:
                            print("ERROR: The number must be between 25 and 53940")
                    except ValueError:
                        print("ERROR: Enter a valid number")

                config.NUM_TRAINING_EXAMPLES = num_examples
                print(f"\nI am regenerating the data with {num_examples} diamonds...")
                df = CategoricalDataFrame(num_diamonds=num_examples)   # type: ignore
                print("\nSUCCESS: Model trained and saved!")
            else:
                print("\nTraining cancelled")
        elif choice == "5":
            df.eda()
            input("\nPress Enter to continue...")
        elif choice == "6":
            df.plot_reliability_diagram()
            df.plot_learning_curve_single_run()
            df.evaluate_model_performance()
            input("\nPress Enter to continue...")
        elif choice == "7":
            print("\n" + "="*60)
            print("THANKS FOR USING THE SYSTEM!".center(60))
            print("="*60)
            break
        else:
            print("\nERROR: Invalid choice. Enter a number from 1 to 7.")