from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path
import re
import os
from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD, URIRef, BNode
from config import MINIKB_PATH, EXKB_PATH, TEST_OUTPUT_DIR
from threshold_system import ExtendedKB, MiniKB, BeautyLevel, Threshold, respects_threshold


EX = Namespace("http://example.org/diamonds#")
SCHEMA = Namespace("http://schema.org/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
QB = Namespace("http://purl.org/linked-data/cube#")
DC = Namespace("http://purl.org/dc/elements/1.1/")


@dataclass
class DiamondRule:
    name: str
    conditions: List[Tuple[str, str, str]]  
    beauty_level: str


def slugify(text: str) -> str:
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def kb_to_rdf(kb: ExtendedKB, kb_metadata: Optional[Dict[str, Any]] = None) -> Graph:
    g = Graph()
    
    g.bind("ex", EX)           
    g.bind("schema", SCHEMA)   
    g.bind("foaf", FOAF)       
    g.bind("qb", QB)           
    g.bind("dc", DC)           
    g.bind("xsd", XSD)         

    g.add((EX.Diamond, RDF.type, RDFS.Class))
    g.add((EX.DiamondFeature, RDF.type, RDFS.Class))
    g.add((EX.QualityThreshold, RDF.type, RDFS.Class))
    g.add((EX.BeautyRule, RDF.type, RDFS.Class))
    g.add((EX.BeautyLevel, RDF.type, RDFS.Class))
    g.add((EX.QualityCondition, RDF.type, RDFS.Class))
    
    g.add((EX.CaratFeature, RDF.type, RDFS.Class))
    g.add((EX.CutFeature, RDF.type, RDFS.Class))
    g.add((EX.ColorFeature, RDF.type, RDFS.Class))
    g.add((EX.ClarityFeature, RDF.type, RDFS.Class))
    g.add((EX.DimensionFeature, RDF.type, RDFS.Class))
    
    g.add((EX.CaratFeature, RDFS.subClassOf, EX.DiamondFeature))
    g.add((EX.CutFeature, RDFS.subClassOf, EX.DiamondFeature))
    g.add((EX.ColorFeature, RDFS.subClassOf, EX.DiamondFeature))
    g.add((EX.ClarityFeature, RDFS.subClassOf, EX.DiamondFeature))
    g.add((EX.DimensionFeature, RDFS.subClassOf, EX.DiamondFeature))

     
    for level in ["LOW", "MEDIUM", "HIGH"]:
        level_uri = EX[level]
        g.add((level_uri, RDF.type, EX.BeautyLevel))      
        g.add((level_uri, RDFS.label, Literal(level)))    
        g.add((level_uri, DC.description, Literal(f"Livello di bellezza {level}")))
        
        if level == "LOW":
            g.add((level_uri, EX.appreciation, Literal("Basso apprezzamento")))
        elif level == "MEDIUM":
            g.add((level_uri, EX.appreciation, Literal("Apprezzamento medio")))
        elif level == "HIGH":
            g.add((level_uri, EX.appreciation, Literal("Alto apprezzamento")))

    g.add((EX.hasThreshold, RDF.type, RDF.Property))
    g.add((EX.thresholdOperator, RDF.type, RDF.Property))
    g.add((EX.thresholdValue, RDF.type, RDF.Property))
    g.add((EX.indicatesBeautyLevel, RDF.type, RDF.Property))
    g.add((EX.thresholdDescription, RDF.type, RDF.Property))
    g.add((EX.hierarchicalValue, RDF.type, RDF.Property))
    
    g.add((EX.hasCondition, RDF.type, RDF.Property))
    g.add((EX.appliesToFeature, RDF.type, RDF.Property))
    g.add((EX.conditionValue, RDF.type, RDF.Property))
    g.add((EX.ruleName, RDF.type, RDF.Property))
    
    g.add((EX.hasFeature, RDF.type, RDF.Property))
    g.add((EX.featureValue, RDF.type, RDF.Property))
    g.add((EX.featureCategory, RDF.type, RDF.Property))
    
    g.add((EX.mapsToDatasetColumn, RDF.type, RDF.Property))

    feature_categories = {
        'carat': EX.CaratFeature,
        'cut': EX.CutFeature,
        'color': EX.ColorFeature,
        'clarity': EX.ClarityFeature,
        'depth': EX.DimensionFeature,
        'table': EX.DimensionFeature,
        'x': EX.DimensionFeature,
        'y': EX.DimensionFeature,
        'z': EX.DimensionFeature,
        'price': EX.DiamondFeature 
    }
    
    for position, threshold in kb._store.items():
        feature_name = threshold.feature
        feature_uri = EX[feature_name]
        
        feature_category = feature_categories.get(feature_name, EX.DiamondFeature)
        g.add((feature_uri, RDF.type, feature_category))
        g.add((feature_uri, RDFS.label, Literal(feature_name)))
        g.add((feature_uri, DC.description, Literal(f"Caratteristica del diamante: {feature_name}")))
        
        if "carat" in feature_name:
            g.add((feature_uri, EX.featureCategory, Literal("weight")))
            g.add((feature_uri, SCHEMA.unitCode, Literal("CT")))  
        elif "cut" in feature_name:
            g.add((feature_uri, EX.featureCategory, Literal("craftsmanship")))
        elif "color" in feature_name:
            g.add((feature_uri, EX.featureCategory, Literal("color_grade")))
        elif "clarity" in feature_name:
            g.add((feature_uri, EX.featureCategory, Literal("purity")))
        elif feature_name in ['depth', 'table', 'x', 'y', 'z']:
            g.add((feature_uri, EX.featureCategory, Literal("dimension")))
            g.add((feature_uri, SCHEMA.unitCode, Literal("MM")))  

        threshold_uri = URIRef(f"{EX}threshold_{feature_name}_{position}")
        g.add((threshold_uri, RDF.type, EX.QualityThreshold))
        
        g.add((threshold_uri, EX.thresholdOperator, Literal(threshold.operator)))
        g.add((threshold_uri, EX.thresholdValue, Literal(threshold.value, datatype=XSD.string)))
        g.add((threshold_uri, EX.indicatesBeautyLevel, EX[str(threshold.level.value).upper()]))
        g.add((threshold_uri, EX.thresholdDescription, Literal(threshold.description)))
        g.add((threshold_uri, EX.hierarchicalValue, Literal(threshold.hierarchical_level, datatype=XSD.integer)))
        
        g.add((feature_uri, EX.hasThreshold, threshold_uri))

    for rule in getattr(kb, "composite_rules", []):
        rule_slug = slugify(rule["name"])
        rule_uri = URIRef(f"{EX}rule_{rule_slug}")
        
        g.add((rule_uri, RDF.type, EX.BeautyRule))
        g.add((rule_uri, EX.ruleName, Literal(rule["name"])))
        g.add((rule_uri, RDFS.label, Literal(rule["name"])))
        g.add((rule_uri, EX.indicatesBeautyLevel, EX[str(rule["BeautyLevel"].value).upper()]))
        g.add((rule_uri, DC.description, Literal(f"Regola composita: {rule['name']}")))
        
        for i, (feature, operator, value) in enumerate(rule["conditions"]):
            cond_uri = URIRef(f"{rule_uri}/condition/{i}")
            g.add((cond_uri, RDF.type, EX.QualityCondition))
            g.add((cond_uri, EX.appliesToFeature, EX[feature]))
            g.add((cond_uri, EX.thresholdOperator, Literal(operator)))
            g.add((cond_uri, EX.conditionValue, Literal(str(value), datatype=XSD.string)))
            g.add((rule_uri, EX.hasCondition, cond_uri))

    kb_metadata_uri = EX["DiamondsKnowledgeBase"]
    g.add((kb_metadata_uri, RDF.type, QB.DataSet))
    
    if kb_metadata is None:
        kb_metadata = {}
    
    g.add((kb_metadata_uri, DC.title, 
           Literal(kb_metadata.get('title', "Knowledge Base per Valutazione Diamanti"))))
    g.add((kb_metadata_uri, DC.creator, 
           Literal(kb_metadata.get('creator', "Sistema di Intelligenza Artificiale"))))
    g.add((kb_metadata_uri, DC.date, 
           Literal(str(kb_metadata.get('date', "2024")), datatype=XSD.gYear)))
    g.add((kb_metadata_uri, DC.description, 
           Literal(kb_metadata.get('description', "Base di conoscenza per la valutazione della qualità dei diamanti basata su caratteristiche delle 4C"))))
    g.add((kb_metadata_uri, EX.numThresholds, Literal(len(kb._store), datatype=XSD.integer)))
    g.add((kb_metadata_uri, EX.numCompositeRules, 
           Literal(len(getattr(kb, "composite_rules", [])), datatype=XSD.integer)))

    return g


def save_kb_to_rdf(kb: ExtendedKB, output_path: str = "diamonds_kb.ttl") -> str:
    g = kb_to_rdf(kb)
    
    g.serialize(destination=output_path, format="turtle")
    
    print(f"[RDF] Knowledge Base esportata in: {output_path}")
    print(f"[RDF] Triplette RDF generate: {len(g)}")
    
    return output_path


def load_kb_from_rdf(rdf_path: str) -> ExtendedKB:
    g = Graph()
    g.parse(rdf_path, format="turtle")
    
    kb = ExtendedKB()
    EX = Namespace("http://example.org/diamonds#")
    
    kb._store.clear()
    kb.position = 0
    kb.composite_rules = []
    
    feature_types = set([EX.DiamondFeature])
    feature_types |= set(g.subjects(RDFS.subClassOf, EX.DiamondFeature))
    
    for feature_uri in g.subjects(RDF.type, None):
        types = set(g.objects(feature_uri, RDF.type))
        
        if not (types & feature_types):
            continue
        
        feature_name = str(feature_uri).split("#")[-1]
        
        for _, _, threshold_uri in g.triples((feature_uri, EX.hasThreshold, None)):
            operator = None
            value = None
            level = None
            description = ""
            
            for _, pred, obj in g.triples((threshold_uri, None, None)):
                pred_name = str(pred).split("#")[-1]
                
                if pred_name == "thresholdOperator":
                    operator = str(obj)
                elif pred_name == "thresholdValue":
                    value = str(obj)
                elif pred_name == "indicatesBeautyLevel":
                    level_str = str(obj).split("#")[-1]
                    level = BeautyLevel(level_str.lower())
                elif pred_name == "thresholdDescription":
                    description = str(obj)
            
            if operator and value and level:
                threshold = Threshold(
                    feature=feature_name,
                    operator=operator,
                    value=value,
                    level=level,
                    description=description
                )
                kb.insert_threshold(threshold)
    
    for rule_uri, _, _ in g.triples((None, RDF.type, EX.BeautyRule)):
        rule_name = None
        beauty_level = None
        
        for _, pred, obj in g.triples((rule_uri, None, None)):
            pred_name = str(pred).split("#")[-1]
            
            if pred_name == "ruleName":
                rule_name = str(obj)
            elif pred_name == "indicatesBeautyLevel":
                level_str = str(obj).split("#")[-1]
                beauty_level = BeautyLevel(level_str.lower())
        
        if rule_name and beauty_level:
            conditions = []
            for _, _, cond_uri in g.triples((rule_uri, EX.hasCondition, None)):
                feature = None
                operator = None
                value = None
                
                for _, pred, obj in g.triples((cond_uri, None, None)):
                    pred_name = str(pred).split("#")[-1]
                    
                    if pred_name == "appliesToFeature":
                        feature = str(obj).split("#")[-1]
                    elif pred_name == "thresholdOperator":
                        operator = str(obj)
                    elif pred_name == "conditionValue":
                        value = str(obj)
                
                if feature and operator and value:
                    conditions.append((feature, operator, value))
            
            if conditions:
                kb.add_composite_rule(rule_name, conditions, beauty_level)
    
    print(f"[RDF] Knowledge Base caricata da: {rdf_path}")
    print(f"[RDF] Soglie caricate: {len(kb._store)}")
    print(f"[RDF] Regole composite caricate: {len(kb.composite_rules)}")
    
    return kb


def generate_diamond_rdf_report(
    diamond: Dict[str, Any],
    kb: ExtendedKB,
    output_path: str = "diamond_evaluation.ttl"
) -> str:
    
    g = Graph()
    
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("dc", DC)
    
    diamond_id = diamond.get("id", "unknown")
    diamond_uri = EX[f"diamond_{slugify(diamond_id)}"]
    
    g.add((diamond_uri, RDF.type, EX.Diamond))
    g.add((diamond_uri, RDFS.label, Literal(f"Diamante {diamond_id}")))
    g.add((diamond_uri, DC.date, Literal("2024", datatype=XSD.gYear)))
    
    for feature, value in diamond.items():
        if feature != "id":  
            feature_uri = EX[feature]
            g.add((diamond_uri, EX.hasFeature, feature_uri))
            
            value_uri = URIRef(f"{diamond_uri}/{feature}_value")
            g.add((value_uri, RDF.type, EX.DiamondFeature))
            g.add((value_uri, EX.featureValue, Literal(value, datatype=XSD.string)))
            g.add((feature_uri, EX.featureValue, Literal(value, datatype=XSD.string)))
    
    fuzzy_score = kb.fuzzy_beauty_score(diamond)
    
    g.add((diamond_uri, EX.fuzzyBeautyScore, 
           Literal(fuzzy_score, datatype=XSD.float)))
    g.add((diamond_uri, EX.beautyCategory, 
           Literal("HIGH" if fuzzy_score > 0.7 else 
                  "MEDIUM" if fuzzy_score > 0.4 else "LOW")))
    
    for position, thr in kb._store.items():
        if thr.feature in diamond:
            observed_value = diamond[thr.feature]
            
            eval_uri = URIRef(f"{diamond_uri}/eval_{thr.feature}")
            g.add((eval_uri, RDF.type, EX.QualityThreshold))
            g.add((eval_uri, EX.appliesToFeature, EX[thr.feature]))
            g.add((eval_uri, EX.observedValue, Literal(observed_value, datatype=XSD.string)))
            g.add((eval_uri, EX.expectedOperator, Literal(thr.operator)))
            g.add((eval_uri, EX.expectedValue, Literal(thr.value, datatype=XSD.string)))
            
            respected = respects_threshold(observed_value, thr.value, thr.operator)
            g.add((eval_uri, EX.thresholdRespected, Literal(respected, datatype=XSD.boolean)))
            g.add((diamond_uri, EX.hasThresholdEvaluation, eval_uri))
    
    g.serialize(destination=output_path, format="turtle")
    
    print(f"[RDF] Report diamante generato in: {output_path}")
    print(f"[RDF] Punteggio fuzzy: {fuzzy_score:.3f}")
    
    return output_path


def is_diamond_report(rdf_path: str) -> bool:
    
    g = Graph()
    g.parse(rdf_path, format="turtle")
    
    return (None, RDF.type, EX.Diamond) in g


def load_diamond_report_from_rdf(rdf_path: str) -> List[Dict[str, Any]]:
    
    g = Graph()
    g.parse(rdf_path, format="turtle")
    
    reports = []
    for diamond_uri, _, _ in g.triples((None, RDF.type, EX.Diamond)):
        report = {"uri": str(diamond_uri)}
        
        label = next(g.objects(diamond_uri, RDFS.label), None)
        report["label"] = str(label) if label is not None else None
        
        fuzzy = next(g.objects(diamond_uri, EX.fuzzyBeautyScore), None)
        report["fuzzy_beauty_score"] = float(fuzzy) if fuzzy is not None else None
        
        category = next(g.objects(diamond_uri, EX.beautyCategory), None)
        report["beauty_category"] = str(category) if category is not None else None
        
        features = {}
        for feature_uri in g.objects(diamond_uri, EX.hasFeature):
            feature_name = str(feature_uri).split("#")[-1]
            value = next(g.objects(feature_uri, EX.featureValue), None)
            features[feature_name] = str(value) if value is not None else None
        report["features"] = features
        
        evaluations = []
        for eval_uri in g.objects(diamond_uri, EX.hasThresholdEvaluation):
            applied = list(g.objects(eval_uri, EX.appliesToFeature))
            observed = next(g.objects(eval_uri, EX.observedValue), None)
            expected_op = next(g.objects(eval_uri, EX.expectedOperator), None)
            expected_val = next(g.objects(eval_uri, EX.expectedValue), None)
            respected = next(g.objects(eval_uri, EX.thresholdRespected), None)
            
            evaluations.append({
                "feature": str(applied[0]).split("#")[-1] if applied else "?",
                "observed": str(observed) if observed is not None else None,
                "expected_operator": str(expected_op) if expected_op is not None else None,
                "expected_value": str(expected_val) if expected_val is not None else None,
                "respected": respected.toPython() if respected is not None else None,
            })
        report["evaluations"] = evaluations
        
        reports.append(report)
    
    return reports


def describe_rdf(rdf_path: str) -> Dict[str, Any]:
    
    g = Graph()
    g.parse(rdf_path, format="turtle")
    
    def localname(uri) -> str:
        return str(uri).split("#")[-1]
    
    def as_str(value) -> Optional[str]:
        return str(value) if value is not None else None
    
    def as_int(value) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    
    feature_types = set([EX.DiamondFeature])
    feature_types |= set(g.subjects(RDFS.subClassOf, EX.DiamondFeature))
    
    info: Dict[str, Any] = {
        "path": rdf_path,
        "triples": len(g),
        "type": "kb",
        "beauty_levels": [],
        "kb": None,
        "models": [],
        "features": [],
        "rules": [],
    }
    
    if (None, RDF.type, EX.Diamond) in g:
        info["type"] = "diamond"
    
    models = list(g.subjects(RDF.type, SCHEMA.SoftwareApplication))
    if models:
        info["type"] = "integrated"
    
    for level_uri in g.subjects(RDF.type, EX.BeautyLevel):
        label = next(g.objects(level_uri, RDFS.label), None)
        info["beauty_levels"].append({
            "uri": str(level_uri),
            "label": as_str(label) if label is not None else localname(level_uri),
            "appreciation": as_str(next(g.objects(level_uri, EX.appreciation), None)),
        })
    
    kb_uri = EX["DiamondsKnowledgeBase"]
    if (kb_uri, RDF.type, QB.DataSet) in g:
        info["kb"] = {
            "uri": str(kb_uri),
            "title": as_str(next(g.objects(kb_uri, DC.title), None)),
            "creator": as_str(next(g.objects(kb_uri, DC.creator), None)),
            "date": as_str(next(g.objects(kb_uri, DC.date), None)),
            "description": as_str(next(g.objects(kb_uri, DC.description), None)),
            "num_thresholds": as_int(next(g.objects(kb_uri, EX.numThresholds), None)),
            "num_composite_rules": as_int(next(g.objects(kb_uri, EX.numCompositeRules), None)),
            "models": [localname(m) for m in g.objects(kb_uri, EX.complementsMLModel)],
        }
    
    for model_uri in models:
        info["models"].append({
            "uri": str(model_uri),
            "name": localname(model_uri),
            "title": as_str(next(g.objects(model_uri, DC.title), None)),
            "description": as_str(next(g.objects(model_uri, DC.description), None)),
            "accuracy": next(g.objects(model_uri, EX.modelAccuracy), None),
            "features": [localname(f) for f in g.objects(model_uri, EX.usesFeature)],
            "kb": [localname(k) for k in g.objects(model_uri, EX.complementsKnowledgeBase)],
        })
    
    feature_uris = set()
    for feature_type in feature_types:
        feature_uris |= set(g.subjects(RDF.type, feature_type))
    
    for f_uri in sorted(feature_uris, key=lambda u: str(u)):
        thresholds = []
        for thr_uri in g.objects(f_uri, EX.hasThreshold):
            thresholds.append({
                "uri": str(thr_uri),
                "operator": as_str(next(g.objects(thr_uri, EX.thresholdOperator), None)),
                "value": as_str(next(g.objects(thr_uri, EX.thresholdValue), None)),
                "level": localname(next(g.objects(thr_uri, EX.indicatesBeautyLevel), None) or ""),
                "description": as_str(next(g.objects(thr_uri, EX.thresholdDescription), None)),
            })
        info["features"].append({
            "name": localname(f_uri),
            "uri": str(f_uri),
            "types": [localname(t) for t in g.objects(f_uri, RDF.type)],
            "category": as_str(next(g.objects(f_uri, EX.featureCategory), None)),
            "unit": as_str(next(g.objects(f_uri, SCHEMA.unitCode), None)),
            "thresholds": thresholds,
        })
    
    for rule_uri in g.subjects(RDF.type, EX.BeautyRule):
        conditions = []
        for cond_uri in g.objects(rule_uri, EX.hasCondition):
            conditions.append({
                "feature": localname(next(g.objects(cond_uri, EX.appliesToFeature), None) or ""),
                "operator": as_str(next(g.objects(cond_uri, EX.thresholdOperator), None)),
                "value": as_str(next(g.objects(cond_uri, EX.conditionValue), None)),
            })
        info["rules"].append({
            "uri": str(rule_uri),
            "name": as_str(next(g.objects(rule_uri, EX.ruleName), None)),
            "level": localname(next(g.objects(rule_uri, EX.indicatesBeautyLevel), None) or ""),
            "conditions": conditions,
        })
    
    return info


def query_rdf_kb(
    rdf_path: str,
    sparql_query: str,
    fallback_simple: bool = True
) -> List[Dict[str, str]]:
    
    from rdflib import Graph
    import json
    
    g = Graph()
    results = []
    
    try:
        g.parse(rdf_path, format="turtle")
        
        qres = g.query(sparql_query)
        
        json_str = qres.serialize(format='json')
        
        if json_str is None or json_str == '':
            print("[RDF] Warning: la serializzazione dei risultati ha restituito una stringa vuota")
            return []
        
        json_data = json.loads(json_str)
        
        if 'results' in json_data and 'bindings' in json_data['results']:
            for binding in json_data['results']['bindings']:
                row_dict = {}
                for key, value_info in binding.items():
                    row_dict[key] = value_info.get('value', '')
                results.append(row_dict)
        
        elif 'boolean' in json_data:
            results.append({'boolean': str(json_data['boolean'])})
        
        print(f"[RDF] Risultati: {len(results)}")
        
    except Exception as e:
        print(f"[RDF] Errore opzione nucleare: {e}")
        import traceback
        traceback.print_exc()
        
        if fallback_simple:
            return query_rdf_kb_fallback(rdf_path, sparql_query)
        else:
            return []
    
    return results


def query_rdf_kb_fallback(
    rdf_path: str,
    sparql_query: str
) -> List[Dict[str, str]]:
    
    from rdflib import Graph
    
    g = Graph()
    results = []
    
    try:
        g.parse(rdf_path, format="turtle")
        qres = g.query(sparql_query)
        
        for row in qres:
            row_dict = {}
            
            if isinstance(row, bool):
                row_dict["result"] = str(row)
            elif hasattr(row, '__iter__') and not isinstance(row, str):
                try:
                    for i, item in enumerate(row):
                        row_dict[f"var_{i}"] = str(item) if item is not None else ""
                except TypeError:
                    row_dict["value"] = str(row) if row is not None else ""
            else:
                row_dict["value"] = str(row) if row is not None else ""
            
            results.append(row_dict)
        
        print(f"[RDF] Fallback: {len(results)} risultati")
        
    except Exception as e:
        print(f"[RDF] Errore anche nel fallback: {e}")
    
    return results



SPARQL_QUERIES = {
    "all_features": """
        PREFIX ex: <http://example.org/diamonds#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?feature ?label
        WHERE {
            ?feature a/rdfs:subClassOf* ex:DiamondFeature .
            ?feature rdfs:label ?label .
        }
        ORDER BY ?label
    """,
    
    "simple_thresholds": """
        PREFIX ex: <http://example.org/diamonds#>
        
        SELECT ?feature ?operator ?value ?level
        WHERE {
            ?feature ex:hasThreshold ?threshold .
            ?threshold ex:thresholdOperator ?operator .
            ?threshold ex:thresholdValue ?value .
            ?threshold ex:indicatesBeautyLevel ?level .
        }
        LIMIT 10
    """,
    
    "count_features": """
        PREFIX ex: <http://example.org/diamonds#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT (COUNT(DISTINCT ?feature) AS ?count)
        WHERE {
            ?feature a/rdfs:subClassOf* ex:DiamondFeature .
        }
    """,
    
    "composite_thresholds": """
        PREFIX ex: <http://example.org/diamonds#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?rule ?name ?level ?feature ?operator ?value
        WHERE {
            ?rule a ex:BeautyRule .
            ?rule ex:ruleName ?name .
            ?rule ex:indicatesBeautyLevel ?level .
            ?rule ex:hasCondition ?condition .
            ?condition ex:appliesToFeature ?feature .
            ?condition ex:thresholdOperator ?operator .
            ?condition ex:conditionValue ?value .
        }
        ORDER BY ?name ?feature
    """
}


def export_kb_rdf(
    kb: ExtendedKB,
    output_base: str = "diamonds_ai_system",
    kb_metadata: Optional[Dict[str, Any]] = None
) -> str:
    
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    output_ttl = os.path.join(TEST_OUTPUT_DIR, f"{output_base}_kb.ttl")
    
    g = kb_to_rdf(kb, kb_metadata)
    g.bind("schema", SCHEMA)
    g.bind("dc", DC)
    
    g.serialize(destination=output_ttl, format="turtle")
    
    print(f"[RDF] Knowledge Base esportata in: {output_ttl}")
    print(f"[RDF] Triplette RDF generate: {len(g)}")
    
    return output_ttl