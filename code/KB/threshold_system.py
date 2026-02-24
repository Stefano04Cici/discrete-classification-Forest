from typing import Tuple, Dict, Any, Optional, List  
from enum import Enum  
import numpy as np  
import pandas as pd  
from config import MINIKB_PATH, EXKB_PATH
import json
from pathlib import Path


class BeautyLevel(Enum):

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


def get_hierarchy_level(categorical_value: str) -> int:
    
    categorical_value = categorical_value.lower()
    
    general_map = {
        'low': 1,
        'medium': 2,
        'high': 3
    }
    
    if categorical_value in general_map:
        return general_map[categorical_value]
    
    cut_map = {
        'fair': 1,
        'good': 2,
        'very_good': 3,
        'premium': 4,
        'ideal': 5
    }
    
    if categorical_value in cut_map:
        return cut_map[categorical_value]
    
    color_map = {
        'j': 1,
        'i': 2,
        'h': 3,
        'g': 4,
        'f': 5,
        'e': 6,
        'd': 7
    }
    
    if categorical_value in color_map:
        return color_map[categorical_value]
    
    clarity_map = {
        'i1': 1,
        'si2': 2,
        'si1': 3,
        'vs2': 4,
        'vs1': 5,
        'vvs2': 6,
        'vvs1': 7,
        'if': 8
    }
    
    if categorical_value in clarity_map:
        return clarity_map[categorical_value]
    
    return 0



class Threshold: 
    
    def __init__(self, 
                feature: str, 
                operator: str, 
                value: str, 
                level: BeautyLevel=BeautyLevel.MEDIUM, 
                description: str="") -> None:
        
        self.feature = feature                                      
        self.operator = operator                                    
        self.value = value                                          
        self.level = level                                          
        self.description = description                              
        self.hierarchical_level = get_hierarchy_level(self.value)  
    

    def show_threshold(self):
        
        thr = [{
            
            "feature": self.feature,
            "operator": self.operator,
            "value": self.value,
            "level": self.level,
            "description": self.description
        }]
        
        return thr
    



class MiniKB:
    
    
    def __init__(self):
        
        self._store: Dict[int,Threshold] = {}
        self.position = 0
        self.populate_default_thresholds()
        
    
    def insert_threshold(self,threshold: Threshold) -> None:
        
        self._store[self.position] = threshold
        self.position += 1
        
    
    def get_threshold(self, position: int) -> Optional[Threshold]:
        
        return self._store.get(position)    
    
    
    def query(self,
              feature: Optional[str] = None,
              operator: Optional[str] = None,
              value: Optional[str] = None,
              level: Optional[BeautyLevel] = None,
              description_like: Optional[str] = None
              ) -> pd.DataFrame:
           
            
        rows = []
        
        for position, thresh in self._store.items():
            if feature is not None and thresh.feature != feature:
                continue
            if operator is not None and thresh.operator != operator:
                continue
            if value is not None and thresh.value != value:
                continue
            if level is not None and thresh.level != level:
                continue
            if description_like is not None and (description_like.lower() not in thresh.description.lower()):
                continue

            rows.append({
                'feature': thresh.feature,
                'operator': thresh.operator,
                'value': thresh.value,
                'level': thresh.level,
                'description': thresh.description,             
                'dataset_column': f"{thresh.feature}_class"
            })
             
        return pd.DataFrame(rows, columns=['feature', 'operator', 'value', 'level', 'description', 'dataset_column'])


    def populate_default_thresholds(self) -> None:
        
        self.insert_threshold(Threshold(
            feature="carat",
            operator="<=",
            value="medium",
            level=BeautyLevel.HIGH,
            description="Caratura non superiore a medium per buon rapporto qualità-prezzo"
        ))
        
        self.insert_threshold(Threshold(
            feature="cut",
            operator=">=",
            value="very_good",
            level=BeautyLevel.MEDIUM,
            description="Taglio almeno very_good per buona brillantezza"
        ))
        
        self.insert_threshold(Threshold(
            feature="color",
            operator=">=",
            value="h",
            level=BeautyLevel.MEDIUM,
            description="Colore almeno H (H, G, F, E, D accettabili)"
        ))
        
        self.insert_threshold(Threshold(
            feature="clarity",
            operator=">=",
            value="si1",
            level=BeautyLevel.MEDIUM,
            description="Chiarezza almeno SI1 per poche inclusioni visibili ad occhio nudo"
        ))
        
        self.insert_threshold(Threshold(
            feature="price",
            operator="<=",
            value="medium",
            level=BeautyLevel.HIGH,
            description="Prezzo non superiore a medium per essere considerato conveniente"
        ))
        
        self.insert_threshold(Threshold(
            feature="depth",
            operator="==",
            value="medium",
            level=BeautyLevel.MEDIUM,
            description="Profondità ottimale (60-64%) per massima brillantezza"
        ))
        
        self.insert_threshold(Threshold(
            feature="table",
            operator="==",
            value="medium",
            level=BeautyLevel.MEDIUM,
            description="Tavola ottimale (55-65%) per proporzioni bilanciate"
        ))


    def fuzzy_beauty_score(self, diamond: Dict[str, Any]) -> float:
        
        def fuzzy_evaluate(value: str, 
                        threshold_value: str, 
                        operator: str, 
                        margin: float = 0.5) -> float:
            
            if value is None or threshold_value is None:
                return 0.0
            
            value_level = get_hierarchy_level(value)
            threshold_level = get_hierarchy_level(threshold_value)
            
            if operator == "==":
                if value_level == threshold_level:
                    return 1.0  
                else:
                    distance = abs(value_level - threshold_level)
                    if distance <= margin:
                        return 1.0 - (distance / margin)
                    else:
                        return 0.0
            
            if operator == ">=":
                if value_level >= threshold_level:
                    return 1.0
                else:
                    distance = threshold_level - value_level
                    if distance <= margin:
                        return 1.0 - (distance / margin)
                    else:
                        return 0.0
            
            elif operator == "<=":
                if value_level <= threshold_level:
                    return 1.0
                else:
                    distance = value_level - threshold_level
                    if distance <= margin:
                        return 1.0 - (distance / margin)
                    else:
                        return 0.0
            
            elif operator == ">":
                if value_level > threshold_level:
                    return 1.0
                elif value_level == threshold_level:
                    return 0.5
                else:
                    distance = threshold_level - value_level
                    if distance <= margin:
                        return 0.5 * (1.0 - (distance / margin))
                    else:
                        return 0.0
            
            elif operator == "<":
                if value_level < threshold_level:
                    return 1.0
                elif value_level == threshold_level:
                    return 0.5
                else:
                    distance = value_level - threshold_level
                    if distance <= margin:
                        return 0.5 * (1.0 - (distance / margin))
                    else:
                        return 0.0
            
            return 0.0
        

        scores: List[float] = []
        
        for position, thr in self._store.items():
            
            if thr.feature not in diamond:
                continue
            
            val = diamond[thr.feature]
            if val is None:
                continue
            
            score = fuzzy_evaluate(str(val), thr.value, thr.operator)
            scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0


    def save_to_json(self) -> None:
        
        kb_data = {
            "metadata": {
                "type": "MiniKB",
                "version": "1.0",
                "num_thresholds": len(self._store),
                "position_counter": self.position
            },
            "thresholds": []
        }
        
        for position in sorted(self._store.keys()):
            thr = self._store[position]
            threshold_data = {
                "position": position,
                "feature": thr.feature,
                "operator": thr.operator,
                "value": thr.value,
                "level": thr.level.value,  
                "description": thr.description,
                "hierarchical_level": thr.hierarchical_level
            }
            kb_data["thresholds"].append(threshold_data)
        
        path = Path(MINIKB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(MINIKB_PATH, 'w', encoding='utf-8') as f:
            json.dump(kb_data, f, indent=2, ensure_ascii=False)


    def load_from_json(self) -> None:
        
        with open(MINIKB_PATH, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        self._store.clear()
        self.position = 0
        
        for thr_data in kb_data.get("thresholds", []):
            level_value = thr_data["level"]
            beauty_level = BeautyLevel(level_value)  
            
            threshold = Threshold(
                feature=thr_data["feature"],
                operator=thr_data["operator"],
                value=thr_data["value"],
                level=beauty_level,
                description=thr_data["description"]
            )
            
            position = thr_data.get("position", self.position)
            self._store[position] = threshold
            
            if position >= self.position:
                self.position = position + 1



class ExtendedKB(MiniKB):
   
    
    def __init__(self):
        super().__init__()
        self.composite_rules: List[Dict[str,Any]]=[]


    def add_composite_rule(self, 
                           name: str, 
                           conditions: List[Tuple[str,str,Any]],
                           beautyLevel: BeautyLevel):
        
        self.composite_rules.append({
            "name": name,
            "conditions": conditions,
            "BeautyLevel": beautyLevel
        })

    
    def save_to_json(self) -> None:
        
        super().save_to_json()
        
        serializable_rules = []
        for rule in self.composite_rules:
            serializable_rule = {
                "name": rule["name"],
                "conditions": rule["conditions"],
                "BeautyLevel": rule["BeautyLevel"].value  
            }
            serializable_rules.append(serializable_rule)
        
        composite_data = {
            "metadata": {
                "type": "ExtendedKB_CompositeRules",
                "version": "1.0",
                "num_rules": len(self.composite_rules)
            },
            "composite_rules": serializable_rules
        }
        
        path = Path(EXKB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(EXKB_PATH, 'w', encoding='utf-8') as f:
            json.dump(composite_data, f, indent=2, ensure_ascii=False)
        

    def load_from_json(self) -> None:
        
        super().load_from_json()
        
        try:
            with open(EXKB_PATH, 'r', encoding='utf-8') as f:
                composite_data = json.load(f)
            
            loaded_rules = []
            for rule_data in composite_data.get("composite_rules", []):
                beauty_level_str = rule_data.get("BeautyLevel", "medium")
                beauty_level = BeautyLevel(beauty_level_str)
                
                rule = {
                    "name": rule_data["name"],
                    "conditions": rule_data["conditions"],
                    "BeautyLevel": beauty_level 
                }
                loaded_rules.append(rule)
            
            self.composite_rules = loaded_rules
            
        except FileNotFoundError:
            print(f"Nessun file regole composite trovato: {EXKB_PATH}")
            self.composite_rules = []
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing JSON: {e}")
            print("Il file JSON potrebbe essere corrotto o incompleto")
            self.composite_rules = []
        except ValueError as e:
            print(f"Errore nel caricamento BeautyLevel: {e}")
            print("Assicurati che i valori BeautyLevel siano 'low', 'medium' o 'high'")
            self.composite_rules = []    

  

