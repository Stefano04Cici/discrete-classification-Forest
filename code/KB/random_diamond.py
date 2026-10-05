from config import CATEGORICAL_CSV
from typing import Dict, Any, Optional
from preprocessing import CategoricalDataFrame
import json
import random
import pandas as pd



def random_diamond(out_path: Optional[str] = None) -> Dict[str, Any]:

    df = pd.read_csv(CATEGORICAL_CSV)
    
    df_without_price = df.copy()  
    
    for price_column_name in ["price", "target", "label", "class"]:
        if price_column_name in df_without_price.columns:
            df_without_price = df_without_price.drop(columns=[price_column_name])
            break  
   
    def generate_random_value(feature: str) -> str:
        
        feature_name = feature.lower()
        
        if "carat" in feature_name:
            return random.choice(["low", "medium", "high"])
        
        if "depth" in feature_name:
            return random.choice(["low", "medium", "high"])
        
        if "table" in feature_name:
            return random.choice(["low", "medium", "high"])
        
        if "x" in feature_name or "y" in feature_name or "z" in feature_name:
            return random.choice(["low", "medium", "high"])
        
        if "cut" in feature_name:
            return random.choice(["fair", "good", "very_good", "premium", "ideal"])
        
        if "color" in feature_name:
            return random.choice(["d", "e", "f", "g", "h", "i", "j"])
        
        if "clarity" in feature_name:
            return random.choice(["i1", "si2", "si1", "vs2", "vs1", "vvs2", "vvs1", "if"])
        
        column = df_without_price[feature]
        
        valid_values = column.dropna().unique().tolist()
        
        if not valid_values:  
            return "medium"  
        
        return random.choice(valid_values)
    
    random_diamond_data = {}
    
    for feature in df_without_price.columns:
        value = generate_random_value(feature)
        random_diamond_data[feature] = value
    
    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as file_json:
            json.dump(
                random_diamond_data,       
                file_json,              
                indent=4,               
                ensure_ascii=False      
            )
    
    return random_diamond_data

