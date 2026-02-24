from config import RANDOM_DIAMOND, CATEGORICAL_CSV
from typing import Tuple, Dict, Any, Optional, List
from preprocessing import CategoricalDataFrame
import json
import random
import pandas as pd



def random_diamond(out_path: str = RANDOM_DIAMOND) -> Dict[str, Any]:

    df = pd.read_csv(CATEGORICAL_CSV)
    
    df_senza_prezzo = df.copy()  
    
    for nome_colonna_prezzo in ["price", "target", "label", "class"]:
        if nome_colonna_prezzo in df_senza_prezzo.columns:
            df_senza_prezzo = df_senza_prezzo.drop(columns=[nome_colonna_prezzo])
            break  
   
    def generate_casual_value(caratteristica: str) -> str:
        
        nome_caratteristica = caratteristica.lower()
        
        if "carat" in nome_caratteristica:
            return random.choice(["low", "medium", "high"])
        
        if "depth" in nome_caratteristica:
            return random.choice(["low", "medium", "high"])
        
        if "table" in nome_caratteristica:
            return random.choice(["low", "medium", "high"])
        
        if "x" in nome_caratteristica or "y" in nome_caratteristica or "z" in nome_caratteristica:
            return random.choice(["low", "medium", "high"])
        
        if "cut" in nome_caratteristica:
            return random.choice(["fair", "good", "very_good", "premium", "ideal"])
        
        if "color" in nome_caratteristica:
            return random.choice(["d", "e", "f", "g", "h", "i", "j"])
        
        if "clarity" in nome_caratteristica:
            return random.choice(["i1", "si2", "si1", "vs2", "vs1", "vvs2", "vvs1", "if"])
        
        colonna = df_senza_prezzo[caratteristica]
        
        valori_validi = colonna.dropna().unique().tolist()
        
        if not valori_validi:  
            return "medium"  
        
        return random.choice(valori_validi)
    
    diamante_casuale = {}
    
    for caratteristica in df_senza_prezzo.columns:
        valore = generate_casual_value(caratteristica)
        diamante_casuale[caratteristica] = valore
    
    with open(out_path, "w", encoding="utf-8") as file_json:
        json.dump(
            diamante_casuale,       
            file_json,              
            indent=4,               
            ensure_ascii=False      
        )
    
    
    return diamante_casuale

