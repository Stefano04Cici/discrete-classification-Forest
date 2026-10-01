from pathlib import Path as PathlibPath
from matplotlib.path import Path
import pandas as pd
import numpy as np
import gc
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyswip import Prolog
from config import PROLOG_FILE, CATEGORICAL_CSV, TARGET_COL, MODEL_PATH, CV_SPLITS, NUM_TRAINING_EXAMPLES
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    cross_val_predict,
    StratifiedKFold,
    learning_curve,
)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
from collections import Counter


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_curve,
    precision_score,
    recall_score,
)

from examples_csv_to_prolog import execute_insert_facts, delete_facts

# Scorers usati dalla cross validation (F1 macro = media non pesata sulle classi,
# coerente con le metriche riportate nella classification report)
F1_MACRO_SCORER = make_scorer(f1_score, average="macro", zero_division=0)

CV_SCORING = {
    "roc_auc_ovo": "roc_auc_ovo",
    "f1_macro": F1_MACRO_SCORER,
    "accuracy": "accuracy",
}


def fmt(value, digits: int = 3) -> str:
    """Formatta una metrica opzionale, mostrando 'n/d' se non disponibile."""
    if value is None:
        return "n/d"
    return f"{value:.{digits}f}"


def _close_figure() -> None:
    """Chiude la figura e libera subito le risorse Tk sul main thread.

    plt.close() da solo NON basta: le PhotoImage e le Variable di tkinter
    restano appese in cicli di riferimenti e sopravvivono alla chiusura della
    finestra. Le finalizza il GC ciclico, e quel GC puo' scattare dentro il
    thread di joblib che smista i risultati (cross_validate, model.predict):
    li' Tk e' threaded, il chiamante non e' il thread dell'interprete, e
    _tkinter solleva "RuntimeError: main thread is not in main loop" dopo
    un secondo di attesa. Forzando qui la raccolta, tutto si chiude sul
    main thread, dove tk.call funziona.
    """
    plt.close("all")
    gc.collect()


def safe_cv_splits(y, model, max_splits: int = CV_SPLITS) -> int:
    """Quanti fold esterni sono utilizzabili per una valutazione out-of-fold.

    Un classificatore calibrato (CalibratedClassifierCV) ha una CV *interna*
    che gira sui dati di addestramento di ogni fold esterno. Contare le classi
    sul dataset intero non basta: la classe piu' rare che *resta* nel training
    di un fold deve soddisfare anche quel vincolo, altrimenti sklearn solleva
    "Requesting N-fold cross-validation but provided less than N examples".

    Restituisce 0 quando i dati non bastano: in quel caso conviene il test set
    holdout, che non richiede alcun ri-addestramento.
    """
    y_arr = np.asarray(y).ravel().astype(int)
    if y_arr.size == 0:
        return 0
    min_count = int(np.min(np.bincount(y_arr)))

    inner_cv = getattr(model, "cv", 0)
    if not isinstance(inner_cv, (int, np.integer)) or isinstance(inner_cv, bool):
        inner_cv = 2   # LeaveOneOut / "prefit": non predicabile, si resta prudenti

    for n in range(min(max_splits, min_count), 1, -1):
        # StratifiedKFold toglie dalla classe rara al massimo ceil(min_count / n)
        held_out = -(-min_count // n)
        if min_count - held_out >= inner_cv:
            return n
    return 0


class CategoricalDataFrame(pd.DataFrame):

    def __init__(self, num_diamonds: int = NUM_TRAINING_EXAMPLES) -> None:
        super().__init__()
        execute_insert_facts(num_diamonds=num_diamonds)
        self.prolog_to_categorical_dataframe()
        self.to_csv()
        self.train_model(num_examples=num_diamonds)

    def delete_csv(self, path: str = CATEGORICAL_CSV) -> None:
        if os.path.exists(path):
            os.remove(path)  
        
    def prolog_to_categorical_dataframe(self: pd.DataFrame) -> None:
    
    
        prolog = Prolog()
        prolog.consult(PROLOG_FILE)
    
    
        risultati = list(prolog.query("prop(Diamond, carat, _)"))
        diamond_ids = list(set([ris["Diamond"] for ris in risultati]))
        diamond_ids.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    
        colonne_finali = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']
    
        dati = {colonna: [] for colonna in colonne_finali}
    
        for diamond_id in diamond_ids:
            for colonna in colonne_finali:
                if colonna in ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']:
                    classe_colonna = f"{colonna}_class"
                    query = list(prolog.query(f"prop({diamond_id}, {classe_colonna}, Value)"))
                    if query:
                        dati[colonna].append(query[0]["Value"])
                    else:
                        dati[colonna].append(None)
                else:
                    query = list(prolog.query(f"prop({diamond_id}, {colonna}, Value)"))
                    if query:
                        dati[colonna].append(query[0]["Value"])
                    else:
                        dati[colonna].append(None)
    
        df = pd.DataFrame(dati)

        for col in df.columns:
            self[col] = df[col]



    def to_csv(self, path: str = CATEGORICAL_CSV) -> None:
        super().to_csv(path, index=False)



    def get_target_column(self: pd.DataFrame) -> str:
    
        if TARGET_COL in self.columns:
            return TARGET_COL
        else:
            raise ValueError("Colonna target", TARGET_COL,"non trovata nel DataFrame.")



    def eda(self, grafici: bool = True) -> None:
        print("\n=== ANALISI STATISTICA DESCRITTIVA ===")
    
        stats_descrittive = pd.DataFrame({
            'Tipo': self.dtypes,
            'Valori Unici': self.nunique(),
            'Valori Non Nulli': self.count(),
           'Valori Nulli': self.isna().sum(),
            'Moda': self.mode().iloc[0] if not self.empty else None,
            'Freq Moda': [self[col].value_counts().iloc[0] if not self[col].empty else 0 for col in self.columns]
        })
    
        print(stats_descrittive)
    
        print("\n=== VALORI NULLI PER COLONNA ===")
        null_counts = self.isna().sum()
        if null_counts.sum() == 0:
            print("Nessun valore nullo trovato!")
        else:
            print(null_counts)

        if not grafici:
            print("\nAnalisi statistica completata. Grafici disattivati.")
            return

        target = 'price'
    
        if target in self.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=target, data=self, order=self[target].value_counts().index)
            plt.title(f"Distribuzione Classe Target ({target})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            _close_figure()
        else:
            print(f"Colonna target '{target}' non trovata")

        print("\n=== MATRICE DI ASSOCIAZIONE CATEGORIALE ===")
    
        colonne_numeriche = []
        colonne_categoriali = self.columns.tolist()
    
        if len(colonne_categoriali) > 1:
            
            def cramers_v(x, y):
                confusion_matrix = pd.crosstab(x, y)
                chi2_result = chi2_contingency(confusion_matrix)
                chi2_stat = float(np.asarray(chi2_result[0], dtype=float).item())
                n = confusion_matrix.sum().sum()
                phi2 = chi2_stat / n
                r, k = confusion_matrix.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        
            cramers_matrix = pd.DataFrame(np.zeros((len(colonne_categoriali), len(colonne_categoriali))),
                                        index=colonne_categoriali, columns=colonne_categoriali)
        
            for i, col1 in enumerate(colonne_categoriali):
                for j, col2 in enumerate(colonne_categoriali):
                    if i == j:
                        cramers_matrix.iloc[i, j] = 1.0
                    else:
                        try:
                            cramers_matrix.iloc[i, j] = cramers_v(self[col1], self[col2])
                        except:
                            cramers_matrix.iloc[i, j] = 0.0
        
            plt.figure(figsize=(12, 10))
            sns.heatmap(cramers_matrix, annot=True, cmap="coolwarm", center=0, 
                       vmin=0, vmax=1, fmt='.2f')
            plt.title("Matrice di Associazione (Cramér's V)")
            plt.tight_layout()
            plt.show()
            _close_figure()
        
            print("Matrice Cramér's V (valori più alti indicano associazione più forte):")
            print(cramers_matrix.round(3))

        variabili_principali = ['carat', 'cut', 'color', 'clarity', target]
        variabili_presenti = [col for col in variabili_principali if col in self.columns]

        if len(variabili_presenti) >= 2:
            n_vars = len(variabili_presenti)
            
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
            
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            
            for i, var_row in enumerate(variabili_presenti):
                for j, var_col in enumerate(variabili_presenti):
                    ax = axes[i, j]
                    
                    if i == j:
                        counts = self[var_row].value_counts().sort_index()
                        ax.bar(range(len(counts)), counts.values, color='skyblue', alpha=0.7)
                        
                        ax.set_title(f'Distribuzione {var_row}', fontsize=9, pad=8)
                        ax.set_xticks(range(len(counts)))
                        
                        ax.set_xticklabels(counts.index, rotation=60, ha='right', fontsize=7)
                        
                        ax.tick_params(axis='y', labelsize=7)
                    
                    else:
                        cross_tab = pd.crosstab(self[var_row], self[var_col])
                        im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
                        
                        ax.set_title(f'{var_row} vs {var_col}', fontsize=8, pad=6)
                        ax.set_xticks(range(len(cross_tab.columns)))
                        
                        ax.set_xticklabels(cross_tab.columns, rotation=60, ha='right', fontsize=6)
                        ax.set_yticks(range(len(cross_tab.index)))
                        ax.set_yticklabels(cross_tab.index, fontsize=6)
                        
                        if cross_tab.shape[0] <= 4 and cross_tab.shape[1] <= 4:
                            for ii in range(len(cross_tab.index)):
                                for jj in range(len(cross_tab.columns)):
                                    ax.text(jj, ii, f'{cross_tab.iloc[ii, jj]}', 
                                        ha="center", va="center", color="black", fontsize=6)
                        elif cross_tab.shape[0] <= 6 and cross_tab.shape[1] <= 6:
                            for ii in range(len(cross_tab.index)):
                                for jj in range(len(cross_tab.columns)):
                                    if cross_tab.iloc[ii, jj] != 0:
                                        ax.text(jj, ii, f'{cross_tab.iloc[ii, jj]}', 
                                            ha="center", va="center", color="black", fontsize=5)

            plt.tight_layout()
            plt.show()
            _close_figure()

        print("\n=== ANALISI DISTRIBUZIONI DETTAGLIATE ===")
    
        for colonna in self.columns:
            print(f"\n{colonna.upper()}:")
            conteggi = self[colonna].value_counts()
            for valore, count in conteggi.items():
                percentuale = (count / len(self)) * 100
                print(f"  {valore}: {count} diamanti ({percentuale:.1f}%)")

        if target in self.columns:
            print(f"\n=== RELAZIONE CON TARGET ({target}) ===")
        
            variabili_predictive = [col for col in self.columns if col != target]
        
            for var in variabili_predictive[:4]:
                print(f"\nRelazione {var} → {target}:")
                cross_tab = pd.crosstab(self[var], self[target], normalize='index') * 100
                print(cross_tab.round(1))
            
                if var in ['carat', 'cut', 'color', 'clarity']:
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='Blues')
                    plt.title(f"Distribuzione {target} per {var} (%)")
                    plt.tight_layout()
                    plt.show()
                    _close_figure()



    def build_preprocessor(self):
        
        target_col = self.get_target_column()
        
        if target_col not in self.columns:
            raise ValueError(f"Colonna target '{target_col}' non trovata")
        
        ordinal_features = ['carat', 'price', 'depth', 'table', 'x', 'y', 'z']
        ordinal_features = [c for c in ordinal_features if c != target_col]
        
        nominal_features = ['cut', 'color', 'clarity']
        
        feature_cols = [c for c in self.columns if c != target_col]
        ordinal_cols = [c for c in ordinal_features if c in feature_cols]
        nominal_cols = [c for c in nominal_features if c in feature_cols]
        other_cols = [c for c in feature_cols if c not in ordinal_cols + nominal_cols]
        
        if ordinal_cols:
            ordinal_t = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
        
        if nominal_cols:
            nominal_t = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        
        if other_cols:
            other_t = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        
        transformers = []
        if ordinal_cols:
            transformers.append(("ordinal", ordinal_t, ordinal_cols))
        if nominal_cols:
            transformers.append(("nominal", nominal_t, nominal_cols))
        if other_cols:
            transformers.append(("other", other_t, other_cols))
        
        preprocessor = ColumnTransformer(transformers)
        selector = SelectKBest(score_func=chi2, k="all")
        
        return preprocessor, selector, target_col, feature_cols



    def plot_reliability_diagram(self, model_path: str = MODEL_PATH,
                                 mode: str = "oof"):
        """Curva di calibrazione e Brier score su predizioni fuori campione.

        mode="oof"  -> out-of-fold via cross_val_predict (usa tutti i dati,
                      accurato ma costa ~CV_SPLITS addestramenti)
        mode="test" -> sul test set holdout salvato nel modello (gratis, ma
                      con pochi campioni la curva e' rumorosa)
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        if mode not in ("oof", "test"):
            raise ValueError(f"mode non valido: {mode!r}. Usa 'oof' o 'test'.")
        
        payload = joblib.load(model_path)
        model = payload["model"]
        le = payload.get("label_encoder")
        test_data = payload.get("test_data")
        
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        if le is None:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = le.transform(y)
        y_encoded = np.asarray(y_encoded).ravel()
        
        if not hasattr(model, 'predict_proba'):
            print("Il modello non supporta predict_proba()")
            return None
        
        if mode == "oof":
            n_splits = safe_cv_splits(y_encoded, model)
            if n_splits < 2:
                mode = "test"
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                y_proba = cross_val_predict(
                    model, X, y_encoded, cv=cv, method="predict_proba", n_jobs=-1
                )
                y_true_cal = y_encoded
        
        if mode != "oof":
            if test_data is None:
                print("✗ Nessun test set salvato nel modello. "
                      "Ri-addestrare il modello oppure usare mode='test'.")
                return None
            y_true_cal = np.asarray(test_data["y_test"]).ravel()
            y_proba = model.predict_proba(test_data["X_test"])
        
        # Se una classe e' assente dai dati di addestramento il modello ha
        # predict_proba con meno colonne: non esiste una colonna su cui plottare.
        n_proba = np.asarray(y_proba).shape[1]
        n_classes = min(len(le.classes_), n_proba)
        
        fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5))
        if n_classes == 1:
            axes = [axes]
        
        for i, (cls_name, ax) in enumerate(zip(le.classes_, axes)):
            prob_true, prob_pred = calibration_curve(
                y_true_cal == i, 
                y_proba[:, i], 
                n_bins=10,
                strategy='uniform'
            )
            
            ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Classe {cls_name}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfettamente calibrato')
            ax.set_xlabel('Probabilità predetta')
            ax.set_ylabel('Frazione osservata')
            ax.set_title(f'Reliability Plot - Classe {cls_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        _close_figure()
        
        from sklearn.metrics import brier_score_loss
        brier_scores = []
        for i in range(n_classes):
            brier = brier_score_loss(y_true_cal == i, y_proba[:, i])
            brier_scores.append((le.classes_[i], brier))
            print(f"Brier score per classe {le.classes_[i]}: {brier:.4f}")
        
        return brier_scores



    def train_model(self, model_path: str = MODEL_PATH, num_examples: int = 500) -> None:
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_

        y_encoded_array = np.asarray(y_encoded).ravel()
        y_encoded_list = list(y_encoded_array)
        class_counts = Counter(y_encoded_list)
        min_class_count = min(class_counts.values())
        stratify_y = y_encoded_array if min_class_count >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded_array, test_size=0.2, stratify=stratify_y, random_state=42
        )
        
        if num_examples < 500:
            n_estimators = 50
            max_depth = 5
            min_samples_split = 10
            min_samples_leaf = 5
            desired_cv = 5
            cal_method = "sigmoid"
        elif num_examples < 2000:
            n_estimators = 100
            max_depth = 8
            min_samples_split = 10
            min_samples_leaf = 3
            desired_cv = 5
            cal_method = "sigmoid"
        elif num_examples < 10000:
            n_estimators = 200
            max_depth = 15
            min_samples_split = 20
            min_samples_leaf = 3
            desired_cv = 5
            cal_method = "sigmoid"
        elif num_examples <= 30000:
            n_estimators = 300
            max_depth = None
            min_samples_split = 20
            min_samples_leaf = 2
            desired_cv = 3
            cal_method = "sigmoid"
        else:
            n_estimators = 500
            max_depth = None
            min_samples_split = 20
            min_samples_leaf = 2
            desired_cv = 3
            cal_method = "isotonic"
        
        # min_class_count e' calcolato sul dataset intero e serve a decidere lo
        # stratify, che agisce su tutti i dati: li' il conteggio e' corretto.
        # Per i fold della calibrazione conta invece la classe rare che resta
        # nel training, che e' solo l'80% del dataset.
        min_train_count = min(Counter(np.asarray(y_train).ravel().tolist()).values())
        # cal_cv = 0 -> non si puo' calibrare (una classe ha un solo campione
        # nel training): si addestra la pipeline nuda invece di far crashare.
        cal_cv = min(desired_cv, min_train_count) if min_train_count >= 2 else 0
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        
        if cal_cv >= 2:
            cal = CalibratedClassifierCV(estimator=pipe, method=cal_method, cv=cal_cv)
            print(f"Addestramento del modello in corso... (n_estimators={n_estimators}, max_depth={max_depth}, calibrazione={cal_method} cv={cal_cv})")
        else:
            cal = pipe
            print(f"Addestramento del modello in corso... (n_estimators={n_estimators}, max_depth={max_depth})")
            print("⚠ Classe con un solo campione nel training: calibrazione saltata.")
        
        cal.fit(X_train, y_train)
        
        if cal_cv >= 2:
            print(f"✓ Modello addestrato con calibrazione ({cal_method}, cv={cal_cv})")
        else:
            print("✓ Modello addestrato")
        
        payload = {
            "model": cal,
            "thresholds": {
                "decision_strategy": "argmax",
                "classes": class_names.tolist()
            },
            "features": feats,
            "calibrated": cal_cv >= 2,
            "calibration": {"method": cal_method if cal_cv >= 2 else None},
            "label_encoder": le,
            "class_names": class_names.tolist(),
            "train_test_split": {
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "random_state": 42
            },
            # Il test set viene persistito: senza di esso le uniche metriche
            # calcolabili su dati mai visti dal modello andrebbero perse.
            "test_data": {
                "X_test": X_test,
                "y_test": y_test,
                "features": feats,
            }
        }
        
        joblib.dump(payload, model_path)
        print(f"✓ Modello salvato in: {model_path}")

        # I fatti Prolog servono solo a costruire il DataFrame: ora che il
        # modello e' persistito occupano disco senza piu' servire.
        removed = delete_facts()
        if removed:
            print(f"✓ Rimossi {removed} fatti da {os.path.basename(PROLOG_FILE)}")
                                         
   
 
   
   
    def plot_learning_curve_single_run(
        self,
        seed: int = 42,
        splits: int = 5,
        n_estimators: int = 300,
        sizes: int | list = 8,
        scoring: str = "f1_weighted",
        title: str | None = None
    ) -> None:
        import matplotlib.pyplot as plt
        
        pre, selector, target, feats = self.build_preprocessor()
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(self[target])

        X, y = self[feats], y_encoded
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )
        
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

        if isinstance(sizes, int):
            train_sizes = np.linspace(0.1, 1.0, sizes)
        else:
            train_sizes = np.array(sizes, dtype=float)

        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
            estimator=pipe,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            return_times=True,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(train_sizes, train_mean, marker="o", label="Training")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        
        ax.plot(train_sizes, test_mean, marker="s", label="Cross-Validation")
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
        
        ax.set_xlabel("Training set size")
        ax.set_ylabel(scoring)
        
        if title is None:
            title = f"Learning Curve (seed={seed}, splits={splits}, n_estimators={n_estimators})"
        ax.set_title(title)
        
        ax.legend()
        plt.tight_layout()
        
        plt.show()
        _close_figure()
    
    
  

         
    def evaluate_model_performance(self, model_path: str = MODEL_PATH, 
                                  plot_confusion_matrix: bool = True):
        print(f"\n{'='*60}")
        print("VALUTAZIONE PERFORMANCE MODELLO".center(60))
        print('='*60)
        
        try:
            payload = joblib.load(model_path)
            model = payload["model"]
            le = payload.get("label_encoder")
            features = payload.get("features")
            class_names = payload.get("class_names", ["low", "medium", "high"])
            test_data = payload.get("test_data")
            
            print(f"✓ Modello caricato da: {model_path}")
            print(f"✓ Classi: {class_names}")
            print(f"✓ Numero di feature: {len(features) if features else 'N/A'}")
            
        except FileNotFoundError:
            print(f"✗ ERRORE: File del modello non trovato in {model_path}")
            raise
        except Exception as e:
            print(f"✗ ERRORE nel caricamento del modello: {e}")
            raise
        
        if features is not None:
            X = self[features]
        else:
            X = self.drop(columns=['price'])
        
        if le is None:
            le = LabelEncoder()
            y_encoded = le.fit_transform(self['price'])
            class_names = le.classes_.tolist()
        else:
            y_encoded = le.transform(self['price'])
            if not isinstance(class_names, list):
                class_names = list(class_names)
        
        y = y_encoded
        print(f"✓ Dimensioni dataset: {X.shape}")
        
        class_distribution = np.bincount(np.asarray(y, dtype=int))
        if hasattr(class_distribution, 'tolist'):
            class_distribution_list = class_distribution.tolist()
        else:
            class_distribution_list = list(class_distribution)
        
        print(f"✓ Distribuzione classi: {class_distribution_list}")
        
        metrics = {}
        n_splits = safe_cv_splits(y, model)
        cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
              if n_splits >= 2 else None)
        
        # ------------------------------------------------------------------
        # 1) CROSS VALIDATION - metriche oneste su tutto il dataset.
        #    cross_validate rifitta il modello su 4/5 dei dati e valuta sul
        #    1/5 mai visto, quindi nessun campione è valutato su dati che il
        #    modello di quel fold ha già visto.
        # ------------------------------------------------------------------
        if cv is None:
            for name in CV_SCORING:
                metrics[f'cv_{name}_mean'] = None
                metrics[f'cv_{name}_std'] = None
        elif hasattr(model, 'predict_proba'):
            try:
                # error_score="raise": un fold che fallisce deve diventare
                # un'eccezione, non un NaN che sipropaga nella media in
                # silenzio e verrebbe stampato come "nan".
                cv_results = cross_validate(
                    model, X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1,
                    error_score="raise",
                )
                
                for name in CV_SCORING:
                    scores = cv_results[f"test_{name}"]
                    metrics[f'cv_{name}_mean'] = float(np.mean(scores))
                    metrics[f'cv_{name}_std'] = float(np.std(scores))
                
                print(f"\n{' CROSS-VALIDATION ':-^60}")
                print(f"ROC-AUC OVO: {metrics['cv_roc_auc_ovo_mean']:.3f} "
                      f"± {metrics['cv_roc_auc_ovo_std']:.3f}")
                print(f"F1 macro:    {metrics['cv_f1_macro_mean']:.3f} "
                      f"± {metrics['cv_f1_macro_std']:.3f}")
                print(f"Accuracy:    {metrics['cv_accuracy_mean']:.3f} "
                      f"± {metrics['cv_accuracy_std']:.3f}")
                
            except (ValueError, IndexError) as e:
                print(f"\n⚠ Cross-validation non disponibile: {e}")
                for name in CV_SCORING:
                    metrics[f'cv_{name}_mean'] = None
                    metrics[f'cv_{name}_std'] = None
        
        # ------------------------------------------------------------------
        # 2) TEST SET HOLDOUT - la valutazione principale.
        #    Il modello è stato addestrato solo su X_train, quindi X_test
        #    non è mai stato visto: è l'unica stima non in-sample.
        # ------------------------------------------------------------------
        if test_data is not None:
            X_test = test_data["X_test"]
            y_test = np.asarray(test_data["y_test"]).ravel()
            
            print(f"\n{' TEST SET HOLDOUT ':-^60}")
            
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            metrics['test_n_samples'] = int(X_test.shape[0])
            metrics['test_accuracy'] = float(accuracy_score(y_test, y_pred_test))
            metrics['test_f1_macro'] = float(
                f1_score(y_test, y_pred_test, average='macro', zero_division=0)
            )
            metrics['test_f1_weighted'] = float(
                f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            )
            
            print(f"Accuracy:           {metrics['test_accuracy']:.3f}")
            print(f"F1-score (macro):   {metrics['test_f1_macro']:.3f}")
            print(f"F1-score (weighted):{metrics['test_f1_weighted']:.3f}")
            
            if y_proba_test is not None:
                try:
                    metrics['test_roc_auc_ovo'] = float(
                        roc_auc_score(y_test, y_proba_test, multi_class='ovo', average='macro')
                    )
                    metrics['test_roc_auc_ovr'] = float(
                        roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='macro')
                    )
                    print(f"ROC-AUC OVO (macro): {metrics['test_roc_auc_ovo']:.3f}")
                    print(f"ROC-AUC OVR (macro): {metrics['test_roc_auc_ovr']:.3f}")
                except (ValueError, IndexError):
                    metrics['test_roc_auc_ovo'] = None
                    metrics['test_roc_auc_ovr'] = None
            
            print(f"\n{' Classification Report ':-^60}")
            print(classification_report(
                le.inverse_transform(y_test),
                le.inverse_transform(y_pred_test),
                target_names=class_names, digits=3
            ))
        else:
            print("\n⚠ Nessun test set nel modello: metriche holdout non disponibili.")
            print("  (ri-addestrare il modello per generarlo)")
            y_test = y_pred_test = None
            metrics['test_n_samples'] = None
        
        # ------------------------------------------------------------------
        # 3) IN-SAMPLE - declassate con prefisso 'insample_'.
        #    Servono solo come confronto: il modello ha già memorizzato
        #    questi dati, quindi non sono una stima di generalizzazione.
        # ------------------------------------------------------------------
        y_pred_insample = model.predict(X)
        y_proba_insample = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        metrics['insample_accuracy'] = float(accuracy_score(y, y_pred_insample))
        metrics['insample_f1_macro'] = float(
            f1_score(y, y_pred_insample, average='macro', zero_division=0)
        )
        metrics['insample_f1_weighted'] = float(
            f1_score(y, y_pred_insample, average='weighted', zero_division=0)
        )
        
        if y_proba_insample is not None:
            try:
                metrics['insample_roc_auc_ovo'] = float(
                    roc_auc_score(y, y_proba_insample, multi_class='ovo', average='macro')
                )
                metrics['insample_roc_auc_ovr'] = float(
                    roc_auc_score(y, y_proba_insample, multi_class='ovr', average='macro')
                )
            except (ValueError, IndexError):
                metrics['insample_roc_auc_ovo'] = None
                metrics['insample_roc_auc_ovr'] = None
        
        print(f"\n{' CONFRONTO ':-^60}")
        print(f"{'':22}{'CV':>10}{'TEST':>12}{'insample':>14}")
        print("-" * 58)
        print(f"{'Accuracy':22}"
              f"{fmt(metrics.get('cv_accuracy_mean')):>10}"
              f"{fmt(metrics.get('test_accuracy')):>12}"
              f"{metrics['insample_accuracy']:>14.3f}")
        print(f"{'F1 macro':22}"
              f"{fmt(metrics.get('cv_f1_macro_mean')):>10}"
              f"{fmt(metrics.get('test_f1_macro')):>12}"
              f"{metrics['insample_f1_macro']:>14.3f}")
        if metrics.get('test_roc_auc_ovo') is not None:
            print(f"{'ROC-AUC OVO':22}"
                  f"{fmt(metrics.get('cv_roc_auc_ovo_mean')):>10}"
                  f"{metrics['test_roc_auc_ovo']:>12.3f}"
                  f"{fmt(metrics.get('insample_roc_auc_ovo')):>14}")
        
        # Matrice di confusione e accuratezza per classe: sul test set
        if y_test is not None and y_pred_test is not None:
            cm = confusion_matrix(y_test, y_pred_test)
            metrics['confusion_matrix'] = cm.tolist()
            
            if plot_confusion_matrix:
                print(f"\n{' Matrice di Confusione ':-^60}")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=class_names
                )
                disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
                ax.set_title(f"Matrice di Confusione (n={len(y_test)})")
                
                ax.text(0.5, -0.15,
                        f"Accuracy: {metrics['test_accuracy']:.3f} | "
                        f"Campioni: {len(y_test)}",
                        transform=ax.transAxes, ha='center', fontsize=10)
                
                plt.tight_layout()
                plt.show()
                _close_figure()
            else:
                print("\n⚠ Matrice di confusione non visualizzata "
                      "(plot_confusion_matrix=False)")
            
            print("\nMatrice di confusione:")
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(cm_df.to_string())
            
            print("\nAccuratezza per classe:")
            per_class = {}
            for i, class_name in enumerate(class_names):
                total = cm[i].sum() if i < len(cm) else 0
                acc = cm[i, i] / total if total > 0 else 0
                per_class[class_name] = float(acc)
                print(f"  {class_name}: {acc:.3f} ({cm[i, i]}/{total})")
            metrics['test_per_class_accuracy'] = per_class
            metrics['confusion_matrix_df'] = cm_df.to_dict()
        else:
            metrics['confusion_matrix'] = None
        
        metrics['model_path'] = model_path
        metrics['dataset_size'] = len(self)
        metrics['n_features'] = X.shape[1]
        metrics['n_classes'] = len(class_names)
        metrics['class_names'] = class_names
        metrics['class_distribution'] = class_distribution_list
        metrics['evaluation_strategy'] = "train_test_split"
        metrics['plot_confusion_matrix'] = plot_confusion_matrix
        
        print(f"\n{' Valutazione completata ':-^60}")
        print(f"Dataset: {metrics['dataset_size']} campioni, {metrics['n_features']} feature")
        print(f"Classi: {len(class_names)} ({', '.join(class_names)})")
        print(f"Strategia: holdout test set ({metrics['test_n_samples']} campioni)")
        print(f"Matrice di confusione visualizzata: {'Sì' if plot_confusion_matrix else 'No'}")
        
        return metrics



