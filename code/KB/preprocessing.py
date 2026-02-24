from pathlib import Path as PathlibPath
from matplotlib.path import Path
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyswip import Prolog
from config import PROLOG_FILE, CATEGORICAL_CSV, TARGET_COL,MODEL_PATH, CV_SPLITS 
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import json


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score        
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
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



class CategoricalDataFrame(pd.DataFrame):
    
    
    def __init__(self) -> None:
        super().__init__()
        self.prolog_to_categorical_dataframe()
        self.to_csv()
        self.train_model()
    

    
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
            
        pd.DataFrame.to_csv(self, path, index=False)



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
        else:
            print(f"Colonna target '{target}' non trovata")

        print("\n=== MATRICE DI ASSOCIAZIONE CATEGORIALE ===")
    
        colonne_numeriche = []
        colonne_categoriali = self.columns.tolist()
    
        if len(colonne_categoriali) > 1:
            
            def cramers_v(x, y):
                confusion_matrix = pd.crosstab(x, y)
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                phi2 = chi2 / n
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



    def plot_reliability_diagram(self, model_path: str = MODEL_PATH):
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        payload = joblib.load(model_path)
        model = payload["model"]
        le = payload.get("label_encoder")
        
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        if le is None:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = le.transform(y)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            
            n_classes = len(le.classes_)
            
            fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5))
            if n_classes == 1:
                axes = [axes]
            
            for i, (cls_name, ax) in enumerate(zip(le.classes_, axes)):
                prob_true, prob_pred = calibration_curve(
                    y_encoded == i, 
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
            
            from sklearn.metrics import brier_score_loss
            brier_scores = []
            for i in range(n_classes):
                brier = brier_score_loss(y_encoded == i, y_proba[:, i])
                brier_scores.append((le.classes_[i], brier))
                print(f"Brier score per classe {le.classes_[i]}: {brier:.4f}")
            
            return brier_scores
        else:
            print("Il modello non supporta predict_proba()")
            return None



    def train_model(self, model_path: str = MODEL_PATH, plot_reliability: bool = True) -> None:
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        
        method: str = "sigmoid"
        cal = CalibratedClassifierCV(estimator=pipe, method=method, cv=3)
        
        print("Addestramento del modello in corso...")
        cal.fit(X_train, y_train)
        print(f"✓ Modello addestrato con calibrazione ({method})")
        
        payload = {
            "model": cal,
            "thresholds": {
                "decision_strategy": "argmax",
                "classes": class_names.tolist()
            },
            "features": feats,
            "calibrated": True,
            "calibration": {"method": method},
            "label_encoder": le,
            "class_names": class_names.tolist(),
            "train_test_split": {
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "random_state": 42
            }
        }
        
        joblib.dump(payload, model_path)
        print(f"✓ Modello salvato in: {model_path}")
        
        if plot_reliability:
            self.plot_reliability_diagram(model_path=model_path)          
                                         
   
   
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
        
        class_distribution = np.bincount(y)
        if hasattr(class_distribution, 'tolist'):
            class_distribution_list = class_distribution.tolist()
        else:
            class_distribution_list = list(class_distribution)
        
        print(f"✓ Distribuzione classi: {class_distribution_list}")
        print(f"✓ Valutazione su: TUTTO il dataset ({len(self)} campioni)")
        
        metrics = {}
        
        if hasattr(model, 'predict_proba'):
            cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
            
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, 
                                          scoring='roc_auc_ovo', n_jobs=-1)
                
                metrics['cv_roc_auc_mean'] = float(cv_scores.mean())
                metrics['cv_roc_auc_std'] = float(cv_scores.std())
                
                print(f"\n{' Cross-Validation ROC-AUC ':-^60}")
                print(f"Media (CV={CV_SPLITS}): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"⚠ Cross-validation non disponibile: {e}")
                metrics['cv_roc_auc_mean'] = None
                metrics['cv_roc_auc_std'] = None
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        metrics['accuracy'] = float(accuracy_score(y, y_pred))
        metrics['f1_macro'] = float(f1_score(y, y_pred, average='macro', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y, y_pred, average='weighted', zero_division=0))
        
        if y_proba is not None:
            try:
                metrics['roc_auc_ovo'] = float(roc_auc_score(y, y_proba, multi_class='ovo', average='macro'))
                metrics['roc_auc_ovr'] = float(roc_auc_score(y, y_proba, multi_class='ovr', average='macro'))
            except:
                metrics['roc_auc_ovo'] = None
                metrics['roc_auc_ovr'] = None
        
        print(f"\n{' Metriche Complete (su tutto il dataset) ':-^60}")
        print(f"Accuracy:           {metrics['accuracy']:.3f}")
        print(f"F1-score (macro):   {metrics['f1_macro']:.3f}")
        print(f"F1-score (weighted):{metrics['f1_weighted']:.3f}")
        
        if metrics.get('roc_auc_ovo') is not None:
            print(f"ROC-AUC OVO (macro): {metrics['roc_auc_ovo']:.3f}")
            print(f"ROC-AUC OVR (macro): {metrics['roc_auc_ovr']:.3f}")
        
        print(f"\n{' Classification Report (su tutto il dataset) ':-^60}")
        y_original = le.inverse_transform(y)
        y_pred_original = le.inverse_transform(y_pred)
        
        print(classification_report(y_original, y_pred_original, 
                                    target_names=class_names, digits=3))
        
        if plot_confusion_matrix:
            print(f"\n{' Matrice di Confusione (su tutto il dataset) ':-^60}")
            
            cm = confusion_matrix(y, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names
            )
            disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
            ax.set_title(f"Matrice di Confusione - Tutto il Dataset (n={len(self)})")
            
            accuracy = accuracy_score(y, y_pred)
            ax.text(0.5, -0.15, f"Accuracy: {accuracy:.3f} | Campioni: {len(self)}", 
                    transform=ax.transAxes, ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.show()
            
            print("\nMatrice di confusione (valori assoluti):")
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(cm_df.to_string())
            
            print("\nAccuratezza per classe:")
            for i, class_name in enumerate(class_names):
                class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                print(f"  {class_name}: {class_accuracy:.3f} ({cm[i, i]}/{cm[i].sum()})")
            
            metrics['confusion_matrix'] = cm.tolist()
            metrics['confusion_matrix_df'] = cm_df.to_dict()
        else:
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            print("\n⚠ Matrice di confusione non visualizzata (plot_confusion_matrix=False)")
        
        metrics['model_path'] = model_path
        metrics['dataset_size'] = len(self)
        metrics['n_features'] = X.shape[1]
        metrics['n_classes'] = len(class_names)
        metrics['class_names'] = class_names
        metrics['class_distribution'] = class_distribution_list
        metrics['evaluation_strategy'] = "full_dataset"
        metrics['plot_confusion_matrix'] = plot_confusion_matrix
        
        print(f"\n{' Valutazione completata ':-^60}")
        print(f"Dataset: {metrics['dataset_size']} campioni, {metrics['n_features']} feature")
        print(f"Classi: {len(class_names)} ({', '.join(class_names)})")
        print(f"Strategia: Valutazione su tutto il dataset")
        print(f"Matrice di confusione visualizzata: {'Sì' if plot_confusion_matrix else 'No'}")
        
        return metrics



