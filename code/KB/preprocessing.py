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

F1_MACRO_SCORER = make_scorer(f1_score, average="macro", zero_division=0)

CV_SCORING = {
    "roc_auc_ovo": "roc_auc_ovo",
    "f1_macro": F1_MACRO_SCORER,
    "accuracy": "accuracy",
}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _close_figure() -> None:
    plt.close("all")
    gc.collect()


def safe_cv_splits(y, model, max_splits: int = CV_SPLITS) -> int:
    y_arr = np.asarray(y).ravel().astype(int)
    if y_arr.size == 0:
        return 0
    min_count = int(np.min(np.bincount(y_arr)))

    inner_cv = getattr(model, "cv", 0)
    if not isinstance(inner_cv, (int, np.integer)) or isinstance(inner_cv, bool):
        inner_cv = 2   
        
    for n in range(min(max_splits, min_count), 1, -1):
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
    
    
        results = list(prolog.query("prop(Diamond, carat, _)"))
        diamond_ids = list(set([r["Diamond"] for r in results]))
        diamond_ids.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    
        final_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']
    
        data = {column: [] for column in final_columns}
    
        for diamond_id in diamond_ids:
            for column in final_columns:
                if column in ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']:
                    column_class = f"{column}_class"
                    query = list(prolog.query(f"prop({diamond_id}, {column_class}, Value)"))
                    if query:
                        data[column].append(query[0]["Value"])
                    else:
                        data[column].append(None)
                else:
                    query = list(prolog.query(f"prop({diamond_id}, {column}, Value)"))
                    if query:
                        data[column].append(query[0]["Value"])
                    else:
                        data[column].append(None)
    
        df = pd.DataFrame(data)

        for col in df.columns:
            self[col] = df[col]



    def to_csv(self, path: str = CATEGORICAL_CSV) -> None:
        super().to_csv(path, index=False)



    def get_target_column(self: pd.DataFrame) -> str:
    
        if TARGET_COL in self.columns:
            return TARGET_COL
        else:
            raise ValueError("Target column", TARGET_COL,"not found in the DataFrame.")



    def eda(self, graphics: bool = True) -> None:
        print("\n=== DESCRIPTIVE STATISTICAL ANALYSIS ===")
    
        descriptive_stats = pd.DataFrame({
            'Type': self.dtypes,
            'Unique Values': self.nunique(),
            'Non-Null Values': self.count(),
           'Null Values': self.isna().sum(),
            'Mode': self.mode().iloc[0] if not self.empty else None,
            'Mode Freq': [self[col].value_counts().iloc[0] if not self[col].empty else 0 for col in self.columns]
        })
    
        print(descriptive_stats)
    
        print("\n=== NULL VALUES PER COLUMN ===")
        null_counts = self.isna().sum()
        if null_counts.sum() == 0:
            print("No null value found!")
        else:
            print(null_counts)

        if not graphics:
            print("\nStatistical analysis completed. Graphics disabled.")
            return

        target = 'price'
    
        if target in self.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=target, data=self, order=self[target].value_counts().index)
            plt.title(f"Target Class Distribution ({target})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            _close_figure()
        else:
            print(f"Target column '{target}' not found")

        print("\n=== CATEGORICAL ASSOCIATION MATRIX ===")
    
        numeric_columns = []
        categorical_columns = self.columns.tolist()
    
        if len(categorical_columns) > 1:
            
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
        
            cramers_matrix = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))),
                                        index=categorical_columns, columns=categorical_columns)
        
            for i, col1 in enumerate(categorical_columns):
                for j, col2 in enumerate(categorical_columns):
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
            plt.title("Association Matrix (Cramér's V)")
            plt.tight_layout()
            plt.show()
            _close_figure()
        
            print("Cramér's V Matrix (higher values indicate stronger association):")
            print(cramers_matrix.round(3))

        main_variables = ['carat', 'cut', 'color', 'clarity', target]
        present_variables = [col for col in main_variables if col in self.columns]

        if len(present_variables) >= 2:
            n_vars = len(present_variables)
            
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
            
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            
            for i, var_row in enumerate(present_variables):
                for j, var_col in enumerate(present_variables):
                    ax = axes[i, j]
                    
                    if i == j:
                        counts = self[var_row].value_counts().sort_index()
                        ax.bar(range(len(counts)), counts.values, color='skyblue', alpha=0.7)
                        
                        ax.set_title(f'Distribution {var_row}', fontsize=9, pad=8)
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

        print("\n=== DETAILED DISTRIBUTIONS ANALYSIS ===")
    
        for column in self.columns:
            print(f"\n{column.upper()}:")
            counts = self[column].value_counts()
            for value, count in counts.items():
                percentage = (count / len(self)) * 100
                print(f"  {value}: {count} diamonds ({percentage:.1f}%)")

        if target in self.columns:
            print(f"\n=== RELATIONSHIP WITH TARGET ({target}) ===")
        
            predictive_variables = [col for col in self.columns if col != target]
        
            for var in predictive_variables[:4]:
                print(f"\nRelationship {var} → {target}:")
                cross_tab = pd.crosstab(self[var], self[target], normalize='index') * 100
                print(cross_tab.round(1))
            
                if var in ['carat', 'cut', 'color', 'clarity']:
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='Blues')
                    plt.title(f"Distribution of {target} by {var} (%)")
                    plt.tight_layout()
                    plt.show()
                    _close_figure()



    def build_preprocessor(self):
        
        target_col = self.get_target_column()
        
        if target_col not in self.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
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
        
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        if mode not in ("oof", "test"):
            raise ValueError(f"invalid mode: {mode!r}. Use 'oof' or 'test'.")
        
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
            print("The model does not support predict_proba()")
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
                print("✗ No test set saved in the model. "
                      "Retrain the model or use mode='test'.")
                return None
            y_true_cal = np.asarray(test_data["y_test"]).ravel()
            y_proba = model.predict_proba(test_data["X_test"])
        
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
            
            ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Class {cls_name}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('Observed fraction')
            ax.set_title(f'Reliability Plot - Class {cls_name}')
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
            print(f"Brier score for class {le.classes_[i]}: {brier:.4f}")
        
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
        
        min_train_count = min(Counter(np.asarray(y_train).ravel().tolist()).values())
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
            print(f"Model training in progress... (n_estimators={n_estimators}, max_depth={max_depth}, calibration={cal_method} cv={cal_cv})")
        else:
            cal = pipe
            print(f"Model training in progress... (n_estimators={n_estimators}, max_depth={max_depth})")
            print("⚠ Class with a single sample in training: calibration skipped.")
        
        cal.fit(X_train, y_train)
        
        if cal_cv >= 2:
            print(f"✓ Model trained with calibration ({cal_method}, cv={cal_cv})")
        else:
            print("✓ Model trained")
        
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
            
            "test_data": {
                "X_test": X_test,
                "y_test": y_test,
                "features": feats,
            }
        }
        
        joblib.dump(payload, model_path)
        print(f"✓ Model saved in: {model_path}")
        
        removed = delete_facts()
        if removed:
            print(f"✓ Removed {removed} facts from {os.path.basename(PROLOG_FILE)}")
                                         
   
 
   
   
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
        print("MODEL PERFORMANCE EVALUATION".center(60))
        print('='*60)
        
        try:
            payload = joblib.load(model_path)
            model = payload["model"]
            le = payload.get("label_encoder")
            features = payload.get("features")
            class_names = payload.get("class_names", ["low", "medium", "high"])
            test_data = payload.get("test_data")
            
            print(f"✓ Model loaded from: {model_path}")
            print(f"✓ Classes: {class_names}")
            print(f"✓ Number of features: {len(features) if features else 'N/A'}")
            
        except FileNotFoundError:
            print(f"✗ ERROR: Model file not found in {model_path}")
            raise
        except Exception as e:
            print(f"✗ ERROR while loading the model: {e}")
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
        print(f"✓ Dataset dimensions: {X.shape}")
        
        class_distribution = np.bincount(np.asarray(y, dtype=int))
        if hasattr(class_distribution, 'tolist'):
            class_distribution_list = class_distribution.tolist()
        else:
            class_distribution_list = list(class_distribution)
        
        print(f"✓ Classes distribution: {class_distribution_list}")
        
        metrics = {}
        n_splits = safe_cv_splits(y, model)
        cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
              if n_splits >= 2 else None)
        
        if cv is None:
            for name in CV_SCORING:
                metrics[f'cv_{name}_mean'] = None
                metrics[f'cv_{name}_std'] = None
        elif hasattr(model, 'predict_proba'):
            try:
                
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
                print(f"\n⚠ Cross-validation not available: {e}")
                for name in CV_SCORING:
                    metrics[f'cv_{name}_mean'] = None
                    metrics[f'cv_{name}_std'] = None
        
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
            print("\n⚠ No test set in the model: holdout metrics not available.")
            print("  (retrain the model to generate it)")
            y_test = y_pred_test = None
            metrics['test_n_samples'] = None
        
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
        
        print(f"\n{' COMPARISON ':-^60}")
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
        
        # Confusion matrix and accuracy per class: on the test set
        if y_test is not None and y_pred_test is not None:
            cm = confusion_matrix(y_test, y_pred_test)
            metrics['confusion_matrix'] = cm.tolist()
            
            if plot_confusion_matrix:
                print(f"\n{' Confusion Matrix ':-^60}")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=class_names
                )
                disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
                ax.set_title(f"Confusion Matrix (n={len(y_test)})")
                
                ax.text(0.5, -0.15,
                        f"Accuracy: {metrics['test_accuracy']:.3f} | "
                        f"Samples: {len(y_test)}",
                        transform=ax.transAxes, ha='center', fontsize=10)
                
                plt.tight_layout()
                plt.show()
                _close_figure()
            else:
                print("\n⚠ Confusion matrix not displayed "
                      "(plot_confusion_matrix=False)")
            
            print("\nConfusion matrix:")
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(cm_df.to_string())
            
            print("\nAccuracy per class:")
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
        
        print(f"\n{' Evaluation completed ':-^60}")
        print(f"Dataset: {metrics['dataset_size']} samples, {metrics['n_features']} features")
        print(f"Classes: {len(class_names)} ({', '.join(class_names)})")
        print(f"Strategy: holdout test set ({metrics['test_n_samples']} samples)")
        print(f"Confusion matrix displayed: {'Yes' if plot_confusion_matrix else 'No'}")
        
        return metrics



