
import sqlite3
import argparse
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
try:
    from .preprocessing import preprocessing_for_linear, preprocessing_for_tree, preprocessing_for_catboost
    from .evaluation import evaluate_model, print_evaluation_summary, plot_confusion_matrix, plot_feature_importance
except ImportError:
    from preprocessing import preprocessing_for_linear, preprocessing_for_tree, preprocessing_for_catboost
    from evaluation import evaluate_model, print_evaluation_summary, plot_confusion_matrix, plot_feature_importance
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
# from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for gas monitoring dataset.
    
    This transformer:
    - Aggregates MOS sensors into summary stats
    - Creates differences and ratios between correlated sensors
    - Adds environment × sensor interaction terms
    - Applies log transforms to skewed sensors
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Nothing to fit, this transformer is stateless
        return self

    def transform(self, X):
        X = X.copy()

        # -----------------------------
        # Aggregate MOS sensors
        # -----------------------------
        mos_cols = [
            "MetalOxideSensor_Unit1",
            "MetalOxideSensor_Unit2",
            "MetalOxideSensor_Unit3",
            "MetalOxideSensor_Unit4"
        ]
        X["MOS_mean"] = X[mos_cols].mean(axis=1)


        # -----------------------------
        # Aggregate CO2 sensors
        # -----------------------------
        X["CO2_mean"] = (
            X["CO2_InfraredSensor"] + X["CO2_ElectroChemicalSensor"]
        ) / 2


        # -----------------------------
        # Environment × sensor interactions
        # -----------------------------
        X["Temp_x_Humidity"] = X["Temperature"] * X["Humidity"]
        X["MOSmean_x_Temp"] = X["MOS_mean"] * X["Temperature"]
        X["MOSmean_x_Humidity"] = X["MOS_mean"] * X["Humidity"]

        # -----------------------------
        # Log transforms (helpful for skewed sensor distributions)
        # -----------------------------
    
        return X

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.valid_ranges = {
            "Temperature": (-40, 85),
            "Humidity": (0, 100),
            "CO2_InfraredSensor": (0, 5000),
            "CO2_ElectroChemicalSensor": (0, 5000),
            "MetalOxideSensor_Unit1": (0, None),
            "MetalOxideSensor_Unit2": (0, None),
            "MetalOxideSensor_Unit3": (0, None),
            "MetalOxideSensor_Unit4": (0, None),
            "CO_GasSensor": (0, None),
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # numeric cleaning
        for col, (low, high) in self.valid_ranges.items():
            if col not in X.columns:
                continue
            if low is not None:
                X.loc[X[col] < low, col] = np.nan
            if high is not None:
                X.loc[X[col] > high, col] = np.nan
            X[col] = X[col].interpolate(method="linear", limit_direction="both")

        # categorical cleaning: only fill NaNs in object/string cols
        cat_cols = X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            X[col] = X[col].fillna("missing")

        return X

def load_data(db_path="data/gas_monitoring.db"):
    """Load dataset from SQLite database"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM gas_monitoring;", conn)
    conn.close()

    # --- Normalize labels (target) ---
    df["Activity Level"] = (
        df["Activity Level"]
        .astype(str)                     
        .str.strip()                     
        .str.lower()                     
        .str.replace(r"[_ ]", "", regex=True)  
    )

    # --- Normalize HVAC Operation Mode ---
    if "HVAC Operation Mode" in df.columns:
        df["HVAC Operation Mode"] = (
            df["HVAC Operation Mode"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[_ ]", "", regex=True)
        )
    mapping = {
        "coolingactive": "cooling",
        "heatingactive": "heating",
        "ventilationonly": "ventilation",
        "maintenancemode": "maintenance",
        "ecomode": "eco",
        "off": "off"
    }
    df["HVAC Operation Mode"] = df["HVAC Operation Mode"].map(
        lambda x: mapping.get(x, x)
    )

    # --- Normalize Ambient Light Level ---
    if "Ambient Light Level" in df.columns:
        df["Ambient Light Level"] = (
            df["Ambient Light Level"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[_ ]", "", regex=True)
        )

    # --- Normalize Time of Day ---
    if "Time of Day" in df.columns:
        df["Time of Day"] = (
            df["Time of Day"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[_ ]", "", regex=True)
        )

    return df




def preprocess(df):
    """Create a preprocessor that auto-selects columns at fit-time (after FE)."""
    # we don't need column lists anymore
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_transformer, make_column_selector(dtype_include=object)),
        ],
        remainder="drop"   # engineered numeric columns will be captured by the numeric selector
    )

    # y/label encoder (unchanged)
    X = df.drop(columns=["Activity Level", "Session ID"])
    le = LabelEncoder()
    y = le.fit_transform(df["Activity Level"])
    return X, y, preprocessor, None, None, le

def get_models():
    """Define candidate models and hyperparameter grids"""
    models = {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"classifier__C": [0.1, 1.0, 10]}
        ),

        "rf": (RandomForestClassifier(random_state=42),
               {"classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20]}),

        "svm": (SVC(probability=True),
                {"classifier__C": [0.1, 1, 10],
                 "classifier__kernel": ["rbf", "linear"]}),
        #included XGBoost due to imbalanced class
        "xgb": (
            XGBClassifier(
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                scale_pos_weight=None  # auto-balances with 'balanced' weights
            ),
            {
                "classifier__n_estimators": [100, 200], 
                "classifier__max_depth": [3, 6, 10],
                "classifier__learning_rate": [0.01, 0.1, 0.2]
            }
        ),
        # "catboost": (
        #     CatBoostClassifier(
        #         loss_function="MultiClass",
        #         random_state=42,
        #         verbose=0
        #     ),
        #     {"classifier__iterations": [200, 500],
        #     "classifier__depth": [4, 6, 8],
        #     "classifier__learning_rate": [0.01, 0.1]}
        # ),

    }
    return models


# def train_and_evaluate_all(df):
#     """Train all models with GridSearchCV and evaluate"""
#     models = get_models()
#     results = {}
#     X, y, preprocessor, _, _, le = preprocess(df)
#     for model_key, (base_model, param_grid) in models.items():
#         print(f"\n=== Training {model_key} ===")
#         clf = Pipeline(steps=[
#             ("cleaner", DataCleaner()),        # <- clean first
#             ("feature_engineer", FeatureEngineer()),   # add this first
#             ("preprocessor", preprocessor),
#             ("classifier", base_model)
#         ])

#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         grid = GridSearchCV(
#             clf,
#             param_grid,
#             cv=cv,
#             scoring="f1_macro",
#             n_jobs=-1
#         )

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, stratify=y, random_state=42
#         )

#         grid.fit(X_train, y_train)

#         # get fitted preprocessor
#         engineered = grid.best_estimator_.named_steps["feature_engineer"].transform(X_train)
#         preprocessor_fitted = grid.best_estimator_.named_steps["preprocessor"]
#         print(engineered.columns)
#         feature_names = preprocessor_fitted.get_feature_names_out(input_features=engineered.columns)
#         numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
#         categorical_cols = X.select_dtypes(include=["object"]).columns

#         print(f"Best Params for {model_key}: {grid.best_params_}")

#         y_pred = grid.predict(X_test)
#         labels = np.arange(len(le.classes_))
#         print("\nClassification Report:")
#         print(classification_report(
#             y_test, y_pred,
#             labels=labels,
#             target_names=le.classes_,
#             zero_division=0
#         ))

#         # Random Forest feature importances
#         if hasattr(grid.best_estimator_["classifier"], "feature_importances_"):
#             importances = grid.best_estimator_["classifier"].feature_importances_
#             # indices = np.argsort(importances)[-20:][::-1]
#             # print("\nTop 20 Feature Importances:")
#             # for idx in indices:
#             #     print(f"{feature_names[idx]}: {importances[idx]:.4f}")
#             sorted_idx = np.argsort(importances)[::-1]
#             for idx in sorted_idx:
#                 print(f"{feature_names[idx]}: {importances[idx]:.4f}")

#         # Logistic Regression or linear SVM coefficients
#         elif hasattr(grid.best_estimator_["classifier"], "coef_"):
#             coefs = grid.best_estimator_["classifier"].coef_[0]
#             # indices = np.argsort(abs(coefs))[-20:][::-1]
#             # print("\nTop 20 Coefficients:")
#             # for idx in indices:
#             #     print(f"{feature_names[idx]}: {coefs[idx]:.4f}")
#             sorted_idx = np.argsort(importances)[::-1]

#             for idx in sorted_idx:
#                 print(f"{feature_names[idx]}: {importances[idx]:.4f}")

#         results[model_key] = grid

#     return results
def train_and_evaluate_all(df):
    """Train all models with GridSearchCV and evaluate"""
    models = get_models()
    results = {}

    # Base X, y
    X = df.drop(columns=["Activity Level", "Session ID"])
    
    le = LabelEncoder()
    y = le.fit_transform(df["Activity Level"])

    for model_key, (base_model, param_grid) in models.items():
        # --- choose preprocessor ---
        if model_key in ["logreg", "svm"]:
            preprocessor = preprocessing_for_linear()
        elif model_key in ["rf", "xgb"]:
            preprocessor = preprocessing_for_tree()
        elif model_key == "catboost":
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
            cat_features_idx = [X.columns.get_loc(c) for c in categorical_cols]

            # no sklearn preprocessor for CatBoost
            preprocessor = None              # CatBoost handles categories internally
        else:
            preprocessor = preprocessing_for_linear()  # fallback

        print(f"\n=== Training {model_key} ===")

        # --- build pipeline ---
        steps = [
            ("cleaner", DataCleaner()),
            ("feature_engineer", FeatureEngineer())
        ]
        if preprocessor is not None:
            steps.append(("preprocessor", preprocessor))
        steps.append(("classifier", base_model))

        clf = Pipeline(steps=steps)

        # --- grid search ---
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(clf, param_grid, cv=cv,
                            scoring="f1_macro", n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        fit_params = {}
        if model_key == "catboost":
            fit_params = {"classifier__cat_features": cat_features_idx}  
        if model_key == "xgb":
            sample_weights = compute_sample_weight("balanced", y_train)
            fit_params = {"classifier__sample_weight": sample_weights}
            grid.fit(X_train, y_train, **fit_params)
        else:
            grid.fit(X_train, y_train)                  
        grid.fit(X_train, y_train, **fit_params)
        # --- feature names (only if preprocessor exists) ---
        feature_names = None
        if preprocessor is not None:
            preprocessor_fitted = grid.best_estimator_.named_steps["preprocessor"]
            try:
                feature_names = preprocessor_fitted.get_feature_names_out()
            except Exception:
                feature_names = [
                    f"f_{i}" for i in range(
                        preprocessor_fitted.transform(
                            grid.best_estimator_.named_steps["feature_engineer"].transform(X_train)
                        ).shape[1]
                    )
                ]

        # --- evaluation ---
        print(f"Best Params for {model_key}: {grid.best_params_}")
        y_pred = grid.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(grid.best_estimator_["classifier"], "predict_proba"):
            y_pred_proba = grid.best_estimator_["classifier"].predict_proba(
                grid.best_estimator_.named_steps["preprocessor"].transform(
                    grid.best_estimator_.named_steps["feature_engineer"].transform(X_test)
                )
            )

        # Comprehensive evaluation
        eval_results = evaluate_model(y_test, y_pred, y_pred_proba, le.classes_)
        print_evaluation_summary(eval_results, le.classes_)
        
        # Plot confusion matrix
        plot_confusion_matrix(eval_results['confusion_matrix'], le.classes_, 
                            f"{model_key.upper()} Confusion Matrix")

        # --- feature importance / coefficients ---
        if feature_names is not None:
            if hasattr(grid.best_estimator_["classifier"], "feature_importances_"):
                importances = grid.best_estimator_["classifier"].feature_importances_
                print("\n=== Feature Importances ===")
                plot_feature_importance(importances, feature_names, top_n=20)

            elif hasattr(grid.best_estimator_["classifier"], "coef_"):
                coefs = grid.best_estimator_["classifier"].coef_
                n_classes, n_features = coefs.shape if coefs.ndim == 2 else (1, coefs.shape[0])

                print("\n=== Coefficients per class ===")
                for class_idx in range(n_classes):
                    class_coefs = coefs[class_idx]
                    sorted_idx = np.argsort(np.abs(class_coefs))[::-1]
                    print(f"\nClass {class_idx} ({le.classes_[class_idx]}):")
                    for idx in sorted_idx[:10]:  # Show top 10 coefficients
                        print(f"{feature_names[idx]}: {class_coefs[idx]:.4f}")

        results[model_key] = grid

    return results

def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    """Get feature names from column transformer"""
    output_features = []

    # numeric columns stay as-is
    output_features.extend(numeric_cols)

    # categorical columns expand after one-hot  ing
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
    output_features.extend(cat_features)

    return output_features



def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--db", type=str, default="data/gas_monitoring.db", help="Path to SQLite DB")
    parser.add_argument("--model", type=str, default="all", help="Model: logreg | rf | svm | xgb | all")
    
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_data(args.db)
    print(f"Dataset shape: {df.shape}")
    print(f"Activity Level distribution:\n{df['Activity Level'].value_counts()}")

    if args.model == "all":
        train_and_evaluate_all(df)
    else:
        print(f"Training {args.model} model...")
        # For single model training, we can implement this later if needed
        train_and_evaluate_all(df)


if __name__ == "__main__":
    main()
