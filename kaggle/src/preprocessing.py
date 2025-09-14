# preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def preprocessing_for_linear():
    """Preprocessor for models that need scaling + OHE"""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_include=object))
    ])

def preprocessing_for_tree():
    """Preprocessor for tree-based models (RF, XGB)"""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_include=object))
    ])

def preprocessing_for_catboost():
    """For CatBoost you don’t preprocess numerics/categoricals separately,
       just handle NaNs, and mark categorical cols when fitting"""
    return None  # CatBoost handles it internally
