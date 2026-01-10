# model_runtime.py
import json
import xgboost as xgb
import pandas as pd

class ModelRuntime:
    def __init__(self, model_path: str, cols_path: str):
        self.model_path = model_path
        self.cols_path = cols_path
        self.bst = None
        self.cols = None

    def load(self):
        if self.bst is None:
            self.bst = xgb.Booster()
            self.bst.load_model(self.model_path)
        if self.cols is None:
            with open(self.cols_path, "r") as f:
                self.cols = json.load(f)

    def predict_proba(self, X: pd.DataFrame) -> float:
        self.load()
        X2 = X.reindex(columns=self.cols, fill_value=0)
        dmat = xgb.DMatrix(X2, feature_names=self.cols)
        return float(self.bst.predict(dmat)[0])
