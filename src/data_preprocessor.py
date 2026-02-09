import pandas as pd
import numpy as np
from typing import  Dict, Any, List


class DataPreprocessor:
    

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be pandas DataFrame")
        self.original_df_ = df.copy()
        self.transformed_df_: pd.DataFrame = None
        self.log_: List[str] = []
        self.dropped_cols_: List[str] = []
        self.ohe_cols_: List[str] = []
        self.impute_strats_: Dict[str, Dict[str, Any]] = {}
        self.scaler_params_: Dict[str, Dict[str, float]] = {}

    def _impute_column(self, series: pd.Series, col: str) -> pd.Series:
        
        if series.dtype.kind in 'biufc':  
            if abs(series.skew()) > 1:  
                val = series.median()
                self.impute_strats_[col] = {'method': 'median', 'value': float(val)}
            else:
                val = series.mean()
                self.impute_strats_[col] = {'method': 'mean', 'value': float(val)}
        else:  
            modes = series.mode()
            val = modes.iloc[0] if not modes.empty else series.dropna().iloc[0] if len(series.dropna()) > 0 else 'MISSING'
            self.impute_strats_[col] = {'method': 'mode', 'value': str(val)}
        return series.fillna(val)



    def remove_missing(self, threshold: float = 0.5) -> pd.DataFrame:
        
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        df = self.original_df_.copy() if self.transformed_df_ is None else self.transformed_df_.copy()

        miss_pct = df.isnull().mean()
        to_drop = miss_pct[miss_pct > threshold].index.tolist()
        self.dropped_cols_ = to_drop
        self.log_.append(f"Dropped cols (> {threshold*100}% NaN): {to_drop}")

        df = df.drop(columns=to_drop)
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = self._impute_column(df[col], col)
        self.transformed_df_ = df
        return df

    def encode_categorical(self) -> pd.DataFrame:
        
        df = self.transformed_df_.copy()
        cat_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(cat_cols) == 0:
            self.log_.append("No categorical cols found")
            return df

        df_encoded = pd.get_dummies(df, columns=cat_cols, prefix_sep='_', dtype=float)
        self.ohe_cols_ = [col for col in df_encoded.columns if any(col.endswith(f'_{c}') for c in cat_cols)]
        self.log_.append(f"OHE cols added: {len(self.ohe_cols_)}")
        self.transformed_df_ = df_encoded
        return df_encoded

    def normalize_numeric(self, method: str = 'minmax') -> pd.DataFrame:
        
        methods = {'minmax', 'std'}
        if method not in methods:
            raise ValueError(f"method must be one of {methods}")
        df = self.transformed_df_.copy()
        num_cols = df.select_dtypes(include=['number']).columns

        self.scaler_params_ = {}
        for col in num_cols:
            if method == 'minmax':
                mn, mx = df[col].min(), df[col].max()
                df[col] = (df[col] - mn) / (mx - mn) if mx != mn else 0
                self.scaler_params_[col] = {'method': 'minmax', 'min': mn, 'max': mx}
            else:  
                mn, st = df[col].mean(), df[col].std()
                df[col] = (df[col] - mn) / st if st != 0 else 0
                self.scaler_params_[col] = {'method': 'std', 'mean': mn, 'std': st}

        self.log_.append(f"Normalized {len(num_cols)} numeric cols with '{method}'")
        self.transformed_df_ = df
        return df

    def fit_transform(self, threshold: float = 0.5, method: str = 'minmax') -> pd.DataFrame:
        
        self.transformed_df_ = None  
        self.log_ = []
        self.remove_missing(threshold)
        self.encode_categorical()
        self.normalize_numeric(method)
        return self.transformed_df_

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if self.transformed_df_ is None:
            raise ValueError("Fit first with fit_transform()")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be DataFrame")

        
        df = df.drop(columns=self.dropped_cols_, errors='ignore')

        
        for col, strat in self.impute_strats_.items():
            if col in df.columns:
                df[col] = df[col].fillna(strat['value'])

        
        cat_cols = [c for c in self.original_df_.select_dtypes(include=['object']).columns if c not in self.dropped_cols_]
        df = pd.get_dummies(df, columns=cat_cols, prefix_sep='_', dtype=float)
        for ohe_col in self.ohe_cols_:
            if ohe_col not in df.columns:
                df[ohe_col] = 0

        
        num_cols = df.select_dtypes(include=['number']).columns.intersection(self.scaler_params_.keys())
        for col, params in self.scaler_params_.items():
            if col in df.columns:
                if params['method'] == 'minmax':
                    mn, mx = params['min'], params['max']
                    df[col] = np.clip((df[col] - mn) / (mx - mn) if mx != mn else 0, 0, 1)
                else:
                    mn, st = params['mean'], params['std']
                    df[col] = (df[col] - mn) / st if st != 0 else 0

        
        df = df.reindex(columns=self.transformed_df_.columns, fill_value=0)
        return df

    def get_log(self) -> List[str]:
        
        return self.log_[:]
