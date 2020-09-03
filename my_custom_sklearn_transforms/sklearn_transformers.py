from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data.drop(labels=self.columns, axis='columns')

class Imputation():
    def __init__(self):
        self.si = SimpleImputer(missing_values=np.nan,strategy='mean', fill_value=0, verbose=0, copy=True)
    def fit(self, X, y=None):
        data = X.copy()
        self.si.fit(data[data.columns[:12]])
        return self
    def transform(self, X):
        data = X.copy()
        target = data[data.columns[len(data.columns)-1]]
        data_si = pd.DataFrame.from_records(data=self.si.transform(X=data[data.columns[:12]]),columns=data.columns[:12])
        return data_si.join(target)
    
class Normalization():
    def __init__(self):
        self.scaler = RobustScaler()
    def fit(self, X, y=None):
        data = X.copy()
        self.scaler.fit(data[data.columns[:12]])
        return self
    def transform(self, X):
        data = X.copy()
        target = data[data.columns[len(data.columns)-1]]
        data_sc = pd.DataFrame.from_records(data=self.scaler.transform(X=data[data.columns[:12]]),columns=data.columns[:12])
        return data_sc.join(target)
