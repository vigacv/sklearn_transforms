from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.scaler = RobustScaler()
        self.columns = columns
        self.si = SimpleImputer(missing_values=np.nan,strategy='mean', fill_value=0, verbose=0, copy=True)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        data_dp = data.drop(labels=self.columns, axis='columns')
        self.scaler.fit(data_dp[data_dp.columns[:12]])
        self.si.fit(data_dp[data_dp.columns[:12]])
        data_si = pd.DataFrame.from_records(data=self.si.transform(X=data_dp[data_dp.columns[:12]]),columns=data_dp.columns[:12])
        data_sc = pd.DataFrame.from_records(data=self.scaler.transform(X=data_si[data_si.columns[:12]]),columns=data_si.columns[:12])
        data_sc = data_sc.join(data_dp['PROFILE'])
        return data_sc
