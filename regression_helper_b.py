import pandas as pd
from sklearn.linear_model import LinearRegression

def make_regression(file):
    df = pd.read_csv(file)
    lr = LinearRegression()
    lr.fit(df['Peso'].values.reshape(-1, 1), df['Altura'].values.reshape(-1, 1))
    return lr.coef_[[0]], lr.intercept_[0]
    