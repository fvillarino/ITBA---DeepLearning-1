import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def make_regression(file):
    df = pd.read_csv(file)
    lr = LinearRegression()
    lr.fit(df['Peso'].values.reshape(-1, 1), df['Altura'].values.reshape(-1, 1))
    return lr.coef_[0], lr.intercept_

if __name__ == '__main__':
    print(make_regression('alturas-pesos.csv'))
