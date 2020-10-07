import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def get_linear_regression(filename, plot_data=False):
    df = pd.read_csv(filename)
    lr = LinearRegression()
    lr.fit(df['Peso'].values.reshape(-1,1),df['Altura'].values.reshape(-1,1))
    if plot_data:
        plt.figure(figsize=(20,10))
        x = np.linspace(20,130,100)
        y = lr.predict(x.reshape(-1,1)) # Predice el valor
        plt.scatter(df['Peso'].values,df['Altura'].values)
        plt.plot(x,y,c='y')
    return lr.coef_[0][0],lr.intercept_[0]

if __name__ == '__main__':
    print(get_linear_regression('data/alturas-pesos.csv',True))