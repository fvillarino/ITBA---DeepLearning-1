import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np

class StandardScaler():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()
        
    def transform(self, X):
        return (X - self.mean)/self.std
    
class Person():
    def __init__(self, fname, lname, age, direccion='No indica'):
        self.fname = fname
        self.lname = lname
        self.age = age
        self.direccion = direccion
        
    def print_name(self):
        print(f'{self.fname} {self.lname} tiene {self.age} años y vive en {self.direccion}')
        
    def __str__(self):
        
        return str({'fname': self.fname, 'lname': self.lname})
    
    def __repr__(self):
        return f'{self.fname} {self.lname} tiene {self.age} años y vive en {self.direccion}'
    
    
class Student(Person):
    def __init__(self, fname, lname, age, legajo, direccion='No indica'):
        #super().__init__(fname, lname, age, direccion='No indica')
        super(fname, lname, age, direccion='No indica')
        self.legajo = legajo
    
    def print_student_name(self):
        print(f'El estudiante {self.fname} {self.lname} tiene legajo {self.legajo}')
    
    

def make_regression(cvs_file):
    df = pd.read_csv(cvs_file)
    lr = LinearRegression()
    lr.fit(df['Peso'].values.reshape(-1, 1), df['Altura'].values.reshape(-1, 1))
    return lr.coef_[0][0], lr.intercept_[0]

def plot_data(cvs_file, coef=None, intercept=None):
    df = pd.read_csv(cvs_file)
    plt.figure(figsize=(20,10))
    plt.scatter(df['Peso'], df['Altura'], s=10, label='puntos')

    if coef is not None:
        x = np.linspace(30, 120, 4)
        y = coef * x + intercept
        plt.plot(x, y, c='y', label='recta')
        plt.title(f'Regresión: y = {coef} x + {intercept}')
        
    plt.title(f'Regresión')    
    plt.xlabel('peso')
    plt.ylabel('altura')
    plt.legend()

if __name__ == '__main__':
    print('Se ejecuto el modulo')
    print(make_regression('data/alturas-pesos.csv'))

if __name__ == 'utils':
    print('Se importo el modulo')