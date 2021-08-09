import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('Fish.csv')

df.rename(columns= {'Length1':'LengthVer', 'Length2':'LengthDia', 'Length3':'LengthCro'}, inplace=True)
#Ver vertical dia.Diagonal, cro nao sei so puis pq n fazia idea 
df.corr()


y = df['Weight']

X = df.iloc[:,2:7]
#Veja que aqui tem um monte de varaiaveis


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=1)


reg = LinearRegression()
reg.fit(X_treino,y_treino)

print('Model intercept: ', reg.intercept_)
print('Model coefficientes: ', reg.coef_)

y_pred= reg.predict(X_treino)
print(r2_score(y_treino, y_cabeca))