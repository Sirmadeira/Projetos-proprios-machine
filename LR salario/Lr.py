import sklearn as sk
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df=pd.read_csv('Salary.csv')
df.fillna(-9999,inplace=True)
#print(df.head)


X=np.array(df[["YearsExperience"]])

y=np.array(df[['Salary']])

X_treino, X_teste,y_treino,y_teste=train_test_split(X,y,test_size=0.2)

regressao=linear_model.LinearRegression(n_jobs=-1)

regressao.fit(X_treino,y_treino)

precisao=regressao.score(X_teste,y_teste)
print(precisao)

anos_de_experiencia=5
predicao=regressao.predict([[anos_de_experiencia]])


print(f'Uma pessoa com 5 anos de experiencia vai ter um salario de {predicao[0][0]}')


plt.scatter(df.YearsExperience,df.Salary,color='BLUE')
plt.xlabel('Anos de experiencia')
plt.ylabel('Salario')
plt.show()