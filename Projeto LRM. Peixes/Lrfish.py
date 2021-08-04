import sklearn as sk
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df=pd.read_csv('Fish.csv')

df.fillna(-9999,inplace=True)

X=np.array(df[["Width"]])

y=np.array(df[["Weight"]])

X_treino, X_teste,y_treino,y_teste=train_test_split(X,y,test_size=0.2)

clf=linear_model.LinearRegression(n_jobs=-1)


clf.fit(X_treino,y_treino)

precisao=clf.score(X_teste,y_teste)

print(precisao)

largura=5
predicao=clf.predict([[largura]])

print(f'Um peixe com uma largura {largura} vai pesar  {predicao[0][0]}')

plt.scatter(df.Width,df.Weight,color='BLUE')
plt.xlabel('Largura')
plt.ylabel('Peso')
plt.show()