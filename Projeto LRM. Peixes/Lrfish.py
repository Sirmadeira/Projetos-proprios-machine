import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df=pd.read_csv('Fish.csv')

print(str('Vendo se tem valores nada ver no dataset: '), df.isnull().values.any())

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