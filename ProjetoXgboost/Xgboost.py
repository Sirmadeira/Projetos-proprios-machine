#O dataset tera como enfoco principal, identificar 
#Os atributos que contribuem  para a doenca de parkinson, algumas dessa variaveis
#Obtive, discutindo com a namorado do meu irmao que e medica
#Agradeco Hannah pelas infos do que voces procuram e do que geralmente originam
#O sofrimento, vlw flw
#Se voce quiser os detalhes, do dataset so acessar a via parkinsons.names
#La vai ter uma explicacao em ingles de cada variavel
#Trabalhei por conda, entao nao tem como compartilhas o environment, sorry :<

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Realmente bugo aqui tive que usar absulute nao faco ideia pq
df=pd.read_csv('C:/Users/FranciscoFroes/Documents/GitHub/Projetos-proprios-machine/ProjetoXgboost/parkinsons.csv')

#print(df.head())

features=df.loc[:,df.columns!='status'].values[:,1:]
#print(features)
#Features=Todas as colunas menos status
labels=df.loc[:,'status'].values
#Label= Status, se tem ou nao tem

print(labels[labels==1].shape[0],labels[labels==0].shape[0])
#Numero de status positivos numero de status negativos

escalanador=MinMaxScaler((-1,1))
#Isso daki vai definir o valor em que se encontra as features,basicamente
#Dividimo e arrendomos para o valor mais proximo
X=escalanador.fit_transform(features)
#Transformando
y=labels


X_treino,X_teste,y_treino,y_teste=train_test_split(X,y,test_size=0.2,random_state=7,)
#Random state e o quanto a gente vai ficar trocando randomicamente os dados

modelo=XGBClassifier(use_label_encoder=False,eval_metric='mlogloss',reg_lambda=2,)
modelo.fit(X_treino,y_treino)

y_pred=modelo.predict(X_teste)
print(accuracy_score(y_teste,y_pred)*100)





