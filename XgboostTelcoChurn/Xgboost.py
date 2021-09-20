#Esse projeto, e um projeto de customer retainment.
#Basicamente, uma tentativa de manter os clientes
#Ao analisar as variaveis, verificaremos se aquele cliente
#Vai parar de se utilizar dos servicos da telco.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix #Criar matriz de confusao
from sklearn.metrics import plot_confusion_matrix# Desenha

df=pd.read_csv('data/Telco-Customer-Churn.csv')

print(df.head())