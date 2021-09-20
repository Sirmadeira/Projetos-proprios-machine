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

df=pd.read_csv('/parkinsons.csv')





