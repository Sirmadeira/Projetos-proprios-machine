# import shutil 
# import os
# import random
# import tensorflow as tf
# #Projeto onde eu tento construir uma rede neural  que identifica tipos de cancer se sao malignos ou nao

# seed = 1
# random.seed(seed)
# diretorio="datasets/data/imagens/"
# treino="data/treino/"
# teste='data/teste/' 
# validacao='data/validacao/'


# # os.makedirs(teste + "benigno/")
# # os.makedirs(teste + "maligno/")
# # os.makedirs(validacao + "benigno/")
# # os.makedirs(validacao + "maligno/")
# #Criando diretorios

# teste_exemplos=treino_exemplos=validacao_exemplos=0
# #Vamos contar quantos exemplos a gente vai por em cada folder

# for linha in open("data/datasets/labels.csv").readlines()[1:]:
# #Lendo somente as linha depois dos titulos
# 	linhe_div=linha.split(',')
# 	img_pasta=linhe_div[0]
# 	benigno_maligno=linhe_div[1]
# 	#Pegando coluna que diferencia se e maligno ou benigno

# 	numero_random=random.random()
# 	#Numero entre 0 e 1

# 	if numero_random <0.8:
# 		locacao=treino
# 		treino_exemplos += 1
# 	elif numero_random < 0.9:
# 		locacao=validacao
# 		validacao_exemplos+=1
# 	else:
# 		locacao=teste
# 		teste_exemplos+= 1
# 	#Pegando cada dataset e direcionado as imagens de acordo com o valor nesse caso a maioria vai para o treino obviamente

# 	if int(float(benigno_maligno))== 0:
# 		shutil.copy(
# 			"data/datasets/imagens/" + img_pasta +'.jpg',
# 			locacao + "benigno/" + img_pasta +'.jpg',
# 		)
# 		#Transferindo para a pasta correta
# 	elif int(float(benigno_maligno)) == 1:
# 		shutil.copy(
# 			"data/datasets/imagens/" + img_pasta +'.jpg',
# 			locacao + "maligno/" + img_pasta +'.jpg',
# 		)
# 		#Mesma coisa para maligno


# print(f'Numero de exemplos de treino exemplos {treino_exemplos}')

# print(f'Numero de exemplos de teste exemplos{teste_exemplos}')

# print(f'Numero de exemplos de validacao exemplos {validacao_exemplos}')

#Esse script foi utilizado para rodar e por os datasets nos locais corretos vamos partir para o trabalho

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

treino_exemplos =20225
teste_exemplos=2555
validacao_exemplos=2551

altura_imagen = largura_imagen= 224

batch_size = 32

#NasNet- Metodo de pre treinamento e modelo vagabundinho
# modelo = keras.Sequential([
#    hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
#                   trainable=True),
#    layers.Dense(1, activation="sigmoid"),
# ])

modelo=keras.models.load_model("pretreinado/")

treino_datager= ImageDataGenerator(
	rescale=1.0/255,
	rotation_range=15,
	zoom_range=(0.95,0.95),
	horizontal_flip=True,
	vertical_flip=True,
	data_format="channels_last",
	dtype=tf.float32)
#Augmentacao de data

validacao_datager=ImageDataGenerator(rescale=1.00/255,dtype=tf.float32)

teste_datager=ImageDataGenerator(rescale=1/255,dtype=tf.float32)
#Jeito mais conveniente de carregar dataset, outros metodos de carregamento podem ser mais eficiente

treino_gerador=treino_datager.flow_from_directory(
	"data/treino/",
	target_size=(altura_imagen,largura_imagen),
	batch_size=batch_size,
	color_mode="rgb",
	class_mode="binary",
	shuffle=True,
	seed=123)

validacao_gerador=validacao_datager.flow_from_directory(
	"data/validacao/",
	target_size=(altura_imagen,largura_imagen),
	batch_size=batch_size,
	color_mode="rgb",
	class_mode="binary",
	shuffle=True,
	seed=123)

teste_gerador=teste_datager.flow_from_directory(
	"data/teste/",
	target_size=(altura_imagen,largura_imagen),
	batch_size=batch_size,
	color_mode="rgb",
	class_mode="binary",
	shuffle=True,
	seed=123)
#Definindo oos diretorios de origem, e as infos das imagens

METRICAS=[
	keras.metrics.BinaryAccuracy(name="accuracy"),
	keras.metrics.Precision(name="precision"),
	keras.metrics.Recall(name="recall"),
	keras.metrics.AUC(name="auc"),
]

modelo.compile(
	optimizer=keras.optimizers.Adam(learning_rate=3e-4),
	loss=[keras.losses.BinaryCrossentropy(from_logits=True)],
	metrics=METRICAS)

#Accuracy, e um metodo ruim. Acho... pq a chance de alguem ter cancer maligno e bem baixa
#Acho que vou usar de precision ou recall, para realmente avaliar quais deles realmente tem cancer
#Lembrete
#Precisao TRUE POSITIVE/True POSITIVE + FALSO NEGATIVO
#Recall TRUE POSITIVE/TRUE POSTIVIE+FALSO NEGATIVO

modelo.fit(
	treino_gerador,
	epochs=1,
	steps_per_epoch=treino_exemplos//batch_size,
	validation_data=validacao_gerador,
	validation_steps=validacao_exemplos//batch_size,
	callbacks=[keras.callbacks.ModelCheckpoint('pretreinado')])
#Salvando o modelo por cada epoch

def plot_roc(labels,data):
	predicoes=modelo.predict(data)
	fp,tp, _ =roc_curve(labels,predicoes)
	#True positive false postive, roc curve e uma funcao do scikit ajuda a montar isso dae
	plt.plot(100*fp,100*tp)
	plt.xlabel("Falso positivos [%]")
	plt.ylabel("Verdadeiros positivos [%]")
	plt.show()

teste_labels=np.array([])
numero_batchs=0

for _,y in teste_gerador:
	teste_labels=np.append(teste_labels,y)
	numero_batchs +=1
	if numero_batchs == math.ceil(teste_exemplos/batch_size):
		break

plot_roc(teste_labels,teste_gerador)

modelo.evaluate(validacao_gerador,verbose=2)
modelo.evaluate(teste_gerador,verbose=2)

modelo.save('pretreinado/')