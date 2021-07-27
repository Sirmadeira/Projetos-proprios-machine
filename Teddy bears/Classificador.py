# import os
# import random
# import shutil

# imagen_pasta= "C:/Users/FranciscoFroes/Documents/GitHub/Tensorflow/Projetos proprios/Teddy bears/teddys/"
# treino = "C:/Users/FranciscoFroes/Documents/GitHub/Tensorflow/Projetos proprios/Teddy bears/data/treino/teddys/"
# teste = "C:/Users/FranciscoFroes/Documents/GitHub/Tensorflow/Projetos proprios/Teddy bears/data/teste/teddys/"
# validacao = "C:/Users/FranciscoFroes/Documents/GitHub/Tensorflow/Projetos proprios/Teddy bears/data/validacao/teddys/"


# teste_contador=treino_contador=validacao_contador=0


# for nome_da_pasta in os.listdir(imagen_pasta):
# 	numero_random = random.random()
# 	if numero_random < 0.8:
# 		shutil.copy(imagen_pasta+nome_da_pasta,treino)
# 		treino_contador +=1
# 	elif numero_random < 0.9:
# 		shutil.copy(imagen_pasta+nome_da_pasta,teste)
# 		teste_contador+=1
# 	else:
# 		shutil.copy(imagen_pasta+nome_da_pasta,validacao)
# 		validacao_contador+=1


# print(f"Numero de imagens de treino{treino_contador}, numero de imagens de teste {teste_contador}, numero de imagens de validacao {validacao_contador}")
#Jeito utilizado para rachar as imagens do datasetss

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


numero_de_imagens_treino=415
numero_de_imagens_teste=53
numero_de_imagens_validacao=55


altura_imagen=largura_imagen=244

batch_size=32

modelo = keras.Sequential(
    [
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(3,activation="softmax"),
    ]
)


treino_datager= ImageDataGenerator(
	rescale=1.0/255,
	rotation_range=15,
	zoom_range=(0.95,0.95),
	horizontal_flip=True,
	vertical_flip=True,
	data_format="channels_last",
	dtype=tf.float32)

validacao_datager=ImageDataGenerator(rescale=1.00/255,dtype=tf.float32)

teste_datager=ImageDataGenerator(rescale=1/255,dtype=tf.float32)

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

METRICAS=[
	keras.metrics.BinaryAccuracy(name="accuracy"),
	keras.metrics.Precision(name="precision"),
	keras.metrics.Recall(name="recall"),
]

modelo.compile(
	optimizer=keras.optimizers.Adam(learning_rate=3e-4),
	loss=[keras.losses.BinaryCrossentropy()],
	metrics=METRICAS)


modelo.fit(
	treino_gerador,
	epochs=10,
	batch_size=batch_size,
	validation_data=validacao_gerador,
	)

modelo.evaluate(validacao_gerador,verbose=2)
modelo.evaluate(teste_gerador,verbose=2)