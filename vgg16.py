# -*- coding: utf-8 -*- 

"""
Deep Learning Türkiye topluluğu için Mert Çobanoğlu tarafından hazırlanmıştır.
Amaç: Keras ile nesne tanıma.
Algoritma: Evrişimli Sinir Ağları (Convolutional Neural Networks)
Ek: Çalışma ile ilgili rehber README.md dosyasında belirtilmiştir.
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

img_path = 'images/bird.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])