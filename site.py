from flask import Flask, render_template

import numpy as np
import PIL
from PIL import Image
from keras import models
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
import efficientnet.keras as ef
from efficientnet.tfkeras import EfficientNetB7
from flask import request
import torch


app = Flask(__name__, template_folder='templates',static_folder='CSSestart')

@app.route('/')
def index():
    return render_template('homepage.HTML')

@app.route('/diagnosticoFinal',methods=['POST'])
def teste():
    img = Image.open(r'C:\Users\gaby\Documents\PROJETOS GABY\melanoma_cancer_dataset-20220802T172035Z-001\melanoma_cancer_dataset-20220802T172035Z-001\melanoma_cancer_dataset\test\malignant\melanoma_10112.jpg')
    img_np = np.array(img,'uint8')
    img_np = np.expand_dims(img_np,axis=0)
    print(img_np.shape)

    modelo = tf.keras.models.load_model(r'C:\Users\gaby\Documents\PROJETOS GABY\chest_orientation_model.hdf5')

    previsor = modelo.predict(img_np)
    print(previsor)

    prev = torch.tensor(previsor)
    novoPrev = torch.rand_like(prev, dtype=torch.float)
    novoPrev = np.array(novoPrev)
    print(novoPrev)

    mal=novoPrev[0][1]
    ben = novoPrev[0][0]

    pred_class = np.argmax(novoPrev)
    if pred_class == 0: 
        ben='BENIGNO'
    elif pred_class == 1:
        mal=='MALIGNO'
    porce_ben = ben*100
    porce_mal = mal*100


    return render_template('homepage.HTML', msg1=int(porce_ben), msg2=int(porce_mal))
 
@app.route('/mel.html')
def mel():
    return render_template('mel.html')
  
@app.route('/gab.html')
def gab():
    return render_template('gab.html')
 

app.run(debug=True)
