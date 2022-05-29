from distutils.command.sdist import sdist
from django.http import HttpResponse
from django.shortcuts import render
from Django_backend.settings import BASE_DIR
from .forms import ImageForm
from .models import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Create your views here.


model = load_model(os.path.join(BASE_DIR, 'models/model3.h5'))


def home(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return render(request, 'analyse.html')
    form = ImageForm()
    img = Image.objects.all()
    return render(request, 'home.html')


def analyse(request):
    img = Image.objects.all()
    l = len(img)
    x = img[l-1]
    img_path = str(BASE_DIR) + x.photo.url
    img = image.load_img(img_path, target_size=(128, 128, 3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction = prediction.argmax()
    meters = {0: 'images/meter-01.png',
              1: 'images/meter-02.png',
              2: 'images/meter-03.png',
              3: 'images/meter-04.png',
              4: 'images/meter-05.png',
              5: 'images/meter-06.png',
              6: 'images/meter-07.png',
              7: 'images/meter-08.png',
              8: 'images/meter-09.png',
              }
    return render(request, 'result.html', {'prediction': prediction, 'path': meters[prediction]})
