# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import requests

from io import BytesIO
from flask import Flask
from flask import request
from flask_restful import Api
from flask_restful import Resource

import librosa
import numpy as np
import noisereduce as nr
import tensorflow as tf
from tensorflow import keras

def fix_audio(data, input_length=64000):
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset : (input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "wrap")
    return data

class Predict(Resource):
    def __init__(self):
        input_wave = keras.layers.Input(shape=(40,126,1))
        vgg = keras.applications.VGG16(input_shape=(40,126,1), weights=None, include_top=False)
        x = vgg(input_wave)
        x = keras.layers.GlobalMaxPool2D()(x)
        x = keras.layers.Dropout(rate=0.3)(x)

        x = keras.layers.Dense(250, activation='relu')(x)
        x = keras.layers.Dense(250, activation='relu')(x)
        x = keras.layers.Dense(10, activation='softmax')(x)
        self.model = keras.models.Model(inputs=input_wave, outputs=x)
        
        self.model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
        self.model.load_weights('/src/model.h5')
        #self.model.load_weights('model.h5')

        self.labels = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

    def post(self):
        payload = request.get_json()
        audio = np.array(payload['audio'])/32768
        noise = np.array(payload['noise'])/32768
        sample_rate = payload['sample_rate']
        #url = request.args.get('url')
        #wav = requests.get(url).content
        #wav = request.data
        #audio,sample_rate = librosa.load(BytesIO(wav),sr=16000,res_type='kaiser_fast')

        audio = nr.reduce_noise(audio_clip=audio, noise_clip=noise, verbose=False)
        audio=fix_audio(audio)
        #audio_tensor=librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        audio_tensor=librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40),ref=np.max)
        batch = np.reshape(audio_tensor,(1,40,126,1))

        out = self.model.predict(batch)

        index = keras.backend.argmax(out[0]).numpy()
        #pct = out * 100
        #return [out.tolist(),str(index)]
        #return str([out, index, pct])
        return self.labels[index]
        #return 'success'


app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
