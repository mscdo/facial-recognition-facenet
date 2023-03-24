import mtcnn
print(mtcnn.__version__)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image
import shutil

import os
import face_recognition

from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template,  flash, request, redirect, url_for, session, send_from_directory

UPLOAD_FOLDER = 'static/images'
INPUT_TRAIN_FOLDER = 'input/data/train'
INPUT_TEST_FOLDER = 'input/data/test'
TARGET_FOLDER = 'target/identify'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    # load train dataset
    trainX, trainy = face_recognition.load_dataset('./input/data/train/')
    print(trainX.shape, trainy.shape)
    # load test dataset
    testX, testy = face_recognition.load_dataset('./input/data/test/')
    print(testX.shape, testy.shape)

    # save and compress the dataset for further use
    np.savez_compressed('geocontrol.npz', trainX, trainy, testX, testy)

    # load the face dataset
    data = np.load('geocontrol.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    # load the facenet model
    facenet_model = load_model('./models/facenet_keras.h5')
    print('Loaded Model')
       # convert each face in the train set into embedding
    emdTrainX = list()
    for face in trainX:
        emd = face_recognition.get_embedding(facenet_model, face)
        emdTrainX.append(emd)       
    emdTrainX = np.asarray(emdTrainX)
    print('EMDTRAINX ************************************', emdTrainX.shape)
    # convert each face in the test set into embedding
    emdTestX = list()
    for face in testX:
        emd = face_recognition.get_embedding(facenet_model, face)
        emdTestX.append(emd)
        
    emdTestX = np.asarray(emdTestX)
    print('EMDTESTX ************************************', emdTestX.shape)
    # save arrays to one file in compressed format
    np.savez_compressed('geocontrol-embeddings.npz', emdTrainX, trainy, emdTestX, testy)
    
    model, in_encoder, out_encoder = face_recognition.create_model(emdTrainX, trainy, emdTestX, testy)
    texto = face_recognition.identify_new_face(model, in_encoder, out_encoder)
    return texto
    # plt.imshow(photo)
    # plt.title(id)
    # plt.show()


def identify_face():
     # load the face dataset
    data = np.load('geocontrol-embeddings.npz')
    emdTrainX, trainy, emdTestX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    model, in_encoder, out_encoder = face_recognition.create_model(emdTrainX, trainy, emdTestX, testy)
    texto, name = face_recognition.identify_new_face(model, in_encoder, out_encoder)
    return texto,name 

def clean_folder(folder:str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

app= Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INPUT_TRAIN_FOLDER'] = INPUT_TRAIN_FOLDER
app.config['TARGET_FOLDER'] = TARGET_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 
app.secret_key = 'geocontrol'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/')
def index():
    return 'INdex'



@app.route('/adicionar',  methods=['GET', 'POST'])
def adicionar():
    return 'Oi'

@app.route('/identificar', methods=['GET', 'POST'])
def identificar():
    filename =''
    predicted_image=''
    image=''
    
    if request.method == "POST":
        # check if the post request has the file part
        
        if request.files: 
            image = request.files["image"]           
            # print(image + "Uploaded to Faces")
            # flash('Image successfully Uploaded to Faces.')
            clean_folder(UPLOAD_FOLDER+'/')
            clean_folder(TARGET_FOLDER+'/')
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
            shutil.copy2(UPLOAD_FOLDER + '/' + image.filename, TARGET_FOLDER +  '/' + image.filename )
            filename = os.path.join(UPLOAD_FOLDER, image.filename)
            print("stored as:" + filename)
            texto, name = identify_face()
            if name =='': 
                return render_template("identificar.html", uploaded_image=filename, predicted_image='', texto=texto)
            predicted_image_name = os.listdir(os.path.join( INPUT_TRAIN_FOLDER, name.strip()) + '/')[0]
            predicted_image =  os.path.join( INPUT_TRAIN_FOLDER, name.strip() + '/', os.listdir(os.path.join( INPUT_TRAIN_FOLDER, name.strip()) + '/')[0])
            shutil.copy2( predicted_image, os.path.join( UPLOAD_FOLDER, predicted_image_name))
            predicted_image = os.path.join( UPLOAD_FOLDER, predicted_image_name)
            return render_template("identificar.html", uploaded_image=filename, predicted_image=predicted_image, texto=texto)
        else:
            print('Nenhuma imagem selecionada'),
            return redirect(request.url)
    return render_template("identificar.html")


@app.route('/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(UPLOAD_FOLDER, filename)


# @app.after_request
# def add_header(r):
#     """
#     Add headers to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     """
#     r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     r.headers["Pragma"] = "no-cache"
#     r.headers["Expires"] = "0"
#     r.headers['Cache-Control'] = 'public, max-age=0'
#     return r

app.run(host='0.0.0.0')
