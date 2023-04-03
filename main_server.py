import os
import numpy as np  # linear algebra
from keras.models import load_model
import shutil
import face_recognition
from werkzeug.utils import secure_filename
from flask import Flask,  jsonify, render_template,  send_file, flash, request, redirect, url_for, session, send_from_directory
from flask_cors import CORS, cross_origin
import base64
import json
from PIL import Image
import io
import cv2

UPLOAD_FOLDER = 'static/images'
INPUT_TRAIN_FOLDER = 'input/data/train'
INPUT_TEST_FOLDER = 'input/data/test'
TARGET_FOLDER = 'target/identify'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, template_folder='templates')
cors = CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INPUT_TRAIN_FOLDER'] = INPUT_TRAIN_FOLDER
app.config['TARGET_FOLDER'] = TARGET_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'geocontrol'
app.config["CACHE_TYPE"] = "null"
# app.config['CORS_HEADERS'] = 'Content-Type'


def identify_face():
    # load the face dataset
    data = np.load('geocontrol-embeddings.npz')
    emdTrainX, trainy, emdTestX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    model, in_encoder, out_encoder = face_recognition.create_model(
        emdTrainX, trainy, emdTestX, testy)
    texto, name = face_recognition.identify_new_face(
        model, in_encoder, out_encoder)
    return texto, name


def clean_folder(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode(
        'ascii')  # encode as base64
    return encoded_img


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/testes')
def testes():
    ok = face_recognition.apply_gaussian_blur('./input/data/train')

    return jsonify(ok)


@app.route('/adicionar',  methods=['GET', 'POST'])
def adicionar():

    if (len(os.listdir(UPLOAD_FOLDER)) > 1):
        clean_folder(UPLOAD_FOLDER + '/')
    if request.method == "POST":
        # check if the post request has the file part
        if 'label' in request.form:
            label = request.form['label']
        else:
            label = 'pessoa_sem_nome'

        if 'trainimages' in request.files:
            trainimages = request.files.getlist("trainimages")
            # cria diretorio com o nome da label fornecida
            if not os.path.exists(os.path.join(INPUT_TRAIN_FOLDER, label)):
                os.mkdir(os.path.join(INPUT_TRAIN_FOLDER, label))

            count = 0
            # salva imagens de treino e copia para pasta de treino do dataset
            for file in trainimages:
                count += 1
                howmany = len(trainimages)
                file.save(os.path.join(
                    INPUT_TRAIN_FOLDER, label, 'image-' + {howmany+count} + '.png'))
            train_image_names = os.listdir(
                os.path.join(INPUT_TRAIN_FOLDER, label))

        if 'testimages' in request.files:
            testimages = request.files.getlist("testimages")

            # cria diretorio com o nome da label fornecida
            if not os.path.exists(os.path.join(INPUT_TEST_FOLDER, label)):
                os.mkdir(os.path.join(INPUT_TEST_FOLDER, label))

            # salva imagens de treino e copia para pasta de treino do dataset
            for file in testimages:
                file.save(os.path.join(
                    INPUT_TEST_FOLDER, label, file.filename))
            test_image_names = os.listdir(
                os.path.join(INPUT_TEST_FOLDER, label))

            try:
                face_recognition.train_dataset()
                return jsonify("%s foi adicionada(o) à base de dados" % (label.capitalize()), '')
            except (IndexError) as error:
                shutil.rmtree(os.path.join(INPUT_TEST_FOLDER, label))
                shutil.rmtree(os.path.join(INPUT_TRAIN_FOLDER, label))
                return jsonify(error.args)


@app.route('/identificar', methods=['GET', 'POST'])
def identificar():
    if (len(os.listdir(TARGET_FOLDER)) > 1):
        clean_folder(TARGET_FOLDER + '/')
    if (len(os.listdir(UPLOAD_FOLDER)) > 1):
        clean_folder(UPLOAD_FOLDER + '/')
    if request.method == "POST":
        if request.files['image']:
            name_file = "identify.png"
            file = request.files['image']
            file.save(os.path.join(UPLOAD_FOLDER, name_file))
            shutil.copy2(os.path.join(UPLOAD_FOLDER, name_file),
                         os.path.join(TARGET_FOLDER, name_file))
            if (os.path.isfile(os.path.join(UPLOAD_FOLDER, name_file))):
                if not (os.path.isfile('./geocontrol-embeddings.npz')):
                    model, in_encode_, out_encode = face_recognition.train_dataset()

                try:
                    texto, name = identify_face()

                # with open('input/data/train/melina/melina.jpeg', "rb") as img:
                #     stringImage = base64.b64encode(
                #         img.read()).decode('utf-8')
                #     response = requests.post(
                #         url='localhost:8080/consulta/pessoa/reconhecimento', json={'user_photo': stringImage})

                # return send_from_directory('input/data/train/melina',
                #                            'melina.jpeg')
                    return jsonify(texto, name)
                except (IndexError) as error:
                    return jsonify(error.args[0], '')

            else:
                return jsonify('Se nao tem arquivo', '')

        else:
            return jsonify('Não foi recebida uma imagem válida', '')


@ app.route('/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(UPLOAD_FOLDER, filename)


app.run(host='0.0.0.0', port='5001')
