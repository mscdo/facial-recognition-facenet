import mtcnn
# print(mtcnn.__version__)
import os
import numpy as np  # linear algebra
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template,  flash, request, redirect, url_for, session, send_from_directory

UPLOAD_FOLDER = 'target/person'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_face(dir):
    faces = list()
    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename
        # print('load face path : ', path)
        face = extract_face(path)
        faces.append(face)
    return faces


def load_dataset(dir):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        # path = dir
        # print('PATH  :   ', path)
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        # print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]


def run_train_dataset():
    # load train dataset
    trainX, trainy = load_dataset('input/data/train/')
    # print(trainX.shape, trainy.shape)

    # save and compress the dataset for further use
    np.savez_compressed('geocontrol_train.npz', trainX, trainy)

    data = np.load('geocontrol_train.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    # print('Loaded: ', trainX.shape, trainy.shape)

    # load the facenet model
    facenet_model = load_model('models/facenet_keras.h5')
    # print('Loaded Model')

    # convert each face in the train set into embedding
    emdTrainX = list()
    for face in trainX:
        emd = get_embedding(facenet_model, face)
        emdTrainX.append(emd)

    emdTrainX = np.asarray(emdTrainX)
    # print('EMDTRAINX ************************************', emdTrainX.shape)

    # save arrays to one file in compressed format
    np.savez_compressed('geocontrol-embeddings_train.npz', emdTrainX, trainy)
    return emdTrainX, trainy


def run_test_dataset():
    # load test dataset
    testX, testy = load_dataset('./target/')
    # print(testX.shape, testy.shape)

    # save and compress the dataset for further use
    np.savez_compressed('geocontrol_test.npz', testX, testy)

    # load the face dataset
    data = np.load('geocontrol_test.npz')
    testX, testy = data['arr_0'], data['arr_1']
    # print('Loaded: ', testX.shape, testy.shape)

    # load the facenet model
    facenet_model = load_model('models/facenet_keras.h5')
    # print('Loaded Model')

    # convert each face in the test set into embedding
    emdTestX = list()
    for face in testX:
        emd = get_embedding(facenet_model, face)
        emdTestX.append(emd)

    emdTestX = np.asarray(emdTestX)
    # print('EMDTESTX ************************************', emdTestX.shape)

    # save arrays to one file in compressed format
    np.savez_compressed('geocontrol-embeddings_test.npz', emdTestX, testy)
    # print('EMDTESTX ************************************')
    # print(emdTestX)
    # print('TESTY ************************************')
    # print(testy)
    return emdTestX, testy


def run_test():
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import Normalizer
    from sklearn.svm import SVC

    data = np.load('geocontrol-embeddings_test.npz')
    emdTrainX, trainy = data['arr_0'], data['arr_1']

    data = np.load('geocontrol-embeddings_train.npz')
    emdTestX, testy = data['arr_0'], data['arr_1']

    # load test dataset
    testX, testy = load_dataset('input/data/test/')
    print(testX.shape, testy.shape)
    print("Dataset: train=%d, test=%d" %
          (emdTrainX.shape[0], emdTestX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer()
    emdTrainX_norm = in_encoder.transform(emdTrainX)
    emdTestX_norm = in_encoder.transform(emdTestX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy_enc = out_encoder.transform(trainy)
    testy_enc = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(emdTrainX_norm, trainy_enc)
    # predict
    yhat_train = model.predict(emdTrainX_norm)
    yhat_test = model.predict(emdTestX_norm)
    # score
    score_train = accuracy_score(trainy_enc, yhat_train)
    score_test = accuracy_score(testy_enc, yhat_test)
    # summarize
    # print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    from random import choice
    # select a random face from test set

    """
    AQUI ENTRARIA A FOTO QUE EU QUERO TESTAR.
    A DIFERENÇA É QUE ELE JÁ TINHA FEITO O PROCESSO PARA TODAS AS IMAGENS PRO TESTE
    NO CASO, 
    """
    selection = choice([i for i in range(testX.shape[0])])
    random_face = testX[selection]
    random_face_emd = emdTestX_norm[selection]
    random_face_class = testy_enc[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])

    # prediction for the face
    samples = np.expand_dims(random_face_emd, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    all_names = out_encoder.inverse_transform([0, 1, 2])
    predicted = 'Predicted: %s (%.3f)' % (predict_names[0], class_probability)
    # print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0]*100))
    expected = 'Expected: %s' % random_face_name[0]
    return jsonify(predicted, expected)
    # plot face
    # plt.imshow(random_face)
    # title = '%s (%.3f)' % (predict_names[0], class_probability)
    # plt.title(title)
    # plt.show()


def run():
    emdTrainX, trainy = run_train_dataset()
    emdTestX, testy = run_test_dataset()
    texto = run_test()
    return texto


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'geocontrol'


@app.route('/')
def index():
    return 'INdex'


@app.route('/adicionar')
def adicionar():
    return 'Oi'


@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            # print(image + "Uploaded to Faces")
            # flash('Image successfully Uploaded to Faces.')
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
            filename = os.path.join(UPLOAD_FOLDER, image.filename)
            print("stored as:" + filename)
            return render_template("upload.html", uploaded_image=filename)
    return render_template("upload.html")


@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):

    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/identificar', methods=['GET', 'POST'])
def identificar():
    # texto = run()
    run_test_dataset()
    texto = run_test()
    return texto


# app.run(host='0.0.0.0')

run()
