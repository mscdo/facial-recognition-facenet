import numpy as np  # linear algebra
import os
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import numpy as np  # linear algebra
import mtcnn
import cv2

print(mtcnn.__version__)


MINIMUM_MATCH = 20


# Ver direitinho depois
def get_model_score(model: SVC, emdTestX, trainy, testy, emdTrainX_norm):
    # normalize input vectors
    in_encoder = Normalizer()
    emdTestX_norm = in_encoder.transform(emdTestX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy_enc = out_encoder.transform(trainy)
    testy_enc = out_encoder.transform(testy)
    # predict
    yhat_train = model.predict(emdTrainX_norm)
    yhat_test = model.predict(emdTestX_norm)
    # score
    score_train = accuracy_score(trainy_enc, yhat_train)
    score_test = accuracy_score(testy_enc, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' %
          (score_train*100, score_test*100))


def create_model(emdTrainX, trainy, emdTestX, testy):
    print("Dataset: train=%d, test=%d" %
          (emdTrainX.shape[0], emdTestX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer()
    try:
       emdTrainX_norm = in_encoder.transform(emdTrainX)
    except:
        raise ValueError('Verifique se há dataset de teste','emdTrainX_norm')
    
    try:
        emdTestX_norm = in_encoder.transform(emdTestX)
    except:
        raise ValueError('Verifique se há dataset de teste','emdTestX_norm')
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
    print('Accuracy: train=%.3f, test=%.3f' %
          (score_train*100, score_test*100))

    return model, in_encoder, out_encoder


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


def train_test_dataset():
    # load test dataset
    testX, testy = load_dataset('./input/data/test/')
    
    # print(testX.shape, testy.shape)
    if len(testy) == 0:
         raise vars('Verifique se há arquivos de teste', 'testy')
    # save and compress the dataset for further use
    #np.savez_compressed('geocontrol_test.npz', testX, testy)
    # load the face dataset
    # data = np.load('geocontrol_test.npz')
    # testX, testy = data['arr_0'], data['arr_1']
    print('Loaded TESTS: ', testX.shape, testy.shape)
    # load the facenet model
    facenet_model = load_model('./models/facenet_keras.h5')
    print('Loaded Model')

    # convert each face in the test set into embedding
    emdTestX = list()
    for face in testX:
        emd = get_embedding(facenet_model, face)
        emdTestX.append(emd)

    emdTestX = np.asarray(emdTestX)
    print('EMDTESTX ************************************', emdTestX.shape)
    return emdTestX, testy



def train_train_dataset():
    # load train dataset

    trainX, trainy = load_dataset('./input/data/train/')
    print(trainX.shape, trainy.shape)

    # # save and compress the dataset for further use
    # np.savez_compressed('geocontrol_train.npz', trainX, trainy)

    # # load the face dataset
    # data = np.load('geocontrol_train.npz')
    # trainX, trainy = data['arr_0'], data['arr_1']
    print('Loaded: ', trainX.shape, trainy.shape,)
    # load the facenet model
    facenet_model = load_model('./models/facenet_keras.h5')
    print('Loaded Model')
    # convert each face in the train set into embedding
    emdTrainX = list()
    for face in trainX:
        emd = get_embedding(facenet_model, face)
        emdTrainX.append(emd)
    emdTrainX = np.asarray(emdTrainX)
    print('EMDTRAINX ************************************', emdTrainX.shape)

    # save arrays to one file in compressed format
    # np.savez_compressed('geocontrol-embeddings_train.npz',
    #                     emdTrainX, trainy)
    return emdTrainX, trainy


def train_dataset():
    emdTrainX, trainy = train_train_dataset()
    emdTestX, testy = train_test_dataset()
    # save arrays to one file in compressed format
    np.savez_compressed('geocontrol-embeddings.npz',
                        emdTrainX, trainy, emdTestX, testy)



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
    try:
        x1, y1, width, height = results[0]['box']
    except:
        raise IndexError(
            'Não foi encontrado um rosto na foto. Favor enviar foto válida.', filename)

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
        print('load face path : ', path)
        face = extract_face(path)
        faces.append(face)
    return faces


def load_dataset(dir):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        # path = dir
        print('PATH  :   ', path)
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" %
              (len(faces), subdir))  # print progress
        X.extend(faces)
        y.extend(labels)

    return np.asarray(X), np.asarray(y)


def apply_gaussian_blur(dir):
    for subdir in os.listdir(dir):
        path = os.path.join(dir, subdir) + '/'
        for filename in os.listdir(path):
            print(os.path.join(path, filename))
            img = cv2.imread(os.path.join(path, filename))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(path, filename), gray)


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


def identify_new_face(model: SVC, in_encoder: LabelEncoder, out_encoder: LabelEncoder):
    testX, testy = load_dataset('target/')

    # print(testX.shape, testy.shape)
    facenet_model = load_model('./models/facenet_keras.h5')
    emdTestX = list()
    for face in testX:
        emd = get_embedding(facenet_model, face)
        emdTestX.append(emd)
    emdTestX = np.asarray(emdTestX)
    emdTestX_norm = in_encoder.transform(emdTestX)

    samples = np.expand_dims(emdTestX_norm[0], axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    # Cria array de acordo com a qtdade de labels (y)
    number_dir = len(next(os.walk(os.path.join('./input/data/train')))[1])
    array = np.arange(number_dir)
    all_names = out_encoder.inverse_transform(array)
    # Verifica se a probabilidade    maior que 60%
    if (class_probability < MINIMUM_MATCH):
        texto, name = f'Não houve nenhuma correspondência com mais de {MINIMUM_MATCH}% na base de dados.',  ''
    # texto, name = f' {predict_names[0]} , {round(class_probability, 2)} %', predict_names[0]
    else:
        name = predict_names[0].split("_")
        if len(name) > 0:
            name = " ".join(name)
        texto = f'{round(class_probability, 2)}%'
        print(texto, name, f'{round(class_probability, 2)}%')
    return texto, name
