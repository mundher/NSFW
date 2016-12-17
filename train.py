import numpy as np
np.random.seed(3)
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import glob
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')

def toNumpy():
    def readImages(path, size):
        images = np.zeros((size,3,128,128), dtype=np.float32)
        for i in range(size):
            im = Image.open(path+str(i)+'.png')
            im = np.array(im)
            im = im.transpose(2, 0, 1)
            im = im.astype(np.float32)/255.0
            images[i] = im
        return images

    def buildDataset(name):
        yTrain = map(int, open('data\\'+name+'.txt').readlines())
        xTrain = readImages('data\\'+name+'\\', len(yTrain))
        yTrain = np.array(yTrain)
        np.save('data\\'+name+'X.npy', xTrain)
        np.save('data\\'+name+'Y.npy', yTrain)
    buildDataset('train')
    buildDataset('valid')
    buildDataset('test')


def buildModel():
    img_rows = img_cols = 128
    conv_input_shape=(3, img_rows, img_cols)
    conv_nb_row = 3
    conv_nb_col = 3
    conv_nb_pool = 2
    decay = l2(0.01)
    model = Sequential()

    model.add(Convolution2D(16,conv_nb_row,conv_nb_col,border_mode='valid',input_shape=conv_input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(conv_nb_pool, conv_nb_pool)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(32, conv_nb_row, conv_nb_row))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(conv_nb_pool, conv_nb_pool)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, conv_nb_row, conv_nb_row))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, conv_nb_row, conv_nb_row))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(conv_nb_pool, conv_nb_pool)))
    model.add(Dropout(0.3))


    model.add(Convolution2D(128, conv_nb_row, conv_nb_row))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, conv_nb_row, conv_nb_row))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, conv_nb_row, conv_nb_row))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(conv_nb_pool, conv_nb_pool)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, W_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1,W_regularizer=decay))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print 'finish build model'
    return model

def trainCNNAug():
    model = buildModel()

    json_string = model.to_json()
    f = open('model2.json', 'w')
    f.write(json_string)
    f.close()

    trainX = np.load('data\\trainX.npy')
    trainY = np.load('data\\trainY.npy')
    validX = np.load('data\\validX.npy')
    validY = np.load('data\\validY.npy')

    mu = np.mean(trainX, axis=0)
    np.save('mu.npy', mu)

    validX -= mu
    print('loaded')

    datagen = ImageDataGenerator(horizontal_flip=True, featurewise_center=True)
    datagen.mean = mu

    ckpoint = ModelCheckpoint('model2.{epoch:02d}.hdf5', monitor='val_acc', save_best_only=True, verbose=1)
    history = model.fit_generator(datagen.flow(trainX, trainY, batch_size=128),
                                  samples_per_epoch=len(trainX), nb_epoch=300, validation_data=(validX, validY),
                                  callbacks=[ckpoint])
    np.save('acc2.npy', np.array(history.history['acc']))
    np.save('val_acc2.npy', np.array(history.history['val_acc']))


def getCnnPredictions(x,y):
    names = glob.glob('model.*.hdf5')[-8:]
    predictions = np.zeros((x.shape[0],len(names)))
    for i in range(len(names)):
        w = names[i]
        js = open('model.json')
        model = model_from_json(js.read())
        model.load_weights(w)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        predictions[:,i]= model.predict(x,batch_size=512).reshape(x.shape[0])
        acc = calcAccuarcy(predictions[:,i],y)
        print('model: {0} accuracy = {1}'.format(w,acc))
    return predictions

def loadTestData(pathx,pathy):
    x = np.load(pathx)
    x -= np.load('mu.npy')
    return x,np.load(pathy)

def calcAccuarcy(predicted, target):
    out = predicted>=0.5
    return np.mean(out == target)

def calcWeightedModels(x,y, test):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression(n_jobs=4)
    lr.fit(x,y)
    print 'coff', lr.coef_, lr.intercept_
    return lr.predict(x), lr.predict(test)

def drawRoc(prediction, y):
    fpr, tpr, thresholds = roc_curve(y, prediction)
    area = auc(fpr, tpr)
    print area
    plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def testRoc():
    print('load validation set')
    x, y = loadTestData('data\\validX.npy', 'data\\validY.npy')
    predictions = getCnnPredictions(x, y)
    xTest, yTest = loadTestData('data\\testX.npy', 'data\\testY.npy')
    print('load test set')
    predictionsTest = getCnnPredictions(xTest, yTest)
    lrPredictions, lrPredictionsTest = calcWeightedModels(predictions, y, predictionsTest)
    avgPredictions = predictions.mean(axis=1)
    avgPredictionsTest = predictionsTest.mean(axis=1)
    acc = calcAccuarcy(lrPredictions, y)
    accTest = calcAccuarcy(lrPredictionsTest, yTest)
    accAvg = calcAccuarcy(avgPredictions, y)
    accAvgTest = calcAccuarcy(avgPredictionsTest, yTest)
    print('Linear Regression on Validation set accuracy = {0}'.format(acc))
    print('Linear Regression on Test Set accuracy = {0}'.format(accTest))
    print('Average models on Validation set accuracy = {0}'.format(accAvg))
    print('Average models on Test Set accuracy = {0}'.format(accAvgTest))
    drawRoc(lrPredictions, y)
    drawRoc(avgPredictions, y)
    drawRoc(lrPredictionsTest, yTest)
    drawRoc(avgPredictionsTest, yTest)


def testPredict():
    print('load validation set')
    x, y = loadTestData('data\\validX.npy','data\\validY.npy')
    predictions = getCnnPredictions(x,y)
    xTest, yTest = loadTestData('data\\testX.npy', 'data\\testY.npy')
    print('load test set')
    predictionsTest = getCnnPredictions(xTest, yTest)
    lrPredictions, lrPredictionsTest = calcWeightedModels(predictions,y, predictionsTest)
    avgPredictions = predictions.mean(axis=1)
    avgPredictionsTest = predictionsTest.mean(axis=1)
    acc = calcAccuarcy(lrPredictions,y)
    accTest = calcAccuarcy(lrPredictionsTest, yTest)
    accAvg = calcAccuarcy(avgPredictions,y)
    accAvgTest = calcAccuarcy(avgPredictionsTest,yTest)
    print('Linear Regression on Validation set accuracy = {0}'.format(acc))
    print('Linear Regression on Test Set accuracy = {0}'.format(accTest))
    print('Average models on Validation set accuracy = {0}'.format(accAvg))
    print('Average models on Test Set accuracy = {0}'.format(accAvgTest))


#toNumpy()
#trainCNNAug()
#testPredict()
testPredict()