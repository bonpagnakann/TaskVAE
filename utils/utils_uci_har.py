# cnn model for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from utils_pamap2 import *
from sklearn.metrics import classification_report
from time import time
from sklearn.model_selection import ShuffleSplit

from numpy.random import seed

seed(1)

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()

    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt',
    'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt',
    'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt',
    'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    #trainy = to_categorical(trainy)
    #testy = to_categorical(testy)
    #X_train, X_val, y_train, y_val = train_test_split(trainX, trainy, test_size=0.1, stratify=trainy)

    return trainX, trainy, testX, testy

# standardize data
def scale_data(trainX, testX, standardize=True):
    # remove overlap
    cut = int(trainX.shape[1] / 2)
    longX = trainX[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
    flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
    # standardize
    if standardize:
        s = StandardScaler()
        # fit on training data
        s.fit(longX)
        # apply to training and test data
        longX = s.transform(longX)
        flatTrainX = s.transform(flatTrainX)
        flatTestX = s.transform(flatTestX)
    # reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
    flatTestX = flatTestX.reshape((testX.shape))
    return flatTrainX, flatTestX

def create_model(n_filters,f_size, n_timesteps, n_features, n_outputs, learning_rate):

    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=f_size, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=n_filters, kernel_size=f_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, n_filters, f_size, learning_rate, weight=None):
    verbose, epochs, batch_size = 0, 50, 64
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model_check.h5', verbose=0, save_best_only=True)
    #print(trainX.shape)
    #print(trainy.shape)
    #print(testX.shape)
    #print(testy.shape)
    model_history = []
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], 12
    # scale data
    trainX, testX = scale_data(trainX, testX)

    # fit network

    model = None
    model = create_model(n_filters, f_size, n_timesteps, n_features, n_outputs, learning_rate)
    start = time()
    model.fit(trainX, trainy, epochs=epochs, validation_split=0.15 , batch_size=batch_size, verbose=verbose, callbacks=[callback,model_checkpoint],shuffle=True)
    duration = time()-start
    #print("Training Duration: ", duration)
    # evaluate model
    model = load_model('model_check.h5')
    loss, accuracy = model.evaluate(testX, testy, verbose=0)

    #predict_y = model.predict(testX)
    #y_pred = np.argmax(predict_y, axis=1)
    #print(classification_report(testy, y_pred, labels=[0, 1, 2, 3, 4, 5,6,7,8,9,10,11]))

    return loss, accuracy, duration 

# summarize scores
def summarize_results(scores, params):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = mean(scores[i]), std(scores[i])
        print('Param=%s: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    pyplot.boxplot(scores, labels=params)
    pyplot.savefig('exp_cnn_standardize.png')

# run an experiment
def run_experiment(params, f_size, learning_rate, repeats=5):
    # load data
    # test each parameter
    all_scores = list()
    for p in params:
        # repeat experiment
        scores = list()
        #print("Number of filters: ", p)
        for f in f_size:
            for lr in learning_rate:
                for r in range(30):
                    trainX, trainy, testX, testy= load_dataset()
                    loss, score, duration = evaluate_model(trainX, trainy, testX, testy, p, f, lr)
                    score = score * 100.0
                    print('%s, %s, %d, %f, %.3f, %.3f, %.3f' % (p, f, r+1, lr, loss, score, duration))
                    #scores.append(score)
                #all_scores.append(scores)
        # summarize results
    #summarize_results(all_scores, params)
