import numpy as np
from scipy.io import loadmat

data = loadmat('ex3data1.mat')
X = data['X']
y_t = data['y']

num_labels = 10
y = np.zeros([y_t.shape[0], num_labels])
for i in range(0, y_t.shape[0]):
    if y_t[i] == 10:
        y_t[i] = 0
    y[i, y_t[i]] = 1

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=25, input_shape=(400,), kernel_initializer='glorot_uniform', activation='sigmoid'))
    classifier.add(Dense(units=10, kernel_initializer='glorot_uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=32, epochs=50)
accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=10, n_jobs=1)