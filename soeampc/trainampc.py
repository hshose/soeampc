import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from keras.layers.core import Lambda
from keras.callbacks import EarlyStopping
# from keras.backend import clip
import keras.backend as backend
from sklearn.model_selection import train_test_split

import math
from pathlib import Path
from datetime import datetime

from .utils import export_model

def clipped_mae(y_true, y_pred):
    #   Args:
    #   y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    #   y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    #   Returns:
    #   Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
    ue   = 0.7853
    eps = 1e-6
    umin = -ue+eps
    umax = 2-ue-eps
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.convert_to_tensor(y_pred)
    mask = backend.equal(backend.less(y_true,umax), backend.greater(y_true, umin))
    mask_ytrue_umax = backend.greater_equal(y_true, umax)
    mask_ytrue_umin = backend.less_equal(y_true, umin)
    mask_ypred_umax = backend.greater_equal(y_pred,umax)
    mask_ypred_umin = backend.less_equal(y_pred,umin)
    mask=tf.math.logical_or(tf.math.logical_and(mask_ytrue_umin,mask_ypred_umin),tf.math.logical_and(mask_ytrue_umax,mask_ypred_umax))
    # print(tf.abs(y_pred - y_true)[tf.math.logical_not(mask)])
    return backend.mean((tf.abs(y_pred - y_true)[tf.math.logical_not(mask)]), axis=-1)

def generatemodel(traindata, architecture, clipped_mae=False):

    X_normalizer = layers.Normalization(input_shape=[architecture[0],], axis=None)
    X_normalizer.adapt(traindata)

    model = keras.Sequential()
    model.add(X_normalizer)

    for units in architecture[:-1]:

        model.add(layers.Dense(units=units, activation="tanh"))

    model.add(layers.Dense(units=architecture[-1], activation="linear"))


    if clipped_mae:

        def lam(x):
            from keras import backend
            ue      = 0.7853
            umin = -ue
            umax = 2-ue
            return backend.clip(x, min_value=umin, max_value=umax)
        
        model.add(Lambda(lam, input_shape=(None, 10), output_shape=(None, 10)))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=clipped_mae,
            metrics=['mse', 'mae'])

    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_absolute_error',
            metrics=['mse', 'mae'])

    return model


def hyperparametertuning(mpc, X, Y, datasetname, architectures, maxepochs=5000, patience=1000):
    print("\nperforming hyperparameter tuning\n")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.84, random_state=42)
    print(X_train.shape)
    print(Y_train.shape)
    maxdepth = 100
    maxwidth = 3
    maxsteps = 5
    p = Path("models").joinpath(datasetname)
    p.mkdir(parents=True,exist_ok=True)
    for a in architectures:
        if a[0] != mpc.nx:
            raise Exception('architecture does not have nx input neurons')
        if a[-1] != mpc.nu*mpc.N:
            raise Exception('architecture does not have nu*N output neurons')
        print("\n\n===============================================")
        print("Training Model",a)
        print("===============================================\n")
        model = generatemodel(X_train, a)
        model.summary()
        batch_size = 10000
        overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = patience)
        history = model.fit(
            X_train,
            Y_train,
            verbose=2,
            batch_size=batch_size,
            epochs=maxepochs,
            validation_split = 0.2,
            callbacks=[overfitCallback]
            )
        
        testresult, mu = statisticaltest(mpc, model, X_test)
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        modelname = '-'.join([str(d) for d in a]) + '_mu=' + ('%.2f' % mu) + '_' + date
        export_model(model, datasetname, modelname)
        if testresult:
            return model
    return None


def statisticaltest(mpc, model, testpoints, p=int(10e3), Tf=20, Nol=10, delta_h=0.10, mu_crit=0.80):
    p = min(np.shape(testpoints)[0], p)
    print("\noffline testing on \n\tp = ", p)
    I = np.zeros(p)
    for j in range(p):
        x0 = testpoints[j, :]
        U = model(x0).numpy()
        U = np.reshape(U, (mpc.N, mpc.nu))
        X = mpc.forwardsim(x0,U)
        I[j] = mpc.feasible(X,U)
    
    mu = np.mean(I)
    print("\t mean(I) =", mu)
    epsilon = math.sqrt(-math.log(delta_h/2)/(2*p))
    print("\t epsilon =", epsilon)
    if mu_crit <= mu - epsilon:
        print("test passed\n")
        return True, mu
    print("test failed for mu_crit <= mu - epsilon with", mu_crit, "!<=", mu-epsilon,"\n")
    return False, mu

def teststatisticaltest(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = keras.models.load_model('models/latest')
    if statisticaltest(model, X_test):
        return model