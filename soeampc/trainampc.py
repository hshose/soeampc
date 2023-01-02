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
import os
import errno

from .datasetutils import export_model

def import_model(dataset_name="latest", modelname="latest"):
    """imports tensorflow keras model    
    """
    p = Path("models").joinpath(dataset_name).joinpath(modelname)
    model = keras.models.load_model(p)
    return model

def export_model(model, dataset_name, modelname):
    """exports tensorflow keras model
    """
    p = Path("models").joinpath(dataset_name)
    model.save(p.joinpath(modelname))
    link_name=p.joinpath("latest")
    target=modelname
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def clipped_mae(y_true, y_pred):
    """clipped mae loss, WARNING: HARDCODED LIMITS!!!! DO NOT USE
    """
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

# def mpc_cost_mae_with_slack(y_true, y_pred):
#     Jtrue = 0
#     Jpred = 0
#     e = 0
#     return Jpred + e - Jtrue

def generate_model(traindata, architecture, output_shape):
    """returns a fully connected feedforward NN normalized to traindata
    
    Input layer is normalizer to traindata, hidden layers are tanh activation,
    output layer is linear activation with N*nu neurons, where N is MPC horizon,
    reshape layer is appended.

    Args:
        traindata:
            numpy array of initial conditions, traindata.shape = (Nsamples, nx)
        architecture:
            array of layer widths: [nx, N_hidden_1, N_hidden_2, ..., N*nu]
        output_shape:
            shape_like of output, e.g. (N, nu)
    Returns:
        tensorflow keras model
    """
    X_normalizer = layers.Normalization(input_shape=[architecture[0],], axis=None)
    X_normalizer.adapt(traindata)

    model = keras.Sequential()
    model.add(X_normalizer)

    for units in architecture[:-1]:
        initializer = tf.keras.initializers.GlorotNormal()
        model.add(layers.Dense(units=units, activation="tanh", kernel_initializer=initializer))

    model.add(layers.Dense(units=architecture[-1], activation="linear"))
    model.add(layers.Reshape(output_shape))


    # if clipped_mae:

    #     def lam(x):
    #         from keras import backend
    #         ue      = 0.7853
    #         umin = -ue
    #         umax = 2-ue
    #         return backend.clip(x, min_value=umin, max_value=umax)
        
    #     model.add(Lambda(lam, input_shape=(None, 10), output_shape=(None, 10)))
    #     loss=clipped_mae
    # else:
        # loss='mean_absolute_error'
    
    loss='mean_squared_error'

    # initial_learning_rate = 0.01
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #         initial_learning_rate,
    #         decay_steps=1000,
    #         decay_rate=0.96,
    #         staircase=True)
 

    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    #     return lr

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # lr_metric = get_lr_metric(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mse', 'mae'])

    return model

def train_model(model, X_train, Y_train, dataset_name, batch_size=int(1e4), max_epochs=int(1e3), patience=int(1e3), learning_rate=1e-3):
    """trains a given model on X,Y dataset

    Args:
        model:
            tensorflow keras model
        X_train:
            training data of initial conditions x0
        Y_train:
            training data of predicted input sequences
        dataset_name:
            used for saving the model with name it has been trained on
        batch_size:
            batch size used for learning
        max_epochs:
            number of epochs used for learning
        patience:
            if loss is not reduced for patience epochs, training is aborted
        learning_rate:
            learning_rate used for training
    Returns:
        trained tensorflow keras model
    """

    backend.set_value(model.optimizer.learning_rate, learning_rate)

    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = patience)

    checkpoint_filepath = Path("models").joinpath(dataset_name, "checkpoint")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        # save_weights_only=True,
        save_freq='epoch',
        period = 100
        )

    history = model.fit(
        X_train,
        Y_train,
        verbose=2,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split = 0.1,
        callbacks=[overfitCallback, model_checkpoint_callback]
        )
    
    return model

def retrain_model(mpc, model, X, Y, architecture_string, dataset_name, batch_size=int(1e4), max_epochs=int(1e3), patience=int(1e3), learning_rate=1e-3):
    """retrains a given model on X, Y dataset
    
    Args:
        mpc:
            mpc class object
        model:
            tensorflow keras class object
        X:
            initial conditions x0
        Y:
            input sequences corresponding to initial conditions x0
        architecture_string:
            string used for model saving
        dataset_name:
            string used for dataset saving
        batch_size:
            batch size used for learning
        max_epochs:
            number of epochs used for learning
        patience:
            if loss is not reduced for patience epochs, training is aborted
        learning_rate:
            learning_rate used for training
    Returns:
        tensorflow keras model
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    print("X[42]", X_train[42])
    model = train_model(model=model, X_train=X_train, Y_train=Y_train, dataset_name=dataset_name, batch_size=batch_size, max_epochs=max_epochs, patience=patience, learning_rate=learning_rate)
    testresult, mu = statisticaltest(mpc, model, X_test, Y_test)
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    modelname = architecture_string + '_mu=' + ('%.2f' % mu) + '_' + date
    export_model(model, dataset_name, modelname)
    return model

def hyperparametertuning(mpc, X, Y, dataset_name, architectures, max_epochs=int(1e5), patience=int(1e3), batch_size=int(1e4)):
    print("\nperforming hyperparameter tuning\n")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    output_shape=Y_train.shape[1:]
    p = Path("models").joinpath(dataset_name)
    p.mkdir(parents=True,exist_ok=True)
    for a in architectures:
        if a[0] != mpc.nx:
            raise Exception('architecture does not have nx input neurons')
        if a[-1] != mpc.nu*mpc.N:
            raise Exception('architecture does not have nu*N output neurons')
        print("\n\n===============================================")
        print("Training Model",a)
        print("===============================================\n")
        model = generatemodel(X_train, a, output_shape)
        model.summary()

        model = train_model(model, X_train, Y_train, dataset_name, batch_size, max_epochs, patience)

        testresult, mu = statisticaltest(mpc, model, X_test, Y_test)
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        modelname = '-'.join([str(d) for d in a]) + '_mu=' + ('%.2f' % mu) + '_' + date
        export_model(model, dataset_name, modelname)
        if testresult:
            return model
    return None


def statisticaltest(mpc, model, testpoints_X, testpoints_V, p=int(10e3), delta_h=0.10, mu_crit=0.80):
    """Tests if model yields a yields a feasible solution to mpc in mu_crit fraction of testpoints_X
    
    Uses Hoeffdings inequality on indicator function I. If I(x0) = 1 iff model(x0) is a feasible solution to mpc problem.
    mean(I) is compared to mu_crit using Hoeffdings inequality with confidence level delta_h.

    Args:
        mpc:
            mpc class object, that should implement a method `mpc.forward_simulate_trajectory(x0,V)` and `mpc.feasible(X,V)`
        model:
            model class object, that should implement a call operator `model(x0).numpy()`.
            This would typically be a keras / tensorflow model
        testpoints_X:
            numpy array of N samples of initial states x0.
            testpoints_X.shape=(Nsamples,mpc.nx)
        testpoints_V:
            numpy array of N samples of input sequences corresponding to states x0.
            testpoints_V.shape=(Nsamples, mpc.N, mpc.nu)
        p:
            number of samples to evaluate, p should be less than Nsamples
        delta_h:
            required confidence level
        mu_crit:
            required accuracy

    Returns:
        A tuple (passed, mu), where passed is boolean indicating if mu_crit is achieved and mu
            is mean(I)
    """
    p = min(np.shape(testpoints_X)[0], p)
    print("\noffline testing on \n\tp = ", p)
    I = np.zeros(p)
    dist = np.zeros((p,mpc.nu))
    for j in range(p):
        x0 = testpoints_X[j, :]
        Vtrue = testpoints_V[j]
        V = model(x0).numpy()
        V = np.reshape(V, (mpc.N, mpc.nu))
        # for k in range(mpc.nu):
        #     U[k,:] = np.clip(U[k,:], mpc.umin[k], mpc.umax[k])
        X = mpc.forward_simulate_trajectory(x0,V)
        I[j] = mpc.feasible(X,V)
        dist[j] = np.linalg.norm(V-Vtrue, np.inf, 1)
    
    mu = np.mean(I)
    print("\t mean(I) =", mu)
    epsilon = math.sqrt(-math.log(delta_h/2)/(2*p))
    print("\t epsilon =", epsilon)

    worst_case_passing_dist = np.max(dist[I==1],0)
    best_case_not_passing_dist = np.min(dist[I==0],0)

    print("\t worst case passing dist (I==1): V-Vtrue =", worst_case_passing_dist)
    print("\t best case not passing dist (I==0): V-Vtrue =", best_case_not_passing_dist)

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