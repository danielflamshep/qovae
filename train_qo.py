from __future__ import print_function
from collections import OrderedDict
import os, h5py, pdb
import numpy as np
import keras
from models.model_qo import QoVAE
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_utils import *
from experiments import *

def main(hypers=None, save_name ='model1'):
    h5f = h5py.File('data/qo_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()
    nt = 2000 #  specify test set size
    XTE, XTR = data[0:nt], data[nt:]

    np.random.seed(1)
     # model name
    model_save = 'results/'+save_name
    model = QoVAE()
    if not hypers:
        hypers = OrderedDict(max_len=10, beta=5.0, lr=0.005, latent=5, epochs=500, batch=256, hidden=128,
                             dense1=128, dense2=64, convf=8, conv1=4, conv2=4, conv3=4)

    if os.path.isfile(model_save): # 3. if this results file exists already load it
        model.load(devices, model_save, hypers=hypers) #; print('loading model'+'---'*100)
    else:
        print('training new model', model_save)
        model.create(devices, hypers=hypers)

    class MC(keras.callbacks.Callback):
        def __init__(self, model):
            self.model_to_save = model
        def on_epoch_end(self, epoch, logs=None):
            if epoch == hypers['epochs']-1:
                self.model_to_save.save(model_save)
    cb = MC(model)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, min_lr = 0.0001)
    hist = model.autoencoder.fit(XTR, XTR, shuffle = True, nb_epoch = hypers['epochs'],
                               batch_size = hypers['batch'], callbacks = [cb], validation_split = 0.1)


if __name__ == '__main__':
    hypers = OrderedDict(max_len=10, beta=5.0, lr=0.005, latent=5, epochs=5, batch=256, hidden=128,
                         dense1=128, dense2=64, convf=8, conv1=4, conv2=4, conv3=4)
    make_data(hypers)
    main(hypers, save_name ='model1')
    run_experiments(hypers, save_name ='model1')
