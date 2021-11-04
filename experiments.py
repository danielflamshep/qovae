import sys, random
import numpy as np

from plotting import *
from data_utils import *
from train_qo import *
from testqo import * # calculate #, ex_setups
import qo_vae


class Experiment():
    def __init__(self, hypers=None, save_name=''):
        """conduct experiments """
        self.train_ss, self.train_es = load(hypers)
        self.lt = len(self.train_ss)
        weights = "results/"+save_name
        self.model = qo_vae.QoCharacterModel(weights, hypers)
        self.latent_size = hypers['latent']

    def plot_save_metrics(self, num=1000, prob=True):
        """ saves a file of metrics computes a bunch of plots"""
        zs = np.random.randn(num, self.latent_size)
        samples = self.model.decode(zs, probabilistic=prob, return_lists=False)

        valid, entropys, ess, srvs, states, probs = entropy_eval(np.array(samples), 'valid')
        valid = ['.'.join(setup).strip() for setup in valid]

        num_gen = len(samples)
        num_valid = len(valid)
        unique = set(valid)
        num_unique = len(unique)

        train_data = ['.'.join(setup).strip() for setup in self.train_ss]
        in_train = set(valid).intersection(train_data)
        novel = [s for s in unique if s not in in_train]
        num_unique_in_train = len(set(valid).intersection(train_data))
        num_novel = num_unique - num_unique_in_train

        m = 'number of entangled = {} , unique = {} , novel = {}'.format(num_valid, num_unique, num_novel)
        log("metrics", m)
        log('ratio entangled', str(float(num_valid)/float(num_gen)))

        return novel

    def plot_latent_space(self):
        """ plot latent space """
        zs = self.model.encode(self.train_ss)
        y = np.array(self.train_es) #np.array(self.train_es)
        plot_latent(zs, y, 'latent', k=2)



def entropy_eval(samples, title='', nrow=None):
    """ calculates entanglement for list of setups and return lists of
        1) entangled experiments 2) S values 3) partition entropies
        4) srvs 5) states 6) basis state probabilties """

    if isinstance(samples[0], str):
        samples = [setup.split('.') for setup in samples]

    valid, entropys, ess, srvs, states, probs = [], [], [], [], [], []
    all_entropys, all_srvs, all_ess, all_states = [], [], [], []

    for setup in samples:
        try:
            state, prob, srv, es, e = calculate(setup, title)
            if e != 0.0:
                print('succes')
                states.append(state)
                probs.append(prob)
                srvs.append(srv)
                ess.append(np.round(es,3))
                entropys.append(np.round(e,3))
                valid.append(setup)

            all_states.append(state)
            all_entropys.append(np.round(e,3))
            all_srvs.append(srv)
            all_ess.append(np.round(es,3))
        except:
            print('FAILED TO CALC ENTROPY', setup)
            all_states.append('F')
            all_srvs.append(['F'])
            all_ess.append(['F'])
            all_entropys.append('F')

    return valid, entropys, ess, srvs, states, probs

def log(fname, text):
    print(text)
    with open(fname, 'a') as f:
        f.write(text + '\n')

def run_experiments(hypers,save_name ='model1'):
    exp = Experiment(hypers, save_name)
    exp.plot_save_metrics(num=1000)
