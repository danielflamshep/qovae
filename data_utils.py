import numpy as np
import pdb, h5py, random
from collections import OrderedDict

def many_one_hot(indices, d):
    t = indices.shape[0]
    oh = np.zeros((t,d))
    oh[np.arange(t), indices] = 1
    return oh

def log_print(fname, text):
    with open(fname, 'a') as f:
        print(text)
        f.write(text + '\n')

def load_entropy():
    """loads entropy values for setup strings"""
    es = open('entropy.smi','r')
    L = []
    for line_e in es:
        line_e = line_e.strip()
        L.append(float(line_e))
    es.close()
    return L

def load(hypers):
    ss, es = [], []

    es_ = load_entropy()
    e = open('setups.smi','r')

    for i, line in enumerate(e):
        line = line.strip()
        line = line.split('.')
        if len(line) < hypers['max_len']:
            ss.append(line)
            es.append(es_[i])
    e.close()
    return ss, es

def make_data(hypers):
    """ creates data tensor """

    ss, es = load(hypers)

    count = 0
    MAX_LEN = hypers['max_len']
    NDATA = len(ss)
    DIM = len(devices)
    OH = np.zeros((NDATA, MAX_LEN, DIM))
    import random
    random.shuffle(ss)
    for setup in ss:
        indices = []
        for device in setup:
            indices.append(devices.index(device))
        if len(indices) < MAX_LEN:
            indices.extend((MAX_LEN-len(indices))*[DIM-1])
        OH[count,:,:] = many_one_hot(np.array(indices), DIM)
        count = count + 1

    h5f = h5py.File('qo_dataset.h5','w')
    h5f.create_dataset('data', data=OH)
    h5f.create_dataset('chr',  data=devices)
    h5f.close()

devices = ['BS(XXX,a,b)', 'BS(XXX,a,c)',
           'BS(XXX,a,d)', 'BS(XXX,a,e)',
           'BS(XXX,a,f)', 'BS(XXX,b,c)',
           'BS(XXX,b,d)', 'BS(XXX,b,e)',
           'BS(XXX,b,f)', 'BS(XXX,c,d)',
           'BS(XXX,c,e)', 'BS(XXX,c,f)',
           'BS(XXX,d,e)', 'BS(XXX,d,f)',
           'BS(XXX,e,f)',
           'DownConv(XXX,1,a,b)', 'DownConv(XXX,1,a,c)', 'DownConv(XXX,1,a,d)',
           'DownConv(XXX,1,a,e)', 'DownConv(XXX,1,a,f)', 'DownConv(XXX,1,b,c)',
           'DownConv(XXX,1,b,d)', 'DownConv(XXX,1,b,e)', 'DownConv(XXX,1,b,f)',
           'DownConv(XXX,1,c,d)', 'DownConv(XXX,1,c,e)', 'DownConv(XXX,1,c,f)',
           'DownConv(XXX,1,d,e)', 'DownConv(XXX,1,d,f)', 'DownConv(XXX,1,e,f)',
           'Reflection(XXX,a)', 'DP(XXX,a)',
           'OAMHolo(XXX,a,1)', 'OAMHolo(XXX,a,2)', 'OAMHolo(XXX,a,3)', 'OAMHolo(XXX,a,4)', 'OAMHolo(XXX,a,5)',
           'OAMHolo(XXX,a,-1)', 'OAMHolo(XXX,a,-2)', 'OAMHolo(XXX,a,-3)', 'OAMHolo(XXX,a,-4)', 'OAMHolo(XXX,a,-5)',
           'Reflection(XXX,b)', 'DP(XXX,b)',
           'OAMHolo(XXX,b,1)', 'OAMHolo(XXX,b,2)', 'OAMHolo(XXX,b,3)', 'OAMHolo(XXX,b,4)', 'OAMHolo(XXX,b,5)',
           'OAMHolo(XXX,b,-1)', 'OAMHolo(XXX,b,-2)', 'OAMHolo(XXX,b,-3)', 'OAMHolo(XXX,b,-4)', 'OAMHolo(XXX,b,-5)',
           'Reflection(XXX,c)', 'DP(XXX,c)',
           'OAMHolo(XXX,c,1)', 'OAMHolo(XXX,c,2)', 'OAMHolo(XXX,c,3)', 'OAMHolo(XXX,c,4)', 'OAMHolo(XXX,c,5)',
           'OAMHolo(XXX,c,-1)', 'OAMHolo(XXX,c,-2)', 'OAMHolo(XXX,c,-3)', 'OAMHolo(XXX,c,-4)', 'OAMHolo(XXX,c,-5)',
           'Reflection(XXX,d)', 'DP(XXX,d)',
           'OAMHolo(XXX,d,1)', 'OAMHolo(XXX,d,2)', 'OAMHolo(XXX,d,3)', 'OAMHolo(XXX,d,4)', 'OAMHolo(XXX,d,5)',
           'OAMHolo(XXX,d,-1)', 'OAMHolo(XXX,d,-2)', 'OAMHolo(XXX,d,-3)', 'OAMHolo(XXX,d,-4)', 'OAMHolo(XXX,d,-5)',
           'Reflection(XXX,e)', 'DP(XXX,e)',
           'OAMHolo(XXX,e,1)', 'OAMHolo(XXX,e,2)', 'OAMHolo(XXX,e,3)', 'OAMHolo(XXX,e,4)', 'OAMHolo(XXX,e,5)',
           'OAMHolo(XXX,e,-1)', 'OAMHolo(XXX,e,-2)', 'OAMHolo(XXX,e,-3)', 'OAMHolo(XXX,e,-4)', 'OAMHolo(XXX,e,-5)',
           'Reflection(XXX,f)', 'DP(XXX,f)',
           'OAMHolo(XXX,f,1)', 'OAMHolo(XXX,f,2)', 'OAMHolo(XXX,f,3)', 'OAMHolo(XXX,f,4)', 'OAMHolo(XXX,f,5)',
           'OAMHolo(XXX,f,-1)', 'OAMHolo(XXX,f,-2)', 'OAMHolo(XXX,f,-3)', 'OAMHolo(XXX,f,-4)', 'OAMHolo(XXX,f,-5)',
            ' ']

if __name__ == '__main__':
    es = np.array(load_entropy())
    idx = np.argsort(es)
    print(es[idx][-80:-1])
