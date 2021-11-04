import sys, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral' # 'cmu serif'#
elev, az = 50,315

def plot_latent(X, y, title='', size =1.01, k=2, c="viridis"):
    if 'small' in title:
        size = 5.
    else:
        size = 0.1

    if k==2:
        plt.scatter(X[:, 0], X[:, 1], s=size, c=y, marker ='o', cmap=plt.cm.get_cmap(c))
        plt.gca().set_xticks([])
        cbar = plt.colorbar()
        cbar.set_label('total entropy')
        #plt.colorbar()
        #plt.savefig('exp/2d'+title+'.pdf')
        plt.axis('off')
        plt.savefig('2dlatent'+title+'.SVG')
        plt.savefig('2dlatent'+title+'.png')
        plt.clf()
    else:
        print('need 2d latent space')
