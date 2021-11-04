import models.model_qo
from data_utils import devices
import numpy as np

class QoCharacterModel(object):
    def __init__(self, weights_file, hypers= {'max_len':16, 'latent':5, 'epochs': 250,
                                             'batch':256, 'load':None,
                                             'hidden': 64, 'dense': 64,
                                             'conv1': 4, 'conv2': 5, 'conv3': 6}):
        self._model = models.model_qo
        self.MAX_LEN = hypers['max_len']
        self.latent_size = hypers['latent']
        self.vae = self._model.QoVAE()
        self.charlist = devices
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        #print(self._char_index)
        self.vae.load(self.charlist, weights_file, hypers)

    def encode(self, setups):
        """Encode a list of setups (as list of device strings) into the latent space """
        #   setup ['device', ... ]
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in setups]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in xrange(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return self.vae.encoderMV.predict(one_hot)[0]

    def decode(self, z, probabilistic=False, return_lists=True):
        """ Sample from the decoder """
        assert z.ndim == 2
        out = self.vae.decoder.predict(z)
        noise = np.random.gumbel(size=out.shape)
        sampled_chars = np.argmax(np.log(out) + noise, axis=-1)
        if not probabilistic:
            sampled_chars = np.argmax(out, axis=-1)
        char_matrix = np.array(self.charlist)[np.array(sampled_chars, dtype=int)]
        #print(char_matrix)#; exit()
        samples = ['.'.join(ch).strip() for ch in char_matrix] #; print(samples)
        setups = [[dev for dev in setup.split('.') if dev != ' ' and dev != ''] for setup in samples] # list of list of devices
        strings = ['.'.join(setup).strip() for setup in setups] #; print(strings)
        if return_lists:
            return setups # list of list of setups  ['device', ... ]
        else:
            return strings # list of strings ['Device.device...', ...]
