import numpy as np
import csv
import string


# default data path
_data_path = 'data/'

#
# load all labels
#

#prob' write a function to autogenerate labels later
f = open(_data_path+'labels.txt', 'r')

#map labels to byte
index2byte = f.read().splitlines()

# byte to index mapping

linearByteArray=np.arange(len(index2byte))

byte2index= dict(zip(index2byte,linearByteArray))

# vocabulary size
voca_size = len(index2byte)

def label_to_byte(label):
        return byte2index[label]

# wave to mfcc conversion function
def _load_mfcc(src_list):

    # label, wave_file
    label, mfcc_file = src_list

    # decode string to integer
    label = np.fromstring(label, np.int)

    # load mfcc
    mfcc = np.load(mfcc_file, allow_pickle=False)

    # speed perturbation augmenting

    return label, mfcc



