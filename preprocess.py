import numpy as np
import pandas as pd
import glob
import csv
import librosa
import data
import os
import subprocess


# data path
_data_path = "data/"


#
# process training data
#

def process(csv_file):



    #init csv writer
    writer= csv.writer(csv_file, delimiter=',')

    #read label
    labels=[d for d in os.listdir(_data_path) if os.path.isdir(os.path.join(_data_path, d))]

    for label in labels:
        #get all file names
        files = [f for f in os.listdir(_data_path+label) if os.path.isfile(os.path.join(_data_path+label, f))]

        #print info
        for file in files:
            print("Processing file "+_data_path+label+'/'+file)

            #skip processed files
            target_filename='training/preprocess/mfcc/'+file+ '.npy'
            if os.path.exists(target_filename) or os.path.splitext(file)[1]!=".wav":
                continue
            #load wave file
            wave_file=_data_path+label+'/'+file
            wave,sr= librosa.load(wave_file, mono=True, sr=16000) #resample to 16k

            #get mfcc
            mfcc= librosa.feature.mfcc(wave, sr=16000)

            #get label byte
            labelByte= data.label_to_byte(label)
            print (labelByte)

            #save meta info
            writer.writerow([file] + [labelByte])

            #save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)







#
# Create directories
#
if not os.path.exists('training/preprocess'):
    os.makedirs('training/preprocess')
if not os.path.exists('training/preprocess/meta'):
    os.makedirs('training/preprocess/meta')
if not os.path.exists('training/preprocess/mfcc'):
    os.makedirs('training/preprocess/mfcc')


#
# Run pre-processing for training
#

csv_f = open('training/preprocess/train.csv', 'w')
process(csv_f)
csv_f.close()