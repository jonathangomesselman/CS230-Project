import os, argparse
import numpy as np
import h5py
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 
import librosa
from scipy import interpolate
from scipy.signal import decimate


filename_train = "vctk-speaker1-train.4.16000.8192.4096.h5"
filename_dev = "vctk-speaker1-val.4.16000.8192.4096.h5"


f_train = h5py.File(filename_train, 'r')
group_keys_train = list(f_train.keys())
inputs_train = np.array(f_train[group_keys_train[0]])
labels_train = np.array(f_train[group_keys_train[1]])


f_dev = h5py.File(filename_dev, 'r')
group_keys_dev = list(f_dev.keys())
inputs_dev = np.array(f_dev[group_keys_dev[0]])
labels_dev = np.array(f_dev[group_keys_dev[1]])


print "Train inputs shape: " + str(inputs_train.shape)
print "Train labels shape: " + str(labels_train.shape)


print "Dev inputs shape: " + str(inputs_dev.shape)
print "Dev labels shape: " + str(labels_dev.shape)

print "Train input example: "
print inputs_train[8][:][:]

print "Train label example: "
print labels_train[8][:][:]

print "L2 resolution distance for this example: " + str(np.linalg.norm(inputs_train[8][:][:]-labels_train[8][:][:]))

#train data statistics

print "Train data statistics: "

L2_distances_train = []
for i in range(len(inputs_train)):
	curr_input = inputs_train[i][:][:]  
	curr_label = labels_train[i][:][:]
	L2_distances_train.append(float(np.linalg.norm(curr_input-curr_label)))
print "Average L2 resolution distance for training set: " + str(sum(L2_distances_train)/float(len(L2_distances_train)))
print "Max L2 resolution distance for training set: " + str(max(L2_distances_train))
print "Min L2 resolution distance for training set: " + str(min(L2_distances_train))

#Dev data statistics

print "Dev data statistics: "

L2_distances_dev = []
for i in range(len(inputs_dev)):
	curr_input = inputs_dev[i][:][:]  
	curr_label = labels_dev[i][:][:]
	L2_distances_dev.append(float(np.linalg.norm(curr_input-curr_label)))

print "Average L2 resolution distance for training set: " + str(sum(L2_distances_dev)/float(len(L2_distances_dev)))
print "Max L2 resolution distance for training set: " + str(max(L2_distances_dev))
print "Min L2 resolution distance for training set: " + str(min(L2_distances_dev))


#h5disp('example.h5')