import h5py
import numpy as np
from scipy.signal import decimate

r = 4

def load_h5(h5_path):
  # load training data
  with h5py.File(h5_path, 'r') as hf:
    print 'List of arrays in input file:', hf.keys()
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    # really sketchy, but begins the concatenation so that vstack can be done later
    x_hr0 = Y[0]
    x_hr1 = Y[1]
    x_lr0 = decimate(x_hr0, r)
    x_lr1 = decimate(x_hr1, r)
    X_hr = np.vstack((x_hr0, x_hr1))
    X_lr = np.vstack((x_lr0, x_lr1))

    for i in range(2, Y.shape[0]):
        newx_hr = Y[i]
        newx_lr = decimate(newx_hr, r)
        X_hr = np.vstack((X_hr, newx_hr))
        X_lr = np.vstack((X_lr, newx_lr))
    print(X_hr.shape)
    print(X_lr.shape)



    print 'Shape of X:', X.shape
    print 'Shape of Y:', Y.shape

def main():
    load_h5('./vctk-speaker1-train.4.16000.8192.4096.h5')

if __name__ == '__main__':
    main()