import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import sys
import numpy as np

import librosa

from matplotlib import pyplot as plt

N_FFT = 1024

# ----------------------------------------------------------------------------

def get_spectrum_from_file(filename, n_fft=N_FFT):
  x, fs = librosa.load(filename)
  return get_spectrum(x, n_fft)

def get_spectrum(x, n_fft=N_FFT):    
  S = librosa.stft(x, n_fft)
  p = np.angle(S)
  
  S = np.log1p(np.abs(S))  
  return S

def get_power(x, n_fft=N_FFT):    
  S = librosa.stft(x, n_fft)
  p = np.angle(S)
  
  S = np.log(np.abs(S)**2 + 1e-8)  
  return S  

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
  plt.imshow(S.T, aspect=1)
  # plt.xlim([0,lim])
  plt.tight_layout()
  plt.savefig(outfile)

if __name__ == '__main__':
  fname = sys.argv[1]
  S = get_spectrum_from_file(fname)
  save_spectrum(S)
