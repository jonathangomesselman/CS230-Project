import argparse
import os
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import librosa
import numpy as np
import pandas as pd

# Adds appropriate argument parsers
def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')
  
  process_parser = subparsers.add_parser('process')
  process_parser.set_defaults(func=process)
  process_parser.add_argument('--wav-file-list', 
    help='list of audio files for evaluation')
  process_parser.add_argument('--r', help='upscaling factor', type=int)
  process_parser.add_argument('--sr', help='high-res sampling rate', 
                                   type=int, default=16000)

  return parser


# Processes wav file into a vector with a y label of 1 for ground truth 
# and 0 for generated example

''' run the following command to turn the data into a numpy array

python processWav.py process \
  --wav-file-list ../data/vctk/speaker1/testDataFormat.txt \
  --r 4
'''
def process(args):
  standardsize = 100000
  labelcounter = 0
  ylabels = []
  xlist = []
  if args.wav_file_list:
    with open(args.wav_file_list) as f:
      for line in f:
        try:
          line = "ProcessedDataRate=4/" + line
          line = line.strip()
          print line.strip()
          x_input, fs = librosa.load(line, sr=args.sr)
          
          if (x_input.shape[0] > standardsize):
          	x_input = x_input[:standardsize]
          else:
          	x_input = np.reshape(x_input, (-1, 1))
          	pad = np.zeros((standardsize - x_input.shape[0], 1))
          	x_input = np.concatenate((x_input, pad), axis = 0)
          #print(x_input.shape)
          #print(x_input)
          xlist.append(x_input.T)
          # this means we are dealing with a highly resolved wav file
          if labelcounter % 2 == 0:
            ylabels.append(1)
          # else we are dealing with a generated wav file
          else:
          	ylabels.append(0)
          
          labelcounter += 1
       	except EOFError:
          print 'WARNING: Error reading file:', line.strip()
  #print(ylabels)
  xlist = np.vstack(xlist)
  ylabels = np.asarray(ylabels)
  ylabels = np.reshape(ylabels, (-1, 1))
  finallist = np.hstack((xlist, ylabels))
  final = pd.DataFrame(finallist)
  final.to_csv('trainData.csv', header = None)

  print(ylabels.shape)
  print(xlist.shape)
  print(finallist.shape)
  print(finallist[0])
def main():
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)

if __name__ == '__main__':
  main()