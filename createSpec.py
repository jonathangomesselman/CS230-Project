from scipy.io import wavfile
import matplotlib.pyplot as plt

def main():
	samplingFreq, signalData = wavfile.read('./TestGeneratorInput/p225_366.wav.singlespeaker-out.lr.wav')
	plt.title('Test Spectogram')
	plt.specgram(signalData, Fs=samplingFreq)
	plt.show()



if __name__ == '__main__':
 	main()