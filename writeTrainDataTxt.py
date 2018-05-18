import os
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

# Writes strings to text files in order to construct training data for discriminator
def main():
  fromfile = open("../data/vctk/speaker1/speaker1-train-files.txt", "r")
  trainfile = open("../data/vctk/speaker1/testDataFormat.txt", "w")
  for line in fromfile:
  	hrline = str(line.strip()) + ".singlespeaker-out.hr.wav\n"
  	prline = str(line.strip()) + ".singlespeaker-out.pr.wav\n"
  	#print(hrline)
  	trainfile.write(hrline)
  	trainfile.write(prline)
  fromfile.close()
  trainfile.close()
  

if __name__ == '__main__':
  main()