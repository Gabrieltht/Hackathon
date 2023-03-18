from SVM import SVM_Classification

import time

while True:
    try:
        f = open("test.txt", "r")
    except FileNotFoundError:
        print("NotFound")
        time.sleep(5)
    else:
        print("Found the File")
        inputFile = f.read(999)
        print("The message I am getting was " + inputFile)
        print(SVM_Classification(inputFile))

        break
