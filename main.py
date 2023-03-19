from SVM import SVM_Classification

import os
import time

while True:
    try:
        f = open("D:\\Users\\12463\\Downloads\\formData.txt", "r")
    except FileNotFoundError:
        print("NotFound")
        time.sleep(5)
    else:
        print("Found the File")
        inputFile = f.read(999)
        f.close()
        os.remove("D:\\Users\\12463\\Downloads\\formData.txt")
        print("The message I am getting was " + inputFile)
        output = SVM_Classification(inputFile)
        if output == 1:
            print("Depressed")
        else:
            print("Not depressed")

