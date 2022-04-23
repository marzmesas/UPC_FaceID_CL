
import os
from os import getcwd
import sys
# print(sys.path)
# print(sys.path)

count = 0
for f in os.listdir("./Datasets/cropped-imgs1-supervised-test"):
	count +=1

print(count)