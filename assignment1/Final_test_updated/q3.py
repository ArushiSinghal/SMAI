import sys
import numpy as np

train = sys.argv[1]
test = sys.argv[2]

classes = ["0", "1"]

with open(test) as f:
    data = f.readlines()
    for line in data:
        print classes[np.random.randint(0,2)]
