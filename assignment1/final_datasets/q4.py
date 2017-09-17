#!/usr/bin/env python

import sys
import os
import numpy as np


classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']

if __name__ == '__main__':
	testdir = sys.argv[1]
	inputdir = [testdir]

	for idir in inputdir:
		for c in classes:
			listing = os.listdir(idir+c)
			for filename in listing:
				print(c)
