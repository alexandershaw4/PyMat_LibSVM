#! /usr/bin/env python

from svmutil import *
import sys
# wrapper code by alex - for interfacing with matlab
# pass entry to split at for train vs predict, e.g.
# ./DoPyTrain.py 40 > output

splt = sys.argv[1]
#print(splt)
splt = int(splt)

# Read data in LIBSVM format
y, x = svm_read_problem('formd')
m = svm_train(y[:splt], x[:splt], '-c 4')
p_label, p_acc, p_val = svm_predict(y[splt:], x[splt:], m)

print(p_label) #push these into output file, in case wanted
print(p_acc)
print(p_val)