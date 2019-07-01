from multi_class_perceptron import *
from corpus import *
import time

#Instantiating objects of multi_class_perceptron
t = multi_class_perceptron()
p = multi_class_perceptron()
v = multi_class_perceptron()

# Prompt the user to enter the number of iteration for perceptron learning
iterarion = input('Enter number of iterarion(s): ')

#The main training procedure for perceptron classifier
p.train("Input_Files\\train.col", iterarion)
#Tagging the corpus after each iteration, base on models that have been dumped in memory from training fuction
t.tagger("Input_Files\\test.col", iterarion)

# start = time.time()5
# #Tagging the corpus based on viterbi perceptron
# v.viterbi_tagger("Input_Files\\test.col")
# end = time.time()
# print "HMM Tagger runs at:", end - start, "seconds"

#v.viterbi_smoothing()
