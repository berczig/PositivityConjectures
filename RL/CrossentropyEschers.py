import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc
import random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import load_model
from statistics import mean
import pickle
import time
import math
import os
import matplotlib.pyplot as plt
import sys
from uio import UIO, ConditionEvaluator, UIODataExtractor
from extra import PartiallyLoadable
from datetime import datetime


#This file is a modification of the CrossentropyCorrectSeq.py file. 
#We use a graph crossentropy RL method to learn condition graphs for (l,k,p) Eschers.
#The count of Escher triples satisfying the conditions encoded by this graph is the Stanley coefficient (l,k,p)
l = 3
k = 2
p = 1
NumCritPoints = 3+3+2  #pairwise insertion points, pairwise splitting points, n=l+k+p and n+l+k+p. 
                        #This is the number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
NUMBER_OF_ORS = 2 #The possible edge types, i.e possible relations of the critical points. It is < or > so their number is 2.
MAX_EXPECTED_EDGES = 3
ALPHABET_SIZE = 1+NUMBER_OF_ORS*3
EDGES = int(NumCritPoints*(NumCritPoints-1)/2)
MYN = ALPHABET_SIZE*EDGES  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.1 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions = 300 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

n_actions = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
			  #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
			  
observation_space = MYN + EDGES #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
						  #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
						  #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
						  #Is there a better way to format the input to make it easier for the neural network to understand things?


						  
len_game = EDGES 
state_dim = (observation_space,)

load_cores_file = "saves/coreTypes_l={}_k={}_p={}_ignore=100.bin".format(l,k,p)
load_model_file = "Master42example" #"saves/170uio2ORs" # "saves/190uio" # "saves/150"
save_model_file = "Master42example"#"saves/170uio2ORs"
reduce_uio = 0

# got  150 uios down to score=0 in 2 steps (300 graphs, 0.1 learning rate)
# got  80 uios down to score=0 in 4 steps 

if load_model_file != "":
	load_model_file += "[l={}k={}p={}]".format(l,k,p)
if save_model_file != "":
	save_model_file += "[l={}k={}p={}]".format(l,k,p)
saving_frequency = 30 # how many seconds to wait between each save
INF = 1000000

def convertStateToConditionMatrix(state):
	# state is of length MYN
	graph = np.ones((NUMBER_OF_ORS, EDGES))*UIO.INCOMPARABLE
	for step in range(EDGES):
		actionvector = state[ALPHABET_SIZE*step:ALPHABET_SIZE*(step+1)]
		if actionvector[0] == 0: # if 1 in 0'th index then do nothing (UIO.INCOMPARABLE)
			row = 0
			edge = np.argmax(actionvector) # 100,101,102,103
			if edge == 0: # actionvector all zeros
				continue
			row = (edge-1)//3
			if row != 0:
				edge -= 3*row
			graph[row][step] = edge + UIO.INCOMPARABLE
	return graph

def calcScore(state):
	global all_scores
	key_state = tuple(state)
	#print("all_scores:", len(all_scores))
	if key_state in all_scores:
		return all_scores[key_state]
	else:
		new_score = CE.evaluate(convertStateToConditionMatrix(state), False)
		all_scores[key_state] = new_score
		return new_score
