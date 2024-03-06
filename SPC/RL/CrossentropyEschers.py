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
from SPC.UIOs.uionew import UIO, ConditionEvaluator, UIODataExtractor
from SPC.Transformers.extra import PartiallyLoadable
from datetime import datetime


#This file is a modification of the CrossentropyCorrectSeq.py file. 
#We use a graph crossentropy RL method to learn condition graphs for (l,k,p) Eschers.
#The count of Escher triples satisfying the conditions encoded by this graph is the Stanley coefficient (l,k,p)
l = 4
k = 2
p = 1
NumCritPoints = 3+3+2  #pairwise insertion points, pairwise splitting points, n=l+k+p and n+l+k+p. 
                        #This is the number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
NUMBER_OF_ORS = 2 #The possible edge types, i.e possible relations of the critical points. It is < or > so their number is 2.
MAX_EXPECTED_EDGES = 3 #The maximum number of edges we expect to have in the graph. 
EDGES = int(NumCritPoints*(NumCritPoints-1)/2)
ALPHABET_SIZE = 1+NUMBER_OF_ORS*3 #The size of the alphabet. We have 1 for the empty word, and 3 for each edge type (>,<,=).
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

load_cores_file = "Saves,Tests/coreTypes_l={}_k={}_p={}_ignore=100.bin".format(l,k,p)
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
	print("state:", state.shape)
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
	print("graph:", graph.shape)
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



def generate_session(agent, n_sessions, verbose = 1):
	"""
	Play n_session games using agent neural network.
	Terminate when games finish 
	
	Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	print("generate_session")
	states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
	actions = np.zeros([n_sessions, len_game, ALPHABET_SIZE], dtype = int)
	state_next = np.zeros([n_sessions,observation_space], dtype = int)
	prob = np.zeros(n_sessions)
	states[:,MYN,0] = 1
	step = 0
	total_score = np.zeros([n_sessions])
	recordsess_time = 0
	play_time = 0
	scorecalc_time = 0
	pred_time = 0
	over_conditioned_graphs = [] # blacklist for graphs with -inf score
	edges = np.zeros(n_sessions)
	while (True):
		step += 1		
		tic = time.time()

		prob = agent.predict(states[:,:,step-1], batch_size = n_sessions, verbose=None)
		# np.random.multinomial implicitly casts to float64, this create the possibility that a row sums to > 1
		prob = np.float64(prob)
		prob = (prob.T / np.sum(prob, axis=1)).T # normalize each row
		
		pred_time += time.time()-tic

		terminal = step == EDGES
		
		#print(step, "over_conditioned_graphs:", len(over_conditioned_graphs))

		#print("first prob:", prob[0])
		for i in range(n_sessions):
			#print("i:", step, i)
			#t0 = time.time()
			if i in over_conditioned_graphs: # even doing nothing is represented by a non-zero vector(1 hot encoding)
				actions[i][step-1][0] = 1 # [1,0,0,0] is do nothing, encoding  as in calcScore
				continue

			vectoraction = np.random.multinomial(1, prob[i], size=1).reshape(ALPHABET_SIZE)

			if vectoraction[0] != 1:
				edges[i] += 1
					
			#print("prob:", prob[i])
			#print("vectoraction:", vectoraction)
			actions[i][step-1] = vectoraction
			tic = time.time()
			state_next[i] = states[i,:,step-1]
			play_time += time.time()-tic

			state_next[i][ALPHABET_SIZE*(step-1):ALPHABET_SIZE*step] = vectoraction

			state_next[i][MYN + step-1] = 0

			#t = time.time()
			score = calcScore(state_next[i]) # actually we only use the first of the vector to calculate the score
			#t1 = time.time()-t
			#print("score:", score)
			tic = time.time()
			if terminal:
				total_score[i] = score
			scorecalc_time += time.time()-tic
			tic = time.time()
			#print(edges[i])
			if score == -np.inf or edges[i] > MAX_EXPECTED_EDGES:
				total_score[i] = calcScore(states[i, :, step-1]) # take score of not over conditioned graph
				over_conditioned_graphs.append(i)
			elif not terminal:
				state_next[i][MYN + step] = 1
				states[i,:,step] = state_next[i]	# update graph with policy from network
			recordsess_time += time.time()-tic
			#t2 = time.time()-t0
			#if t2 != 0:
			#	print(100*t1/t2)
		
		if terminal:
			break
	#If you want, print out how much time each step has taken. This is useful to find the bottleneck in the program.		
	if (verbose):
		print("Predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time))
	return states, actions, total_score



def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
	"""
	Select states and actions from games that have rewards >= percentile
	:param states_batch: list of lists of states, states_batch[session_i][t]
	:param actions_batch: list of lists of actions, actions_batch[session_i][t]
	:param rewards_batch: list of rewards, rewards_batch[session_i]
	:returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
	
	This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	If this function is the bottleneck, it can easily be sped up using numba

	hard penalty: if more than 3 edges stop 
	"""
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)
	elite_states = []
	elite_actions = []
	elite_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:		
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				for item in states_batch[i]:
					elite_states.append(item.tolist())
				for item in actions_batch[i]: #### TODO step size ALPHABET_SIZE ####
					elite_actions.append(item)			
			counter -= 1
	elite_states = np.array(elite_states, dtype = int)	
	elite_actions = np.array(elite_actions, dtype = int)	
	return elite_states, elite_actions
	
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
	"""
	Select all the sessions that will survive to the next generation
	Similar to select_elites function
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	super_states = []
	super_actions = []
	super_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				super_states.append(states_batch[i])
				super_actions.append(actions_batch[i])
				super_rewards.append(rewards_batch[i])
				counter -= 1
	super_states = np.array(super_states, dtype = int)
	super_actions = np.array(super_actions, dtype = int)
	super_rewards = np.array(super_rewards)
	return super_states, super_actions, super_rewards

class DataSaver(PartiallyLoadable):

	def __init__(self, save_vars, load_model_file, CE):
		super().__init__(save_vars) # set saveable variables

		self.step = 0
		self.bestscore_history = [] # history of best score
		self.meanscore_history = [] # history of mean score
		self.numgraph_history = [] # history of number of graphs which we allready have calculated the score of
		self.calculationtime_history = []
		self.all_scores = {} # graph:score
		self.CE = CE

		if load_model_file == "":
			#Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
			#I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
			#It is important that the loss is binary cross-entropy if alphabet size is 2.
			self.model = Sequential()
			self.model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
			self.model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
			self.model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
			self.model.add(Dense(ALPHABET_SIZE, activation="softmax"))
			self.model.build((None, observation_space))
			self.model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate = LEARNING_RATE), run_eagerly=True) #Adam optimizer also works well, with lower learning rate
		else:
			self.load(load_model_file)
			self.make_plots()

		print(self.model.summary())


	def save(self, filename):
		super().save(filename+".mybin")
		print("saving model:", save_model_file)
		self.model.save(filename)

	def load(self, filename):
		super().load(filename+".mybin")
		print("step:::::::::::::", self._savehelper.step)
		print("loading model:", load_model_file)
		self.model = load_model(filename)

	def make_plots(self):
		n = len(self.bestscore_history)
		times = list(range(n))
		plt.title("best score ("+str(self.bestscore_history[-1])+")")
		plt.plot(times, self.bestscore_history)
		plt.show()
		plt.title("mean score ("+str(self.meanscore_history[-1])+")")
		plt.plot(times, self.meanscore_history)
		plt.show()
		plt.title("number of different conditions checked")
		plt.plot(times, self.numgraph_history)
		plt.show()
		plt.title("computation time of the i'th step")
		plt.plot(times, self.calculationtime_history)
		plt.show()

		print("looking for best score...")
		bestscore = -99999999999
		beststate = None
		for state in self.all_scores:
			if self.all_scores[state] > bestscore:
				bestscore = self.all_scores[state]
				beststate = state
		condmat = convertStateToConditionMatrix(beststate)
		conditiontext = self.CE.convertConditionMatrixToText(condmat)
		print(conditiontext, "\nhas a score of ", self.CE.evaluate(condmat))
	
if __name__ == "__main__":
	#CE = ConditionEvaluator(l=l, k=k, p=p, ignoreEdge=UIO.INCOMPARABLE)
	CE = None
	if load_cores_file == "":
		CE = ConditionEvaluator(l=l, k=k, p=p, ignoreEdge=UIO.INCOMPARABLE, uiodataextractor=UIODataExtractor(l,k,p))
	else:
		CE = ConditionEvaluator(l=l, k=k, p=p, ignoreEdge=UIO.INCOMPARABLE)
		CE.load(load_cores_file)
	if reduce_uio != 0:
		CE.narrowCoreTypeSelection(list(range(reduce_uio)))

	DS = DataSaver(["step", "bestscore_history", "meanscore_history", "numgraph_history", 
	"calculationtime_history", "all_scores"],
		 load_model_file, CE)
	model = DS.model
	startstep = DS.step
	all_scores = DS.all_scores

	print("only zero state has reward of", calcScore(np.zeros(observation_space)))


	super_states =  np.empty((0,len_game,observation_space), dtype = int)
	super_actions = np.array([], dtype = int)
	super_rewards = np.array([])
	sessgen_time = 0
	fit_time = 0
	score_time = 0
	next_save_time = time.time() + saving_frequency



	myRand = random.randint(0,1000) #used in the filename

	for i in range(startstep, INF): #1000000 generations should be plenty
		#generate new sessions
		#performance can be improved with joblib
		tic0 = time.time()
		tic = time.time()
		sessions = generate_session(model,n_sessions,0) #change 0 to 1 to print out how much time each step in generate_session takes 
		sessgen_time = time.time()-tic
		tic = time.time()
		
		states_batch = np.array(sessions[0], dtype = int)
		actions_batch = np.array(sessions[1], dtype = int)
		rewards_batch = np.array(sessions[2])
		states_batch = np.transpose(states_batch,axes=[0,2,1])
		states_batch = np.append(states_batch,super_states,axis=0)

		if i > startstep:
			actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	
		rewards_batch = np.append(rewards_batch,super_rewards)
			
		randomcomp_time = time.time()-tic 
		tic = time.time()

		elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
		select1_time = time.time()-tic

		tic = time.time()
		super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
		select2_time = time.time()-tic
		
		tic = time.time()
		super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
		super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
		select3_time = time.time()-tic
		
		tic = time.time()
		print("elite_states:", elite_states.shape)
		print("elite_actions:", elite_actions.shape)
		model.fit(elite_states, elite_actions) #learn from the elite sessions
		fit_time = time.time()-tic
		
		tic = time.time()
		
		super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
		super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
		super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
		
		rewards_batch.sort()
		mean_all_reward = np.mean(rewards_batch[-100:])	
		mean_best_reward = np.mean(super_rewards)	

		score_time = time.time()-tic
		
		print("all scores:", len(all_scores))
		print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
		DS.bestscore_history.append(super_rewards[0])
		DS.meanscore_history.append(np.mean(super_rewards))
		DS.numgraph_history.append(len(all_scores))
		DS.calculationtime_history.append(time.time()-tic0)
		
		#uncomment below line to print out how much time each step in this loop takes. 
		print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
		
		
		if (i%20 == 1): #Write all important info to files every 20 iterations
			with open('best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
				pickle.dump(super_actions, fp)
			with open('best_species_txt_'+str(myRand)+'.txt', 'w') as f:
				for item in super_actions:
					f.write(str(item))
					f.write("\n")
			with open('best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
				for item in super_rewards:
					f.write(str(item))
					f.write("\n")
			with open('best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
				f.write(str(mean_all_reward)+"\n")
			with open('best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
				f.write(str(mean_best_reward)+"\n")
		if (i%200==2): # To create a timeline, like in Figure 3
			with open('best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
				f.write(str(super_actions[0]))
				f.write("\n")

		if time.time() > next_save_time:
			print("#"*500)
			print("saving at ", datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %I:%M:%S"))
			DS.step = i+1
			DS.all_scores = all_scores
			if save_model_file != "":
				DS.save(save_model_file)
				#DS.make_plots()
			print("done saving at ", datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %I:%M:%S"))
			next_save_time = time.time() + saving_frequency