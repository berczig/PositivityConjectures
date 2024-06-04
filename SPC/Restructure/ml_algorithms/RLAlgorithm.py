from SPC.Restructure.ml_algorithms.LearningAlgorithm import LearningAlgorithm

import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc
import random
import numpy as np
from keras.utils import to_categorical
from statistics import mean
import pickle
import time
import math
import os
import matplotlib.pyplot as plt
import sys
from SPC.Restructure.UIO import UIO
from SPC.misc.extra import PartiallyLoadable
from datetime import datetime
from SPC.Restructure.FilterEvaluator import FilterEvaluator
from SPC.Restructure.ml_models.RLNNModel_CorrectSequence import RLNNModel_CorrectSequence

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # fix to omp: error #15 on my laptop

class RLAlgorithm(LearningAlgorithm):

    # PARAMETERS
    n_sessions = 300 #number of new sessions per iteration
    percentile = 93 #top 100-X percentile we are learning from
    super_percentile = 94 #top 100-X percentile that survives to next iteration
    reduce_uio = 0
    INF = 1000000
	#########################
 

    def train(self, iterations, model_save_path="", model_save_time=0):
        self.model : RLNNModel_CorrectSequence
        self.model = self.model_logger.get_model()

        self.FE = FilterEvaluator(self.trainingdata_input, self.trainingdata_output, FilterEvaluator.DEFAULT_IGNORE_VALUE, self.model.CORE_LENGTH, self.model_logger)

        startstep = self.model_logger.step
        print("startstep:", startstep)

        print("the \"only zero state\" has a reward of", self.calcScore(np.zeros(self.model.observation_space)))

        super_states =  np.empty((0,self.model.len_game,self.model.observation_space), dtype = int)
        super_actions = np.array([], dtype = int)
        super_rewards = np.array([])
        sessgen_time = 0
        fit_time = 0
        score_time = 0
        next_save_time = time.time() + model_save_time


        myRand = random.randint(0,1000) #used in the filename

        for i in range(startstep, startstep+iterations): #1000000 generations should be plenty
            #generate new sessions
            #performance can be improved with joblib
            tic0 = time.time()
            tic = time.time()
            sessions = self.generate_session(self.model,self.n_sessions,0) #change 0 to 1 to print out how much time each step in generate_session takes 
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

            elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch, percentile=self.percentile) #pick the sessions to learn from
            select1_time = time.time()-tic

            tic = time.time()
            super_sessions = self.select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=self.super_percentile) #pick the sessions to survive
            select2_time = time.time()-tic
            
            tic = time.time()
            super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
            super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
            select3_time = time.time()-tic
            
            tic = time.time()
            print("elite_states:", elite_states.shape)
            print("elite_actions:", elite_actions.shape)
            self.model.fit(elite_states, elite_actions) #learn from the elite sessions
            fit_time = time.time()-tic
            
            tic = time.time()
            
            print("super_sessions:", len(super_sessions))
            super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
            super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
            super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
            
            rewards_batch.sort()
            mean_all_reward = np.mean(rewards_batch[-100:])	
            mean_best_reward = np.mean(super_rewards)	

            score_time = time.time()-tic
            
            print("all scores:", len(self.model_logger.all_scores))
            print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
            print(self.FE.convertConditionMatrixToText(self.convertStateToConditionMatrix(self.getbeststate())))
            self.model_logger.bestscore_history.append(super_rewards[0])
            self.model_logger.meanscore_history.append(np.mean(super_rewards))
            self.model_logger.numgraph_history.append(len(self.model_logger.all_scores))
            self.model_logger.calculationtime_history.append(time.time()-tic0)
            
            #uncomment below line to print out how much time each step in this loop takes. 
            print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
            
            """
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
            """

            if model_save_path != "":
                if time.time() > next_save_time:
                    print("#"*100)
                    print("saving at ", datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %I:%M:%S"))
                    print("saving model...")
                    self.model_logger.step = i+1
                    self.model_logger.save_model_logger(model_save_path)
                    print("done saving at ", datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %I:%M:%S"))
                    next_save_time = time.time() + model_save_time
        self.model_logger.step += iterations

			


    def convertStateToConditionMatrix(self, state):
        # state is of length MYN
        columns = int(self.model.CORE_LENGTH * (self.model.CORE_LENGTH-1) / 2)
        graph = np.ones((self.model.ROWS_IN_CONDITIONMATRIX, columns))*self.FE.ignore_edge
        for i in range(self.model.ROWS_IN_CONDITIONMATRIX):
            for j in range(columns):
                actionvector = state[self.model.ALPHABET_SIZE*(i*columns + j) : self.model.ALPHABET_SIZE*(i*columns + j+1)]
                argmax = np.argmax(actionvector)
                # argmax = 0: either because all zero (no edgetype set yet) -> ignore edge ||| or because the first element is 1, meaning the "ignore edge" type was choosen
                if argmax != 0: 
                    graph[i][j] = argmax + UIO.LESS - 1
        #print("graph:", graph)
        return graph

    def calcScore(self, state):
        key_state = tuple(state)
        #print("all_scores:", len(all_scores))
        if key_state in self.model_logger.all_scores:
            return self.model_logger.all_scores[key_state]
        else:
            new_score = self.FE.evaluate(self.convertStateToConditionMatrix(state), False)
            self.model_logger.all_scores[key_state] = new_score
            return new_score


    ####No need to change anything below here. 
        
        
                            

    def generate_session(self, agent, n_sessions, verbose = 1):
        """
        Play n_session games using agent neural network.
        Terminate when games finish 
        
        Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
        """
        print("generate_session")
        states =  np.zeros([n_sessions, self.model.observation_space, self.model.len_game], dtype=int)
        actions = np.zeros([n_sessions, self.model.len_game, self.model.ALPHABET_SIZE], dtype = int)
        state_next = np.zeros([n_sessions,self.model.observation_space], dtype = int)
        prob = np.zeros(n_sessions)
        states[:,self.model.MYN,0] = 1
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

            terminal = step == self.model.EDGES
            
            #print(step, "over_conditioned_graphs:", len(over_conditioned_graphs))

            #print("first prob:", prob[0])
            for i in range(n_sessions):
                #print("i:", step, i)
                #t0 = time.time()
                if i in over_conditioned_graphs: # even doing nothing is represented by a non-zero vector(1 hot encoding)
                    actions[i][step-1][0] = 1 # [1,0,0,0] is do nothing, encoding  as in calcScore
                    continue

                vectoraction = np.random.multinomial(1, prob[i], size=1).reshape(self.model.ALPHABET_SIZE)

                if vectoraction[0] != 1:
                    edges[i] += 1
                        
                #print("prob:", prob[i])
                #print("vectoraction:", vectoraction)
                actions[i][step-1] = vectoraction
                tic = time.time()
                state_next[i] = states[i,:,step-1]
                play_time += time.time()-tic

                state_next[i][self.model.ALPHABET_SIZE*(step-1):self.model.ALPHABET_SIZE*step] = vectoraction

                state_next[i][self.model.MYN + step-1] = 0

                #t = time.time()
                score = self.calcScore(state_next[i]) # actually we only use the first of the vector to calculate the score
                #t1 = time.time()-t
                #print("score:", score)
                tic = time.time()
                if terminal:
                    total_score[i] = score
                scorecalc_time += time.time()-tic
                tic = time.time()
                #print(edges[i])
                if score == -FilterEvaluator.INF or edges[i] > self.model.MAX_EXPECTED_EDGES:
                    total_score[i] = self.calcScore(states[i, :, step-1]) # take score of not over conditioned graph
                    over_conditioned_graphs.append(i)
                elif not terminal:
                    state_next[i][self.model.MYN + step] = 1
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



    def select_elites(self, states_batch, actions_batch, rewards_batch, percentile=50):
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
        counter = self.n_sessions * (100.0 - percentile) / 100.0
        reward_threshold = np.percentile(rewards_batch,percentile)
        elite_states = []
        elite_actions = []
        elite_rewards = []
        for i in range(len(states_batch)):
            #print("i:", i)
            if rewards_batch[i] >= reward_threshold-0.0001:		
                if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0001):
                    #print("yo:", len(states_batch[i]))
                    for item in states_batch[i]:
                        elite_states.append(item.tolist())
                    for item in actions_batch[i]: #### TODO step size ALPHABET_SIZE ####
                        elite_actions.append(item)			
                counter -= 1
        elite_states = np.array(elite_states, dtype = int)	
        elite_actions = np.array(elite_actions, dtype = int)	
        return elite_states, elite_actions
        
    def select_super_sessions(self, states_batch, actions_batch, rewards_batch, percentile=90):
        """
        Select all the sessions that will survive to the next generation
        Similar to select_elites function
        If this function is the bottleneck, it can easily be sped up using numba
        """
        
        counter = self.n_sessions * (100.0 - percentile) / 100.0
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
    
    def getbeststate(self):
        bestscore = -99999999999
        beststate = None
        for state in self.model_logger.all_scores:
            if self.model_logger.all_scores[state] > bestscore:
                bestscore = self.model_logger.all_scores[state]
                beststate = state
        return beststate