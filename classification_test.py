from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.layers import Softmax
import numpy as np

N = 6   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
ALPHABET_SIZE = 4
MYN = ALPHABET_SIZE*int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.0001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions =1000 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

n_actions = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
			  #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
			  
observation_space = 2*MYN #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
						  #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
						  #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
						  #Is there a better way to format the input to make it easier for the neural network to understand things?


						  
len_game = MYN 
state_dim = (observation_space,)

INF = 1000000

model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(ALPHABET_SIZE, activation="sigmoid"))
model.add(Softmax())

model.build((None, observation_space))
model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate = LEARNING_RATE)) #Adam optimizer also works well, with lower learning rate

X = np.zeros((3, observation_space))
X[0][2] = 1
X[1][3] = 1
X[2][4] = 1
Y = np.array([[0.5, 0.5, 0, 0],[0, 0, 0.5, 0.5],[0, 0.5, 0, 0.5]])
model.fit(X,Y)
pred = model.predict(X)
print("pred:", pred)