from SPC.Restructure.ml_models.MLModel import MLModel
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

class RLNNModel(MLModel):

    l = 4
    k = 2
    p = 1
    # k+2+2*p
    NumCritIntervals = k+2+2*p   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
    NUMBER_OF_ORS = 2
    ALPHABET_SIZE = 1+NUMBER_OF_ORS*3
    EDGES = int(NumCritIntervals*(NumCritIntervals-1)/2)
    MYN = ALPHABET_SIZE*EDGES  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)
    observation_space = MYN + EDGES #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
                            #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                            #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
                            #Is there a better way to format the input to make it easier for the neural network to understand things?

    MAX_EXPECTED_EDGES = 2*k                            
    len_game = EDGES 
    state_dim = (observation_space,)
    
    FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
    SECOND_LAYER_NEURONS = 64
    THIRD_LAYER_NEURONS = 4

    LEARNING_RATE = 0.1 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.

    def __init__(self):
        super().__init__()
        self.add(Dense(self.FIRST_LAYER_NEURONS,  activation="relu"))
        self.add(Dense(self.SECOND_LAYER_NEURONS, activation="relu"))
        self.add(Dense(self.THIRD_LAYER_NEURONS, activation="relu"))
        self.add(Dense(self.ALPHABET_SIZE, activation="softmax"))
        self.build((None, self.observation_space))
        self.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate = self.LEARNING_RATE), run_eagerly=True) #Adam optimizer also works well, with lower learning rate