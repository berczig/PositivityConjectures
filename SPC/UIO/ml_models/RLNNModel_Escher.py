from SPC.UIO.ml_models.MLModel import MLModel
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

class RLNNModel_Escher(MLModel):
    """

    """


    FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
    SECOND_LAYER_NEURONS = 64
    THIRD_LAYER_NEURONS = 4
 
    LEARNING_RATE = 0.05 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.

    def setParameters(self, partition, condition_rows, core_length, corerep_length):
        print("model using", partition, "partition")
        self.partition = partition

        # k+2+2*p
        self.CORE_LENGTH = core_length   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
        self.ROWS_IN_CONDITIONMATRIX = condition_rows
        self.ALPHABET_SIZE = 4
        self.COLUMNS_IN_CONDITIONMATRIX = corerep_length
        self.EDGES = self.COLUMNS_IN_CONDITIONMATRIX * self.ROWS_IN_CONDITIONMATRIX
        self.MYN = self.ALPHABET_SIZE*self.EDGES  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)
        self.observation_space = self.MYN + self.EDGES #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
                                #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                                #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
                                #Is there a better way to format the input to make it easier for the neural network to understand things?

        self.MAX_EXPECTED_EDGES = 3
        self.len_game = self.EDGES 
        self.state_dim = (self.observation_space,)

        print("self.ALPHABET_SIZE:", self.ALPHABET_SIZE)
        print("self.COLUMNS_IN_CONDITIONMATRIX:", self.COLUMNS_IN_CONDITIONMATRIX)
        print("self.CORE_LENGTH:", self.CORE_LENGTH)
        print("self.EDGES:", self.EDGES)
        print("self.MYN:", self.MYN)
        print("self.observation_space:", self.observation_space)

    def build_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(self.FIRST_LAYER_NEURONS,  activation="relu"))
        model.add(Dense(self.SECOND_LAYER_NEURONS, activation="relu"))
        model.add(Dense(self.THIRD_LAYER_NEURONS, activation="relu"))
        model.add(Dense(self.ALPHABET_SIZE, activation="softmax"))
        
        model.build((None, self.observation_space))
        model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate = self.LEARNING_RATE), run_eagerly=True) #Adam optimizer also works well, with lower learning rate

        return model