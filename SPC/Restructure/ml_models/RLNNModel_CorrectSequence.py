from SPC.Restructure.ml_models.MLModel import MLModel
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.utils import register_keras_serializable

@register_keras_serializable()
class RLNNModel_CorrectSequence(MLModel):
    """

    """
    p = 1

    FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
    SECOND_LAYER_NEURONS = 64
    THIRD_LAYER_NEURONS = 4

    LEARNING_RATE = 0.1 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.

    def setParameters(self, partition):
        print("model using", partition, "partition")
        self.partition = partition

        self.l,self.k = partition
        # k+2+2*p
        self.CORE_LENGTH = 2*self.p + 1 + self.k+1   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
        self.ROWS_IN_CONDITIONMATRIX = 2
        self.ALPHABET_SIZE = 4
        print("CORE_LENGTH:", self.CORE_LENGTH)
        self.COLUMNS_IN_CONDITIONMATRIX = int(self.CORE_LENGTH*(self.CORE_LENGTH-1)/2)
        self.EDGES = self.COLUMNS_IN_CONDITIONMATRIX * self.ROWS_IN_CONDITIONMATRIX
        self.MYN = self.ALPHABET_SIZE*self.EDGES  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)
        self.observation_space = self.MYN + self.EDGES #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
                                #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                                #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
                                #Is there a better way to format the input to make it easier for the neural network to understand things?

        self.MAX_EXPECTED_EDGES = 4                      
        self.len_game = self.EDGES 
        self.state_dim = (self.observation_space,)

    def build_model(self):
        self.add(Dense(self.FIRST_LAYER_NEURONS,  activation="relu"))
        self.add(Dense(self.SECOND_LAYER_NEURONS, activation="relu"))
        self.add(Dense(self.THIRD_LAYER_NEURONS, activation="relu"))
        self.add(Dense(self.ALPHABET_SIZE, activation="softmax"))
        self.build((None, self.observation_space))
        self.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate = self.LEARNING_RATE), run_eagerly=True) #Adam optimizer also works well, with lower learning rate

