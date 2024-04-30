import matplotlib.pyplot as plt
from SPC.Restructure.FilterEvaluator import FilterEvaluator
from SPC.misc.extra import PartiallyLoadable
from SPC.Restructure.ml_models.MLModel import MLModel
from SPC.Restructure.ml_models.RLNNModel import RLNNModel
from keras.models import load_model
from keras.utils import custom_object_scope

class ModelLogger(PartiallyLoadable):
    def __init__(self):
        super().__init__(["step", "all_scores", "bestscore_history", "meanscore_history", "numgraph_history", "calculationtime_history", "partition"])
        self.step = 0
        self.partition = None
        self.all_scores = {} # {filter_as_tuple:score}
        self.bestscore_history = [] # history of best score
        self.meanscore_history = [] # history of mean score
        self.numgraph_history = [] # history of number of graphs which we allready have calculated the score of
        self.calculationtime_history = []
        self.model : MLModel

    def load_model_logger(self, model_path:str):
        assert model_path[len(model_path)-6:] == ".keras", "wrong file format"
        self.load(model_path[:len(model_path)-6]+".my_meta")
        self.model = load_model(model_path)
        self.model.setParameters(self.partition)

    def set_model(self, model):
        self.model = model
        self.partition = model.partition

    def get_model(self):
        return self.model
    
    def save_model_logger(self, model_path):
        self.model.save(model_path)
        self.save(model_path[:len(model_path)-6]+".my_meta")

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
        print("bestscore:", bestscore, "from", beststate)

        # implement this again(?):
        #from SPC.Restructure.ml_algorithms.RLAlgorithm import RLAlgorithm
        #condmat = RLAlgorithm.convertStateToConditionMatrix(beststate)
        #conditiontext = evaluator.convertConditionMatrixToText(condmat)
        #print(conditiontext, "\nhas a score of ", evaluator.evaluate(condmat))