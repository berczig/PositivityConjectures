import importlib
from SPC.Restructure.GlobalUIODataPreparer import GlobalUIODataPreparer
from SPC.Restructure.ModelLogger import ModelLogger
from SPC.Restructure.ml_algorithms.MLAlgorithm import MLAlgorithm
from SPC.Restructure.ml_algorithms.RLAlgorithm import RLAlgorithm
from SPC.Restructure.ml_models.MLModel import MLModel
from SPC.Restructure.ml_models.RLNNModel import RLNNModel


if __name__ == "__main__":

     # parameters
    partition = (4,2)
    uio_length = sum(partition)
    training_data_load_path = ""
    training_data_save_path = ""
    model_load_path = ""
    model_save_path = ""
    ml_training_algorithm_type = "RLAlgorithm" # exact name of the algorithm python class
    ml_model_type = "RLNNModel" # exact name of the model python class. The model is the component that contains the weights and perform computations, but the algorithm decides how the model is used
    iteration_steps = 100



    # 1) get Training Data
    print("main - step 1 - get data")
    Preparer = GlobalUIODataPreparer(uio_length)
    if training_data_load_path != "":
        Preparer.loadTrainingData(training_data_load_path)
    else:
        Preparer.computeTrainingData(partition)

    # save Training data?
    if training_data_save_path != "":
        Preparer.saveTrainingData(training_data_save_path)


    # 2) get Model
    print("main - step 2 - get model")
    modelLogger = ModelLogger()
    model : MLModel
    if model_load_path != "":
        modelLogger.load_model_logger(model_load_path)
        model = modelLogger.get_model()
    else:
        class_ = getattr(importlib.import_module("SPC.Restructure.ml_models."+ml_model_type), ml_model_type)
        model = class_()
        modelLogger.set_model(model)


    # 3) train
    print("main - step 3 - train")
    algorithm : MLAlgorithm
    class_ = getattr(importlib.import_module("SPC.Restructure.ml_algorithms."+ml_training_algorithm_type), ml_training_algorithm_type)
    algorithm = class_(modelLogger)
    algorithm.setTrainingData(*Preparer.getTrainingData())
    algorithm.train(iteration_steps)


    # 4) save model
    print("main - step 4 - save model")
    if model_save_path != "":
        modelLogger.save_model_logger(model_save_path)


     

