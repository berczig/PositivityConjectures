from SPC.Restructure.GlobalUIODataPreparer import GlobalUIODataPreparer
from SPC.Restructure.ModelLogger import ModelLogger
from SPC.Restructure.MLAlgorithm import MLAlgorithm


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
    Preparer = GlobalUIODataPreparer(uio_length)
    if training_data_load_path != "":
        Preparer.loadTrainingData(training_data_load_path)
    else:
        Preparer.computeTrainingData(partition)

    # save Training data?
    if training_data_save_path != "":
        Preparer.saveTrainingData(training_data_save_path)


    # 2) get Model
    modelLogger = ModelLogger()
    if model_load_path != "":
        modelLogger.load_model_logger(model_load_path)
        model = modelLogger.get_model()
    else:
        model = exec(ml_model_type+"()")
        modelLogger.set_model(model)


    # 3) train
    algorithm : MLAlgorithm
    algorithm = exec(ml_training_algorithm_type+"()")
    algorithm.setTrainingData(Preparer.getTrainingData())
    algorithm.train(model, iteration_steps)


    # 4) save model
    if model_save_path != "":
        modelLogger.save_model_logger(model_save_path)


     

