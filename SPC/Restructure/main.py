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
    training_data_load_path = "" # "SPC/Saves,Tests/Trainingdata/partition_4_2.bin"
    training_data_save_path = "" # "SPC/Saves,Tests/Trainingdata/partition_4_2.bin"
    model_load_path = "" #"SPC/Saves,Tests/models/my_newmodel.keras"
    model_save_path = "" # "SPC/Saves,Tests/models/my_newmodel.keras"
    model_save_time = 300 # how many seconds have to have elapsed before saving
    ml_training_algorithm_type = "RLAlgorithm" # exact name of the algorithm python class
    ml_model_type = "RLNNModel_Escher" # exact name of the model python class. The model is the component that contains the weights and perform computations, but the algorithm decides how the model is used
    core_data_type = "escher" # escher or correctsequence
    iteration_steps = 5000
    plot_after_training = False



    # 1) get Training Data
    print("[main - step 1 - get data]")
    Preparer = GlobalUIODataPreparer(uio_length)
    if training_data_load_path != "":
        print("loading training data...")
        Preparer.loadTrainingData(training_data_load_path)
    else:
        print("computing training data...")
        Preparer.computeTrainingData(partition, core_data_type)

        # save Training data?
        if training_data_save_path != "":
            print("saving training data...")
            Preparer.saveTrainingData(training_data_save_path)


    # 2) get Model
    print("[main - step 2 - get model]")
    modelLogger = ModelLogger()
    model : MLModel
    if model_load_path != "":
        print("loading model...")
        modelLogger.load_model_logger(model_load_path)
        model = modelLogger.get_model()
    else:
        print("creating new model...")
        class_ = getattr(importlib.import_module("SPC.Restructure.ml_models."+ml_model_type), ml_model_type)
        model = class_()
        model.setParameters(partition)
        model.build_model()
        modelLogger.set_model(model)
    model.summary()
    assert model.partition == Preparer.partition, "model parition does not match training data partition"


    # 3) train
    print("[main - step 3 - train]")
    algorithm : MLAlgorithm
    class_ = getattr(importlib.import_module("SPC.Restructure.ml_algorithms."+ml_training_algorithm_type), ml_training_algorithm_type)
    algorithm = class_(modelLogger)
    algorithm.setTrainingData(*Preparer.getTrainingData())
    algorithm.train(iteration_steps, model_save_path, model_save_time)


    # 4) save model
    print("[main - step 4 - save model]")
    if model_save_path != "":
        print("saving model...")
        modelLogger.save_model_logger(model_save_path)
    else:
        print("no model save path provided")

    # 5) plot
    if plot_after_training:
        modelLogger.make_plots()


     

