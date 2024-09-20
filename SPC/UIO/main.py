import importlib
from SPC.UIO.GlobalUIODataPreparer import GlobalUIODataPreparer
from SPC.UIO.ModelLogger import ModelLogger
from SPC.UIO.ml_algorithms.LearningAlgorithm import LearningAlgorithm
from SPC.UIO.ml_algorithms.RLAlgorithm import RLAlgorithm
from SPC.UIO.ml_models.MLModel import MLModel
from SPC.UIO.ml_models.RLNNModel_CorrectSequence import RLNNModel_CorrectSequence
from SPC.misc.misc import refpath
from keras.models import Sequential
import tensorflow as tf
import os


def main(partition, training_data_load_path, training_data_save_path, model_load_path, model_save_path, model_save_time, ml_training_algorithm_type, ml_model_type, 
         core_generator_type, iteration_steps, plot_after_training, RL_n_graphs, condition_rows, edgePenalty, uio_size):

    # Let all relative paths start in the folder containing SPC
    training_data_load_path = "" if training_data_load_path == "" else os.path.join(refpath, training_data_load_path)
    training_data_save_path = "" if training_data_save_path == "" else os.path.join(refpath, training_data_save_path)
    model_load_path = "" if model_load_path == "" else os.path.join(refpath, model_load_path)
    model_save_path = "" if model_save_path == "" else os.path.join(refpath, model_save_path)


    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("gpu:", tf.test.is_gpu_available())
    device_name = tf.test.gpu_device_name()
    print(device_name)

    # core_data_type

    # 1) get Training Data
    print("[main - step 1 - get data]")
    Preparer = GlobalUIODataPreparer(uio_size)
    if training_data_load_path != "":
        print("loading training data...")
        Preparer.loadTrainingData(training_data_load_path)
    else:
        Preparer.computeTrainingData(partition, core_generator_type)

        # save Training data?
        if training_data_save_path != "":
            print("saving training data...")
            Preparer.saveTrainingData(training_data_save_path)


    # 2) get Model
    print("[main - step 2 - get model]")
    modelLogger = ModelLogger()
    modelLogger.setParameters(partition, core_generator_type, RL_n_graphs, ml_training_algorithm_type, 
    ml_model_type, condition_rows, edgePenalty, uio_size)
    
    if model_load_path != "":
        print("loading model...")
        modelLogger.load_model_logger(model_load_path)
    else:
        print("creating new model...")
        class_ = getattr(importlib.import_module("SPC.UIO.ml_models."+ml_model_type), ml_model_type)
        model_structure: MLModel
        model_structure = class_()
        model_structure.setParameters(partition, condition_rows, modelLogger.core_length, modelLogger.core_representation_length)
        modelLogger.set_model_structure(model_structure)
        modelLogger.set_keras_model(model_structure.build_model())

    modelLogger.get_keras_model().summary()
    assert modelLogger.partition == Preparer.partition, "model parition({}) does not match training data partition({})".format(modelLogger.partition, Preparer.partition)


    # 3) train
    print("[main - step 3 - train]")
    algorithm : LearningAlgorithm
    class_ = getattr(importlib.import_module("SPC.UIO.ml_algorithms."+ml_training_algorithm_type), ml_training_algorithm_type)
    algorithm = class_(modelLogger)
    algorithm.setUIO_preparer(Preparer)
    algorithm.setTrainingData(*Preparer.getTrainingData())
    algorithm.train(iteration_steps, model_save_path, model_save_time)


    # 4) save model
    print("[main - step 4 - save model]")
    if model_save_path != "":
        if algorithm.saveable_model:
            print("saving model...")
            modelLogger.save_model_logger(model_save_path)
        else:
            print("Algorithm not saveable")
    else:
        print("no model save path provided")

    # 5) plot
    if plot_after_training:
        modelLogger.make_plots()



     
if __name__ == "__main__":

     # parameters
    uio_size = 4
    partition = (2,1,1) 
    training_data_load_path = "" #"SPC/Saves,Tests/Trainingdata/partition_2_1_1_n5.bin" # "SPC/Saves,Tests/Trainingdata/5_2_2.bin" # "SPC/Saves,Tests/5_2_2.bin" # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core_9_7_2024.bin"
    training_data_save_path = "" # "SPC/Saves,Tests/Trainingdata/partition_2_1_1_n8.bin" # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core.bin"
    model_load_path =  "" # "SPC/Saves,Tests/models/Quadruple6.keras" # "SPC/Saves,Tests/models/for_result_viewer_test.keras" #"SPC/Saves,Tests/models/for_result_viewer_test.keras" # "SPC/Saves,Tests/models/tripple_escher.keras"#"" #"SPC/Saves,Tests/models/my_newmodel.keras"
    model_save_path = "SPC/Saves,Tests/models/new_2_1_1.keras"
    model_save_time = 1800 # how many seconds have to have elapsed before saving
    ml_training_algorithm_type = "CheckSpecificAlgorithm" # CheckSpecificAlgorithm/BruteForceAlgorithm/RLAlgorithm - exact name of the algorithm python class
    ml_model_type =  "RLNNModel_Escher_TrippleNoEqual" # RLNNModel_Escher_TrippleNoEqual RLNNModel_CorrectSequence or RLNNModel_Escher or RLNNModel_Escher_Tripple - exact name of the model python class. The model is the component that contains the weights and perform computations, but the algorithm decides how the model is used
    core_generator_type =  "EscherCoreGeneratorTrippleSymmetricNoEqual" # "EscherCoreGeneratorQuadruple "EscherCoreGeneratorBasic"   "EscherCoreGeneratorTripple" "EscherCoreGeneratorTrippleSymmetricNoEqual"
    iteration_steps = 10
    RL_n_graphs = 200
    condition_rows = 4
    edgePenalty = 1
    plot_after_training = False

    main(partition, training_data_load_path, training_data_save_path, model_load_path, model_save_path, model_save_time, ml_training_algorithm_type, 
         ml_model_type, core_generator_type, iteration_steps, plot_after_training, RL_n_graphs, condition_rows, edgePenalty, uio_size)