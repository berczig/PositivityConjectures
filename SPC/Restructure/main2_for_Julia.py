import importlib
from SPC.Restructure.GlobalUIODataPreparer import GlobalUIODataPreparer

# returns coreRepresentationsCategorizer, coefficients and UIO encodings
def main(partition, training_data_load_path,core_generator_type):

    uio_length = sum(partition)
    # core_data_type

    # 1) get Training Data
    print("[main - step 1 - get data]")
    Preparer = GlobalUIODataPreparer(uio_length)
    if training_data_load_path != "":
        print("loading training data...")
        Preparer.loadTrainingData(training_data_load_path)
    else:
        Preparer.computeTrainingData(partition, core_generator_type)

        # save Training data?
        if training_data_save_path != "":
            print("saving training data...")
            Preparer.saveTrainingData(training_data_save_path)

    coreRepresentationsCategorizer, coefficients = Preparer.getTrainingData()
    encodings = Preparer.getAllUIOEncodings()

    #print("coreRepresentationsCategorizer:", coreRepresentationsCategorizer)
    print("coeffs:", coefficients)
    print("UIO encodings:", encodings)

    return coreRepresentationsCategorizer, coefficients, encodings




     
if __name__ == "__main__":

     # parameters
    partition = (3,2,1,1) 
    training_data_load_path = "" # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core_9_7_2024.bin"
    training_data_save_path = "" #"SPC/Saves,Tests/Trainingdata/partition_4_3_2_3rows.bin" # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core.bin"
    core_generator_type =  "EscherCoreGeneratorQuadruple" # "EscherCoreGeneratorTrippleSymmetricNoEqual" # "EscherCoreGeneratorBasic"   "EscherCoreGeneratorTripple"

    main(partition, training_data_load_path, core_generator_type)
