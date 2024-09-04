module Main2ForJulia

using GlobalUIODataPreparerJulia

# Import the necessary Python module
#@pyimport SPC.Restructure.GlobalUIODataPreparer as GlobalUIODataPreparer

# Function to get coreRepresentationsCategorizer, coefficients, and UIO encodings
function main(partition::Tuple{Vararg{Int}}, training_data_load_path::String, core_generator_type::String)
    uio_length = sum(partition)
    
    # 1) get Training Data
    println("[main - step 1 - get data]")
    Preparer = GlobalUIODataPreparer.GlobalUIODataPreparer(uio_length)
    
    if training_data_load_path != ""
        println("loading training data...")
        Preparer.loadTrainingData(training_data_load_path)
    else
        Preparer.computeTrainingData(partition, core_generator_type)
        
        # save Training data?
        if training_data_save_path != ""
            println("saving training data...")
            Preparer.saveTrainingData(training_data_save_path)
        end
    end
    
    coreRepresentationsCategorizer, coefficients = Preparer.getTrainingData()
    encodings = Preparer.getAllUIOEncodings()
    
    println("coeffs: ", coefficients)
    println("UIO encodings: ", encodings)
    
    return coreRepresentationsCategorizer, coefficients, encodings
end

# Main execution block
function __main__()
    # parameters
    partition = (3, 2, 1, 1)
    training_data_load_path = ""  # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core_9_7_2024.bin"
    training_data_save_path = ""  # "SPC/Saves,Tests/Trainingdata/partition_4_3_2_3rows.bin" # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core.bin"
    core_generator_type = "EscherCoreGeneratorQuadruple"  # "EscherCoreGeneratorTrippleSymmetricNoEqual" # "EscherCoreGeneratorBasic"   "EscherCoreGeneratorTripple"
    
    main(partition, training_data_load_path, core_generator_type)
end

# Execute the main function if the script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    __main__()
end

end  # module Main2ForJulia