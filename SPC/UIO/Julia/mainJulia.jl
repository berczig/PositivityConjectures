
include("UIODataExtractorJulia.jl")
include("GlobalUIODataPreparerJulia.jl")


using .UIODataExtractorJulia
using .GlobalUIODataPreparerJulia
import .GlobalUIODataPreparerJulia: GlobalUIODataPreparer, computeTrainingData, saveTrainingData, loadTrainingData


# Function to get coreRepresentationsCategorizer, coefficients, and UIO encodings
function main(partition::Tuple{Vararg{Int}}, training_data_load_path::String, core_generator_type::String)
    uio_length = sum(partition)
    
    # 1) get Training Data
    println("[main - step 1 - get data]")
    Preparer = GlobalUIODataPreparer(uio_length)
    if training_data_load_path != ""
        println("loading training data...")
        loadTrainingData(Preparer,training_data_load_path)
    else
        computeTrainingData(Preparer,partition, core_generator_type)
        
        # save Training data?
        if training_data_save_path != ""
            println("saving training data...")
            saveTrainingData(Preparer,training_data_save_path)
        end
    end
    
    coreRepresentationsCategorizer, coefficients = getTrainingData(Preparer)
    encodings = getAllUIOEncodings(Preparer)
    
    println("coeffs: ", coefficients)
    println("UIO encodings: ", encodings)
    
    return coreRepresentationsCategorizer, coefficients, encodings
end

# Main execution block
function __main__()
    println("Running mainJulia...")
    # parameters
    partition = (3, 2, 1)
    training_data_load_path = ""  # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core_9_7_2024.bin"
    training_data_save_path = ""  # "SPC/Saves,Tests/Trainingdata/partition_4_3_2_3rows.bin" # "SPC/Saves,Tests/Trainingdata/partition_5_4__5_core.bin"
    core_generator_type = "EscherCoreGeneratorTrippleJulia"  # "EscherCoreGeneratorTrippleSymmetricNoEqual" # "EscherCoreGeneratorBasic"   "EscherCoreGeneratorTripple"
    
    main(partition, training_data_load_path, core_generator_type)
end

# Execute the main function if the script is run directly
#if abspath(PROGRAM_FILE) == @__FILE__
#    println("Running mainJulia...")
__main__()
#end

#end  # module MainJulia