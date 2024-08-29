module GlobalUIODataPreparerJulia

# Include the module file
include("UIODataExtractorJulia.jl")
include("UIOJulia.jl")
include("CoreGeneratorJulia.jl")
include("miscextraJulia.jl")


using UIODataExtractorJulia
using UIOJulia
using CoreGeneratorJulia
using miscextraJulia

# Import PyCall
using PyCall

# Correct usage of @pyimport with alias
@pyimport SPC.Restructure.cores.CorrectSequenceCoreGenerator as CorrectSequenceCoreGenerator
@pyimport SPC.Restructure.cores.CorrectSequenceCoreGeneratorAbstract as CorrectSequenceCoreGeneratorAbstract
@pyimport SPC.Restructure.cores.EscherCoreGeneratorBasic as EscherCoreGeneratorBasic
@pyimport SPC.Restructure.cores.EscherCoreGeneratorTripple as EscherCoreGeneratorTripple
@pyimport SPC.Restructure.cores.EscherCoreGeneratorAbstract as EscherCoreGeneratorAbstract


mutable struct GlobalUIODataPreparer <: PartiallyLoadable
    n::Int
    coreRepresentationsCategorizer::Dict{Any, Dict{Any, Int}}
    coefficients::Vector{Int}
    partition::Tuple{Vararg{Int}}
    extractors::Vector{UIODataExtractor}

    function GlobalUIODataPreparer(n::Int)
        super(PartiallyLoadable, ["coreRepresentationsCategorizer", "coefficients", "partition"])
        new(n, Dict{Any, Dict{Any, Int}}(), Int[], (), UIODataExtractor[])
    end
end

function initUIOs(preparer::GlobalUIODataPreparer, core_generator_type::Type{CoreGenerator})
    encodings = generate_all_uio_encodings(preparer.n)
    preparer.extractors = [UIODataExtractor(UIO(enc), core_generator_type) for enc in encodings]
    println("Initialized $(length(encodings)) UIOs")
end

function getUIOs(preparer::GlobalUIODataPreparer, i::Int)
    return preparer.extractors[i].uio.encoding
end

function computeTrainingData(preparer::GlobalUIODataPreparer, partition::Tuple{Vararg{Int}}, core_generator_type::Type{CoreGenerator})
    println("computing training data...")
    if !haskey(preparer, :extractors)
        initUIOs(preparer, core_generator_type)
    end
    preparer.partition = partition
    countCoreRepresentations(preparer, partition)
    preparer.coefficients = [extractor.getCoefficient(partition) for extractor in preparer.extractors]
end

function loadTrainingData(preparer::GlobalUIODataPreparer, filepath::String)
    load(preparer, filepath)
end

function getTrainingData(preparer::GlobalUIODataPreparer)
    return preparer.coreRepresentationsCategorizer, preparer.coefficients
end

function saveTrainingData(preparer::GlobalUIODataPreparer, filepath::String)
    save(preparer, filepath)
end

function generate_all_uio_encodings(n::Int)
    function generate_uio_encoding_rec(A::Vector{Vector{Int}}, uio::Vector{Int}, n::Int, i::Int)
        if i == n
            push!(A, uio)
            return
        end
        for j in uio[i-1]:i+1
            generate_uio_encoding_rec(A, vcat(uio, [j]), n, i+1)
        end
    end

    A = Vector{Vector{Int}}()
    generate_uio_encoding_rec(A, [0], n, 1)
    println("Generated ", length(A), " unit order intervals encodings")
    return A
end

function getInputdataAsCountsMatrix(preparer::GlobalUIODataPreparer)
    countmatrix = fill(0, length(preparer.coefficients), length(preparer.coreRepresentationsCategorizer))
    for (corerepID, corerep) in enumerate(keys(preparer.coreRepresentationsCategorizer))
        UIOcounts = preparer.coreRepresentationsCategorizer[corerep]
        for (UIOID, count) in UIOcounts
            countmatrix[UIOID, corerepID] += count
        end
    end
    return countmatrix
end

function countCoreRepresentations(preparer::GlobalUIODataPreparer, partition::Tuple{Vararg{Int}})
    println("Categorizing core representations...")
    core_rep_categories = Dict{Any, Int}()
    counter = Dict{Int, Dict{Int, Int}}()
    corerep_generators = [extractor.getCoreRepresentations(partition) for extractor in preparer.extractors]

    ID = 0
    total_corereps = 0
    n = length(corerep_generators)
    n_10 = n ÷ 10
    for (uioID, corerep_generator) in enumerate(corerep_generators)
        if uioID % n_10 == 0
            println(" > current UIO: $(uioID+1)/$n")
        end
        for corerep in corerep_generator
            total_corereps += 1
            if !haskey(core_rep_categories, corerep)
                ID = length(core_rep_categories)
                core_rep_categories[corerep] = ID
            else
                ID = core_rep_categories[corerep]
            end

            if !haskey(counter, ID)
                counter[ID] = Dict{Int, Int}(uioID => 1)
            else
                categoryOccurrencer = counter[ID]
                if !haskey(categoryOccurrencer, uioID)
                    categoryOccurrencer[uioID] = 1
                else
                    categoryOccurrencer[uioID] += 1
                end
            end
        end
    end

    println("Found $total_corereps core representations in total")
    for (cat, id) in core_rep_categories
        preparer.coreRepresentationsCategorizer[cat] = counter[id]
    end
    println("Found ", length(preparer.coreRepresentationsCategorizer), " distinct core representations / categories")
end

if abspath(PROGRAM_FILE) == @__FILE__
    Preparer = GlobalUIODataPreparer(6)
    println("here")
    # X, y = Preparer.computeTrainingData((4, 2), CoreGenerator)
    X, y = Preparer.loadTrainingData("SPC/Saves,Tests/Preparersavetest.bin")
    # for key in X
    #     println(key, Preparer.coreRepresentationsCategorizer[key])
    # end
    println(y)
    Preparer.saveTrainingData("SPC/Saves,Tests/Preparersavetest.bin")
end

end # module GlobalUIODataPreparerJulia