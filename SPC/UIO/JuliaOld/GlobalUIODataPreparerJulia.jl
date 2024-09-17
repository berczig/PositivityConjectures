module GlobalUIODataPreparerJulia

include("UIOJulia.jl")
include("CoreGeneratorJulia.jl")
include("UIODataExtractorJulia.jl")
include("miscextraJulia.jl")
include("EscherCoreGeneratorTrippleJulia.jl")

using .UIOJulia
using .CoreGeneratorJulia
using .UIODataExtractorJulia: UIODataExtractor
using .miscextraJulia: PartiallyLoadable
using .EscherCoreGeneratorTrippleJulia

mutable struct GlobalUIODataPreparer
    n::Int
    coreRepresentationsCategorizer::Dict{Any, Dict{Any, Int}}
    coefficients::Vector{Any}
    partition::Tuple{Vararg{Int64}}
    extractors::Vector{UIODataExtractor}
    partially_loadable::PartiallyLoadable
    
    

    function GlobalUIODataPreparer(n::Int)
        partially_loadable = PartiallyLoadable([:coreRepresentationsCategorizer, :coefficients, :partition])
        new(n, Dict{Any, Dict{Any, Int}}(), [], (), [], partially_loadable)
    end
end

function initUIOs!(self::GlobalUIODataPreparer, core_generator_type::String)
    println("Initializing UIOs...")
    module_name = Symbol("." * core_generator_type)
    class_name = Symbol(core_generator_type)
    # Dynamically import the module and retrieve the class
    @eval class_ = $(Symbol(module_name)).$(class_name)
    encodings = generate_all_uio_encodings(self.n)
    println("Generated $(length(encodings)) UIO encodings")
    self.extractors = [UIODataExtractor(UIO(enc), class_) for enc in encodings]
    println("Initialized $(length(encodings)) UIOs")
end

function computeTrainingData(self::GlobalUIODataPreparer, partition::Tuple{Vararg{Int64}}, core_generator_type::String)
    println("Computing training data...")
    self.partition = partition
    initUIOs!(self, core_generator_type)
    # Add logic to compute training data and populate coreRepresentationsCategorizer and coefficients
end

function saveTrainingData(self::GlobalUIODataPreparer, filename::String)
    println("Saving training data to $filename...")
    # Add logic to save training data
end

function loadTrainingData(self::GlobalUIODataPreparer, filename::String)
    println("Loading training data from $filename...")
    # Add logic to load training data
end

function generate_all_uio_encodings(n::Int)
    function generate_uio_encoding_rec!(A::Vector{Vector{Int}}, uio::Vector{Int}, n::Int, i::Int)
        if i == n
            push!(A, uio)
            return
        end
        for j in uio[i-1]:i
            generate_uio_encoding_rec!(A, vcat(uio, [j]), n, i+1)
        end
    end

    A = Vector{Vector{Int}}()
    generate_uio_encoding_rec!(A, [0], n, 1)
    println("Generated $(length(A)) unit order intervals encodings")
    return A
end

end  # module GlobalUIODataPreparerJulia