using Combinatorics
using Printf
using Random
using LinearAlgebra

#### Parameters Start ########
uio_partition = (3, 2)
uio_max_size = 5
#### Parameters End   ########

#### Calculate constants ########
#uio_size = sum(uio_partition)
#### Calculate constants ########


struct UIO
    N::Int
    encoding::Vector{Int}
    comparison_matrix::Matrix{Int}
    repr::String
end

const INCOMPARABLE = 100
const LESS = 101
const GREATER = 102
const EQUAL = 103
const RELATIONTEXT = Dict(LESS => "<", EQUAL => "=", GREATER => ">")
const RELATIONTEXT2 = Dict(LESS => "LE", EQUAL => "EQ", GREATER => "GR")

function UIO(uio_encoding::Vector{Int})
    N = length(uio_encoding)
    comparison_matrix = fill(EQUAL, N, N)
    for i in 1:N
        for j in i+1:N
            if uio_encoding[j] <= i
                comparison_matrix[i, j] = INCOMPARABLE
                comparison_matrix[j, i] = INCOMPARABLE
            else
                comparison_matrix[i, j] = LESS
                comparison_matrix[j, i] = GREATER
            end
        end
    end
    repr = string(uio_encoding)
    UIO(N, uio_encoding, comparison_matrix, repr)
end

Base.show(io::IO, uio::UIO) = println(io, uio.repr)



struct EscherDoublePartitionGenerator
end

struct UIODataExtractor
    uio::UIO
    core_generator_class::String

    function UIODataExtractor(uio::UIO, core_generator_class::String)
        new(uio, core_generator_class)
    end

    
end

mutable struct GlobalUIODataPreparer
    n::Int
    uios_initialized::Bool
    coreRepresentationsCategorizer::Dict{Any, Dict{Any, Int}}
    coefficients::Vector{Any}
    extractors::Vector{UIODataExtractor}
    partition::Tuple{Vararg{Int}}

    function GlobalUIODataPreparer(n::Int)
        new(n, false, Dict{Any, Dict{Any, Int}}(), [], [], ())
    end
end

Base.show(io::IO, extractor::UIODataExtractor) = println(io, "EXTRACTOR OF [", extractor.uio.encoding, "]")

function getKPermutationsOfN(n, k)
    return collect(permutations(1:n, k))
end

function partitionsOfN(n)
    result = []
    stack = [(n, 1, [])]
    
    while !isempty(stack)
        (remaining, start, current) = pop!(stack)
        
        if remaining == 0
            push!(result, current)
        else
            for i in start:remaining
                push!(stack, (remaining - i, i, vcat(current, [i])))
            end
        end
    end
    
    return result
end

function count(iterable)
    return length(iterable)
end

function getUnusedFilepath(filepath)
    folder, filename = splitpath(filepath)
    base_filename, extension = splitext(filename)
    newfilename = filename
    i = 1
    while newfilename in readdir(folder)
        newfilename = "$(base_filename)_$(lpad(i, 3, '0'))$(extension)"
        i += 1
    end
    return joinpath(folder, newfilename)
end

function isescher(self::UIO, seq::Vector{Int})
    for i in 1:length(seq)-1
        if !isarrow(self, seq, i, i+1)
            return false
        end
    end
    return isarrow(self, seq, length(seq), 1)
end

function iscorrect(self::UIO, seq::Vector{Int})
    for i in 2:length(seq)
        if !isarrow(self, seq, i-1, i)
            return false
        end
        intersects = false
        for j in 1:i-1
            if self.comparison_matrix[seq[i], seq[j]] in [UIO.LESS, UIO.INCOMPARABLE]
                intersects = true
            end
        end
        if !intersects
            return false
        end
    end
    return true
end

function isarrow(self::UIO, escher::Vector{Int}, i::Int, j::Int, verbose::Bool=false)
    if verbose
        println("arrow", escher, i, j, self.comparison_matrix[escher[i], escher[j]] != GREATER)
    end
    return self.comparison_matrix[escher[i], escher[j]] != GREATER
end

function intervalsAreIntersecting(self::UIO, i::Int, j::Int)
    return self.comparison_matrix[i, j] == UIO.INCOMPARABLE
end

function intervalIsToTheRight(self::UIO, i::Int, j::Int)
    return self.comparison_matrix[i, j] == UIO.GREATER
end

function toPosetData(self::UIO, seq::Vector{Int})
    k = length(seq)
    return tuple([self.comparison_matrix[seq[i], seq[j]] for i in 1:k for j in i+1:k]...)
end

function addupto(List::Vector{Int}, element::Int, uptoindex::Int)
    if length(List) == uptoindex
        return List
    end
    A = fill(element, uptoindex - length(List))
    return vcat(List, A)
end


function getsubuioencoding(self::UIO, seq::Vector{Int})
    N = length(seq)
    encod = []
    for i in 1:N
        for j in i+1:N
            if self.comparison_matrix[seq[i], seq[j]] != self.INCOMPARABLE
                encod = addupto(encod, i, j)
                break
            elseif j == N
                encod = addupto(encod, i, N)
            end
        end
    end
    if self.comparison_matrix[seq[end], seq[end-1]] != self.INCOMPARABLE
        push!(encod, N-1)
    end
    return encod
end

function getCores(self::UIODataExtractor, partition::Tuple{Vararg{Int}})
    cores = []
    if self.core_generator_class == "EscherDoublePartitionGenerator"
        GEN = EscherDoublePartitionGenerator(self.uio, partition)
        for escher in getEschers(self, partition)
            push!(cores, GEN.generateCore(escher))
        end
    end
    return cores
end

function getCoreRepresentations(self::UIODataExtractor, partition::Tuple{Vararg{Int}})
    core_representations = []
    if self.core_generator_class == "EscherDoublePartitionGenerator"
        GEN = EscherDoublePartitionGenerator(self.uio, partition)
        for core in getCores(self, partition)
            push!(core_representations, GEN.getCoreRepresentation(core))
        end
    end
    return core_representations
end


function getCoefficient(self::UIODataExtractor, partition::Tuple{Vararg{Int}})
    if length(partition) == 1
        return count(getEschers(self, partition))
    elseif length(partition) == 2
        return countEschers(self, partition) - countEschers(self, (self.uio.N,))
    elseif length(partition) == 3
        n, k, l = partition
        return 2 * countEschers(self, (n+k+l,)) + countEschers(self, partition) - countEschers(self, (n+l, k)) - countEschers(self, (n+k, l)) - countEschers(self, (l+k, n))
    elseif length(partition) == 4
        a, b, c, d = partition
        return countEschers(self, (a, b, c, d)) - countEschers(self, (a+b, c, d)) - countEschers(self, (a+c, b, d)) - countEschers(self, (a+d, b, c)) - countEschers(self, (b+c, a, d)) - countEschers(self, (b+d, a, c)) - countEschers(self, (c+d, a, b)) + countEschers(self, (a+b, c+d)) + countEschers(self, (a+c, b+d)) + countEschers(self, (a+d, b+c)) + 2 * countEschers(self, (a+b+c, d)) + 2 * countEschers(self, (a+b+d, c)) + 2 * countEschers(self, (a+c+d, b)) + 2 * countEschers(self, (b+c+d, a)) - 6 * countEschers(self, (a+b+c+d,))
    end
end

function countEschers(self::UIODataExtractor, partition::Tuple{Vararg{Int}})
    return count(getEschers(self, partition))
end

function getEschers(self::UIODataExtractor, partition::Tuple{Vararg{Int}})
    P = getKPermutationsOfN(self.uio.N, sum(partition))
    eschers = []
    if length(partition) == 1
        for seq in P
            if isescher(self.uio, seq)
                push!(eschers, seq)
            end
        end
    elseif length(partition) == 2
        a = partition[1]
        for seq in P
            if isescher(self.uio, seq[1:a]) && isescher(self.uio, seq[a+1:end])
                push!(eschers, (seq[1:a], seq[a+1:end]))
            end
        end
    elseif length(partition) == 3
        a, b, c = partition
        for seq in P
            if isescher(self.uio, seq[1:a]) && isescher(self.uio, seq[a+1:a+b]) && isescher(self.uio, seq[a+b+1:end])
                push!(eschers, (seq[1:a], seq[a+1:a+b], seq[a+b+1:end]))
            end
        end
    elseif length(partition) == 4
        a, b, c, d = partition
        for seq in P
            if isescher(self.uio, seq[1:a]) && isescher(self.uio, seq[a+1:a+b]) && isescher(self.uio, seq[a+b+1:a+b+c]) && isescher(self.uio, seq[a+b+c+1:end])
                push!(eschers, (seq[1:a], seq[a+1:a+b], seq[a+b+1:a+b+c], seq[a+b+c+1:end]))
            end
        end
    end
    return eschers
end


function initUIOs!(self::GlobalUIODataPreparer, core_generator_type::String)
    self.uios_initialized = true
    encodings = generate_all_uio_encodings(self.n)
    self.extractors = [UIODataExtractor(UIO(enc), core_generator_type) for enc in encodings]
    println("Initialized $(length(encodings)) UIOs")
end

function getUIOs(self::GlobalUIODataPreparer, i::Int)
    return self.extractors[i].uio.encoding
end

function getAllUIOEncodings(self::GlobalUIODataPreparer)
    return generate_all_uio_encodings(self.n)
end

function computeTrainingData(self::GlobalUIODataPreparer, partition::Tuple{Vararg{Int}}, core_generator_type::String)
    println("computing training data...")

    if !self.uios_initialized
        initUIOs!(self, core_generator_type)
    end

    self.partition = partition
    n = length(self.extractors)
    n_10 = max(1, div(n, 10))
    println("Calculating coefficients... (print every $n_10 iterations)")
    self.coefficients = []
    for (uioID, extractor) in enumerate(self.extractors)
        if uioID % n_10 == 0
            println(" > current UIO: $(uioID + 1)/$n")
        end
        push!(self.coefficients, getCoefficient(extractor,partition))
    end
end

function loadTrainingData(self::GlobalUIODataPreparer, filepath::String)
    # Placeholder for the actual implementation
    println("Loading training data from $filepath")
end

function getTrainingData(self::GlobalUIODataPreparer)
    return self.coreRepresentationsCategorizer, self.coefficients
end

function saveTrainingData(self::GlobalUIODataPreparer, filepath::String)
    # Placeholder for the actual implementation
    println("Saving training data to $filepath")
end

function generate_uio_encoding_rec!(A::Vector{Vector{Int}}, uio::Vector{Int}, n::Int, i::Int)
    if i == n
        push!(A, copy(uio))
        return
    end
    # Adjust indexing to start correctly from 1
    # When i == 1, use 0 as the start for the first element
    start_value = (i == 1) ? 0 : uio[i]
    for j in start_value:i
        generate_uio_encoding_rec!(A, vcat(uio, [j]), n, i + 1)
    end
end

function generate_all_uio_encodings(n::Int)
    A = Vector{Vector{Int}}()
    generate_uio_encoding_rec!(A, [0], n, 1)
    println("Generated ", length(A), " unit order intervals encodings")
    return A
end


# function generate_uio_encoding_rec!(A::Vector{Vector{Int}}, uio::Vector{Int}, n::Int, i::Int)
#     if i == n
#         push!(A, copy(uio))
#         return
#     end
#     for j in uio[i]:(i+1)
#         generate_uio_encoding_rec!(A, vcat(uio, [j]), n, i + 1)
#     end
# end

# function generate_all_uio_encodings(n::Int)
#     A = Vector{Vector{Int}}()
#     generate_uio_encoding_rec!(A, [0], n, 1)
#     println("Generated ", length(A), " unit order intervals encodings")
#     return A
# end






function getInputdataAsCountsMatrix(self::GlobalUIODataPreparer)
    countmatrix = fill(0, length(self.coefficients), length(self.coreRepresentationsCategorizer))
    for (corerepID, corerep) in enumerate(keys(self.coreRepresentationsCategorizer))
        UIOcounts = self.coreRepresentationsCategorizer[corerep]
        for (UIOID, count) in UIOcounts
            countmatrix[UIOID + 1, corerepID + 1] += count
        end
    end
    return countmatrix
end

function countCoreRepresentations(self::GlobalUIODataPreparer, partition::Tuple{Vararg{Int}})
    core_rep_categories = Dict{Any, Int}()
    counter = Dict{Int, Dict{Int, Int}}()
    corerep_generators = [extractor.getCoreRepresentations(partition) for extractor in self.extractors]

    total_corereps = 0
    n = length(corerep_generators)
    n_10 = max(1, div(n, 10))
    println("Categorizing core representations... (print every $n_10 iterations)")
    for (uioID, corerep_generator) in enumerate(corerep_generators)
        if uioID % n_10 == 0
            println(" > current UIO: $(uioID + 1)/$n")
        end
        for corerep in corerep_generator
            total_corereps += 1
            if !haskey(core_rep_categories, corerep)
                core_rep_categories[corerep] = length(core_rep_categories)
            end
            ID = core_rep_categories[corerep]
            if !haskey(counter, ID)
                counter[ID] = Dict(uioID => 1)
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

    for (cat, ID) in core_rep_categories
        self.coreRepresentationsCategorizer[cat] = counter[ID]
    end
    println("Found $(length(self.coreRepresentationsCategorizer)) distinct core representations / categories")
end

function getUIOData(uio_max_size::Int, partition::Tuple{Vararg{Int}})
    DataPreparer = GlobalUIODataPreparer(uio_max_size)
    computeTrainingData(DataPreparer, partition, "EMPTY")
    return generate_all_uio_encodings(uio_max_size), DataPreparer.coefficients
end

function getAllUIODataUpTo(uio_max_size::Int)
    for uio_size in 1:uio_max_size
        for partitionsize in 1:uio_size
            for partition in partitionsOfN(partitionsize)
                printUIOData(getUIOData(uio_size, Tuple(partition)), Tuple(partition))
            end
        end
    end
end

function printUIOData(data::Tuple{Vector{Vector{Int}}, Vector{Any}}, partition::Tuple{Vararg{Int}})
    encod, coeffs = data
    n = length(encod)
    for i in 1:n
        println("$(i)th UIO $(encod[i]) has $(partition)-coeff $(coeffs[i])")
    end
end

# Saving the encod[i],coeffs[i] pairs into a file

using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")

using CSV
using DataFrames

using CSV
using DataFrames

function saveUIODataToCSV(data::Tuple{Vector{Vector{Int}}, Vector{Any}}, filename::String)
    encod, coeffs = data
    n = length(encod)
    
    # Create a DataFrame to store the data
    df = DataFrame(encod = Vector{String}(undef, n), coeffs = Vector{Any}(undef, n))
    
    for i in 1:n
        df.encod[i] = string(encod[i])
        df.coeffs[i] = coeffs[i]
    end
    
    # Save the DataFrame to a CSV file
    CSV.write(filename, df)
end

# Example usage
@time begin
data = getUIOData(uio_max_size, uio_partition)
# the filename should be the format uio_data_uio_partition_n=uio_max_size.csv
filename = "SPC/Transformers/uio_data_$(join(uio_partition, "_"))_n=$(uio_max_size).csv" 
saveUIODataToCSV(data, filename)
printUIOData(data, uio_partition)
end

