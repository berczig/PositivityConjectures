module UIOJulia

using LinearAlgebra

abstract type AbstractUIO end

const INCOMPARABLE = 100    # (i,j) is INCOMPARABLE if neither i < j nor j < i nor i = j -- connected in the incomparability graph -- intersecting intervals
const LESS = 101            # (i,j) is LE iff i < j     interval i is to the left of j 
const GREATER = 102         # (i,j) is LE iff i > j     interval i is to the right of j
const EQUAL = 103           # (i,j) is EQ iff i = j     interval i and interval j are same interval
const RELATIONTEXT = Dict(LESS => "<", EQUAL => "=", GREATER => ">")
const RELATIONTEXT2 = Dict(LESS => "LE", EQUAL => "EQ", GREATER => "GR")

struct ConcreteUIO <: AbstractUIO
    N::Int
    encoding::Vector{Int}
    repr::String
    comparison_matrix::Matrix{Int}

    function ConcreteUIO(uio_encoding::Vector{Int})
        N = length(uio_encoding)
        encoding = uio_encoding
        repr = string(encoding)
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

        new(N, encoding, repr, comparison_matrix)
    end
end





function Base.show(io::IO, uio::ConcreteUIO)
    println(io, uio.repr)
end

### POSET STRUCTURE ###

function isescher(uio::ConcreteUIO, seq::Vector{Int})
    for i in 1:length(seq)-1
        if !isarrow(uio, seq, i, i+1)
            return false
        end
    end
    return isarrow(uio, seq, length(seq), 1)
end

function iscorrect(uio::ConcreteUIO, seq::Vector{Int})
    for i in 2:length(seq)
        # 1) arrow condition
        if !isarrow(uio, seq, i-1, i)
            return false
        end
        # 2) intersects with some previous interval
        intersects = false
        for j in 1:i-1
            if uio.comparison_matrix[seq[i], seq[j]] in [LESS, INCOMPARABLE]
                intersects = true
                break
            end
        end
        if !intersects
            return false
        end
    end
    return true
end

function isarrow(uio::ConcreteUIO, escher::Vector{Int}, i::Int, j::Int, verbose::Bool=false)
    if verbose
        println("arrow", escher, i, j, uio.comparison_matrix[escher[i], escher[j]] != GREATER)
    end
    return uio.comparison_matrix[escher[i], escher[j]] != GREATER  # EQUAL also intersects
end

function intervalsAreIntersecting(uio::ConcreteUIO, i::Int, j::Int)
    return uio.comparison_matrix[i, j] == INCOMPARABLE
end

function intervalIsToTheRight(uio::ConcreteUIO, i::Int, j::Int)
    return uio.comparison_matrix[i, j] == GREATER
end

function toPosetData(uio::ConcreteUIO, seq::Vector{Int})
    k = length(seq)
    return Tuple([uio.comparison_matrix[seq[i], seq[j]] for i in 1:k for j in i+1:k])
end

### SUB-UIO ###

function getsubuioencoding(uio::ConcreteUIO, seq::Vector{Int})

    function addupto(List::Vector{Int}, element::Int, uptoindex::Int)
        # adds element to List up to index uptoindex (exclusive)
        if length(List) == uptoindex
            return List
        end
        A = fill(element, uptoindex - length(List))
        return vcat(List, A)
    end

    N = length(seq)
    encod = Int[]
    for i in 1:N
        for j in i+1:N
            if uio.comparison_matrix[seq[i], seq[j]] != INCOMPARABLE  # not intersect
                encod = addupto(encod, i, j)
                break
            elseif j == N
                encod = addupto(encod, i, N)
        end
    end
    end
    if uio.comparison_matrix[seq[end], seq[end-1]] != INCOMPARABLE
        push!(encod, N)
    end

    return encod
end

function getsubUIO(uio::ConcreteUIO, seq::Vector{Int})
    return SubUIO(getsubuioencoding(uio, seq), seq)
end

struct SubUIO <: AbstractUIO
    rename::Dict{Int, Int}

    function SubUIO(uio_encoding::Vector{Int}, subseq::Vector{Int})
        super(uio_encoding)
        rename = Dict{Int, Int}()
        for (ID, globalID) in enumerate(subseq)
            rename[globalID] = ID
        end
        new(uio_encoding, subseq, rename)
    end

    function to_internal_indexing(subuio::SubUIO, seq::Vector{Int})
        return Tuple([subuio.rename[w] for w in seq])
    end
end

end  # module UIOJulia