module UIOJulia

using LinearAlgebra

const INCOMPARABLE = 100    # (i,j) is INCOMPARABLE if neither i < j nor j < i nor i = j -- connected in the incomparability graph -- intersecting intervals
const LESS = 101            # (i,j) is LE iff i < j     interval i is to the left of j 
const GREATER = 102         # (i,j) is LE iff i > j     interval i is to the right of j
const EQUAL = 103           # (i,j) is EQ iff i = j     interval i and interval j are same interval
const RELATIONTEXT = Dict(LESS => "<", EQUAL => "=", GREATER => ">")

mutable struct UIO
    N::Int
    encoding::Vector{Int}
    repr::String
    comparison_matrix::Matrix{Int}

    function UIO(uio_encoding::Vector{Int})
        N = length(uio_encoding)
        encoding = uio_encoding
        repr = string(encoding)

        # decode encoding to get comparison matrix
        comparison_matrix = fill(EQUAL, N, N) # (i,j)'th index says how i is in relation to j
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

function Base.show(io::IO, uio::UIO)
    print(io, uio.repr)
end

### POSET STRUCTURE ###

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
        # 1) arrow condition
        if !isarrow(self, seq, i-1, i)
            return false
        end
        # 2) intersects with some previous interval
        intersects = false
        for j in 1:i-1
            if self.comparison_matrix[seq[i], seq[j]] in [LESS, INCOMPARABLE]
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

function isarrow(self::UIO, escher::Vector{Int}, i::Int, j::Int, verbose::Bool=false)
    if verbose
        println("arrow", escher, i, j, self.comparison_matrix[escher[i], escher[j]] != GREATER)
    end
    return self.comparison_matrix[escher[i], escher[j]] != GREATER # EQUAL also intersects
end

function intervalsAreIntersecting(self::UIO, i::Int, j::Int)
    return self.comparison_matrix[i, j] == INCOMPARABLE
end

function intervalIsToTheRight(self::UIO, i::Int, j::Int)
    return self.comparison_matrix[i, j] == GREATER
end

function toPosetData(self::UIO, seq::Vector{Int})
    k = length(seq)
    return tuple([self.comparison_matrix[seq[i], seq[j]] for i in 1:k for j in i+1:k]...)
end

### SUB-UIO ###

function getsubuioencoding(self::UIO, seq::Vector{Int})

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
            if self.comparison_matrix[seq[i], seq[j]] != INCOMPARABLE
                encod = addupto(encod, i, j)
                break
            elseif j == N
                encod = addupto(encod, i, N)
        end
    end
    if self.comparison_matrix[seq[end], seq[end-1]] != INCOMPARABLE
        push!(encod, N)
    end

    return encod
end

function getsubUIO(self::UIO, seq::Vector{Int})
    return SubUIO(getsubuioencoding(self, seq), seq)
end

mutable struct SubUIO <: UIO
    rename::Dict{Int, Int}

    function SubUIO(uio_encoding::Vector{Int}, subseq::Vector{Int})
        obj = new(UIO(uio_encoding))
        obj.rename = Dict{Int, Int}()
        for (ID, globalID) in enumerate(subseq)
            obj.rename[globalID] = ID
        end
        return obj
    end
end

function to_internal_indexing(self::SubUIO, seq::Vector{Int})
    return tuple([self.rename[w] for w in seq]...)
end

end # module UIOJulia
end