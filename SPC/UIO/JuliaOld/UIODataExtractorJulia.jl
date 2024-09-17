module UIODataExtractorJulia

# Include the necessary module files
include("UIOJulia.jl")
include("CoreGeneratorJulia.jl")
include("EscherCoreGeneratorAbstractJulia.jl")
include("miscJulia.jl")

import .UIOJulia: AbstractUIO
import .CoreGeneratorJulia: CoreGenerator
using .UIOJulia
using .CoreGeneratorJulia
using .EscherCoreGeneratorAbstractJulia
using .miscJulia

# Import PyCall for lru_cache
#using PyCall
#@pyimport functools as pyfunctools

struct UIODataExtractor
    uio::AbstractUIO
    core_generator_class::Type{CoreGenerator}

    function UIODataExtractor(uio::AbstractUIO, core_generator_class::Type{CoreGenerator})
        new(uio, core_generator_class)
    end
end

function getCores(extractor::UIODataExtractor, partition)
    GEN = extractor.core_generator_class(extractor.uio, partition)
    cores = []
    if GEN isa EscherCoreGeneratorAbstract
        for escher in getEschers(extractor, partition)
            push!(cores, GEN.generateCore(escher))
        end
    # elseif GEN isa CorrectSequenceCoreGenerator
    #     for corseq in getCorrectSequences(extractor, partition)
    #         push!(cores, GEN.generateCore(corseq))
    #     end
    else
        error("$(extractor.core_generator_class) is not a subclass of EscherCoreGeneratorAbstract or CorrectSequenceCoreGenerator")
    end
    return cores
end

function getCoreRepresentations(extractor::UIODataExtractor, partition)
    GEN = extractor.core_generator_class(extractor.uio, partition)
    representations = []
    for core in getCores(extractor, partition)
        push!(representations, GEN.getCoreRepresentation(core))
    end
    return representations
end

function getEschers(extractor::UIODataExtractor, partition)
    P = getPermutationsOfN(extractor.uio.N)
    eschers = []
    if length(partition) == 1
        for seq in P
            if extractor.uio.isescher(seq)
                push!(eschers, seq)
            end
        end
    elseif length(partition) == 2
        a = partition[1]
        for seq in P
            if extractor.uio.isescher(seq[1:a]) && extractor.uio.isescher(seq[a+1:end])
                push!(eschers, (seq[1:a], seq[a+1:end]))
            end
        end
    elseif length(partition) == 3
        a, b, c = partition
        for seq in P
            if extractor.uio.isescher(seq[1:a]) && extractor.uio.isescher(seq[a+1:a+b]) && extractor.uio.isescher(seq[a+b+1:end])
                push!(eschers, (seq[1:a], seq[a+1:a+b], seq[a+b+1:end]))
            end
        end
    elseif length(partition) == 4
        a, b, c, d = partition
        for seq in P
            if extractor.uio.isescher(seq[1:a]) && extractor.uio.isescher(seq[a+1:a+b]) && extractor.uio.isescher(seq[a+b+1:a+b+c]) && extractor.uio.isescher(seq[a+b+c+1:end])
                push!(eschers, (seq[1:a], seq[a+1:a+b], seq[a+b+1:a+b+c], seq[a+b+c+1:end]))
            end
        end
    end
    return eschers
end

function getCoefficient(extractor::UIODataExtractor, partition)
    if length(partition) == 1
        return count(getCorrectSequences(extractor, partition))
    elseif length(partition) == 2
        return count(getCorrectSequences(extractor, partition)) - count(getCorrectSequences(extractor, (extractor.uio.N,)))
    elseif length(partition) == 3
        n, k, l = partition
        return 2 * countEschers(extractor, (n + k + l,)) + countEschers(extractor, partition) - countEschers(extractor, (n + l, k)) - countEschers(extractor, (n + k, l)) - countEschers(extractor, (l + k, n))
    elseif length(partition) == 4
        a, b, c, d = partition
        return countEschers(extractor, (a, b, c, d)) - countEschers(extractor, (a + b, c, d)) - countEschers(extractor, (a + c, b, d)) - countEschers(extractor, (a + d, b, c)) - countEschers(extractor, (b + c, a, d)) - countEschers(extractor, (b + d, a, c)) - countEschers(extractor, (c + d, a, b)) + countEschers(extractor, (a + b, c + d)) + countEschers(extractor, (a + c, b + d)) + countEschers(extractor, (a + d, b + c)) + 2 * countEschers(extractor, (a + b + c, d)) + 2 * countEschers(extractor, (a + b + d, c)) + 2 * countEschers(extractor, (a + c + d, b)) + 2 * countEschers(extractor, (b + c + d, a)) - 6 * countEschers(extractor, (a + b + c + d,))
    end
end

#@pyfunctools.lru_cache(maxsize=None)
function countEschers(extractor::UIODataExtractor, partition)
    return count(getEschers(extractor, partition))
end

function Base.show(io::IO, extractor::UIODataExtractor)
    println(io, "EXTRACTOR OF [", extractor.uio.encoding, "]")
end

end  # module UIODataExtractorJulia