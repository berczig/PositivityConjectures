module UIODataExtractorJulia

# Python calls
using PyCall
@pyimport SPC.UIO.cores.CorrectSequenceCoreGenerator
@pyimport SPC.UIO.cores.CorrectSequenceCoreGeneratorAbstract
@pyimport SPC.UIO.cores.EscherCoreGeneratorBasic
@pyimport SPC.UIO.cores.EscherCoreGeneratorTripple
@pyimport SPC.UIO.cores.EscherCoreGeneratorAbstract

# Include the module file

include("UIOJulia.jl")
include("CoreGeneratorJulia.jl")
include("miscJulia.jl")

# Julia calls
using CoreGeneratorJulia
using UIOJulia
using miscJulia
using Memoize


"""
    struct UIODataExtractor
The `UIODataExtractor` is focused on a specific UIO. It can generate and keep track of all possible λ-eschers and λ-correct sequences of the UIO and generates the cores using CoreGenerator.
Note: λ-correct sequences are returned as 1 sequence whereas λ-eschers are returned as tuples: λ-escher = (escher_1, escher_2)
"""
mutable struct UIODataExtractor
    uio::UIO
    core_generator_class::Type{CoreGenerator}

    function UIODataExtractor(uio::UIO, core_generator_class::Type{CoreGenerator})
        new(uio, core_generator_class)
    end
end

@memoize function getCorrectSequences(extractor::UIODataExtractor, partition)
    P = getPermutationsOfN(extractor.uio.N)
    if length(partition) == 1
        for seq in P
            if extractor.uio.iscorrect(seq)
                yield(seq)
            end
        end
    elseif length(partition) == 2
        a = partition[1]
        for seq in P
            if extractor.uio.iscorrect(seq[1:a]) && extractor.uio.iscorrect(seq[a+1:end])
                yield(seq)
            end
        end
    end
end

function getCores(extractor::UIODataExtractor, partition)
    GEN = extractor.core_generator_class(extractor.uio, partition)
    if GEN isa EscherCoreGeneratorAbstract
        for escher in getEschers(extractor, partition)
            yield(GEN.generateCore(escher))
        end
    elseif GEN isa CorrectSequenceCoreGenerator
        for corseq in getCorrectSequences(extractor, partition)
            yield(GEN.generateCore(corseq))
        end
    else
        error("$(extractor.core_generator_class) is not a subclass of EscherCoreGeneratorAbstract or CorrectSequenceCoreGenerator")
    end
end

function getCoreRepresentations(extractor::UIODataExtractor, partition)
    GEN = extractor.core_generator_class(extractor.uio, partition)
    for core in getCores(extractor, partition)
        yield(GEN.getCoreRepresentation(core))
    end
end

@memoize function getEschers(extractor::UIODataExtractor, partition)
    P = getPermutationsOfN(extractor.uio.N)
    if length(partition) == 1
        for seq in P
            if extractor.uio.isescher(seq)
                yield(seq)
            end
        end
    elseif length(partition) == 2
        a = partition[1]
        for seq in P
            if extractor.uio.isescher(seq[1:a]) && extractor.uio.isescher(seq[a+1:end])
                yield((seq[1:a], seq[a+1:end]))
            end
        end
    elseif length(partition) == 3
        a, b, c = partition
        for seq in P
            if extractor.uio.isescher(seq[1:a]) && extractor.uio.isescher(seq[a+1:a+b]) && extractor.uio.isescher(seq[a+b+1:end])
                yield((seq[1:a], seq[a+1:a+b], seq[a+b+1:end]))
            end
        end
    end
end

function getCoefficient(extractor::UIODataExtractor, partition)
    if length(partition) == 1
        return count(getCorrectSequences(extractor, partition))
    elseif length(partition) == 2
        return count(getCorrectSequences(extractor, partition)) - count(getCorrectSequences(extractor, (extractor.uio.N,)))
    elseif length(partition) == 3
        n, k, l = partition
        return 2 * count(getEschers(extractor, (n + k + l,))) +
               count(getEschers(extractor, partition)) -
               count(getEschers(extractor, (n + l, k))) -
               count(getEschers(extractor, (n + k, l))) -
               count(getEschers(extractor, (l + k, n)))
    end
end

function Base.show(io::IO, extractor::UIODataExtractor)
    print(io, "EXTRACTOR OF [", extractor.uio.encoding, "]")
end

end # module SPC.UIO.Julia