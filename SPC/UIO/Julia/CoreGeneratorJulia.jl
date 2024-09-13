module CoreGeneratorJulia

include("UIOJulia.jl")

using UIOJulia

abstract type CoreGenerator end

function generateCore(self::CoreGenerator, seq)
    """
    abstract function
    """
    error("generateCore is an abstract function and should be implemented by a subclass.")
end

function compareTwoCoreElements(self::CoreGenerator, a, b)
    """
    abstract function
    """
    error("compareTwoCoreElements is an abstract function and should be implemented by a subclass.")
end

function getCoreComparisions(partition)
    """
    static method
    """
    error("getCoreComparisions is an abstract function and should be implemented by a subclass.")
end

function getCoreLabels(partition)
    """
    static method
    """
    error("getCoreLabels is an abstract function and should be implemented by a subclass.")
end

mutable struct CoreGeneratorImpl <: CoreGenerator
    uio::UIO
    partition::Any

    function CoreGeneratorImpl(uio::UIO, partition)
        obj = new(uio, partition)
        calculate_comp_indices(CoreGeneratorImpl, partition)
        return obj
    end
end

function calculate_comp_indices(cls::Type{CoreGeneratorImpl}, partition)
    # this attribute is common for all coregenerator even when they have different UIO. So it should be a class attribute
    if !haskey(cls, :comp_indices)
        comp_indices = []

        labels = getCoreLabels(partition)
        comp = getCoreComparisions(partition)

        for (first_index, first_label) in enumerate(labels)
            if haskey(comp, first_label)
                for second_label in comp[first_label]
                    second_index = findfirst(isequal(second_label), labels)
                    push!(comp_indices, (first_index, second_index))
                end
            end
        end

        # Set the calculated comp_indices for all CoreGenerators of this type (but not the base class)
        cls.comp_indices = comp_indices
        println("Reduced size of core representations: $(length(labels) * (length(labels) - 1) / 2) -> $(length(comp_indices))")
    end
end

function getCoreRepresentation(self::CoreGeneratorImpl, core)
    if core == "GOOD"
        return "GOOD"
    elseif core == "BAD"
        return "BAD"
    end

    return tuple([compareTwoCoreElements(self, core[f_index], core[s_index]) for (f_index, s_index) in self.comp_indices])
end

function getCoreRepresentationLength(cls::Type{CoreGeneratorImpl}, partition)
    labels = getCoreLabels(partition)
    comp = getCoreComparisions(partition)
    return sum([length(comp[label]) for label in labels if haskey(comp, label)])
end

function getCoreLength(cls::Type{CoreGeneratorImpl}, partition)
    labels = getCoreLabels(partition)
    @assert length(labels) == length(Set(labels)) "the labels are not unique"
    return length(labels)
end

function getAllCoreComparisions(cls::Type{CoreGeneratorImpl}, partition)
    comp = Dict{Any, Any}()
    labels = getCoreLabels(partition)
    for (index, label) in enumerate(labels)
        comp[label] = labels[index+1:end]
    end
    return comp
end

function convertCorerepToText(cls::Type{CoreGeneratorImpl}, corerep, partition)
    calculate_comp_indices(cls, partition)
    labels = getCoreLabels(partition)
    s = []
    for (index, (i, j)) in enumerate(cls.comp_indices)
        edge = corerep[index]
        push!(s, labels[i] * " " * UIO.RELATIONTEXT[edge] * " " * labels[j])
    end
    return join(s, "\n")
end

end # module SPC.UIO