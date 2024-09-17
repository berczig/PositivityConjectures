module miscextraJulia

using Serialization

abstract type AbstractLoadable end

mutable struct Loadable <: AbstractLoadable
end

function save(self::AbstractLoadable, filename::String)
    open(filename, "w") do f
        serialize(f, self)
    end
end

function load(self::AbstractLoadable, filename::String)
    open(filename, "r") do f
        loaded_instance = deserialize(f)
        for var in fieldnames(typeof(loaded_instance))
            setfield!(self, var, getfield(loaded_instance, var))
        end
    end
end

abstract type AbstractPartiallyLoadable end

mutable struct PartiallyLoadable <: AbstractPartiallyLoadable
    saveable_variables::Vector{Symbol}
    default_values::Dict{Symbol, Any}
    _savehelper::AbstractLoadable

    function PartiallyLoadable(saveable_variables::Vector{Symbol}, default_values::Dict{Symbol, Any} = Dict{Symbol, Any}())
        new(saveable_variables, default_values, Loadable())
    end
end

function save(self::PartiallyLoadable, filename::String)
    for var in self.saveable_variables
        setfield!(self._savehelper, var, getfield(self, var))
    end
    save(self._savehelper, filename)
end

function load(self::PartiallyLoadable, filename::String)
    load(self._savehelper, filename)
    for var in self.saveable_variables
        if hasfield(typeof(self._savehelper), var)
            setfield!(self, var, getfield(self._savehelper, var))
        else
            value = get(self.default_values, var, nothing)
            println("\n#####\nTHE FILE \"$filename\" IS OUTDATED!\nIT IS MISSING THE ATTRIBUTE \"$var\"\nSETTING $var = $value\n#####\n")
            setfield!(self, var, value)
        end
    end
end

end  # module miscextraJulia