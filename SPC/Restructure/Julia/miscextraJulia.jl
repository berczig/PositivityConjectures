using Serialization

abstract type Loadable end

function save(instance::Loadable, filename::String)
    # saves all currently present attributes of the instance
    open(filename, "w") do f
        serialize(f, instance)
    end
end

function load(instance::Loadable, filename::String)
    # loads dump and sets all attributes of dump as attributes of the current instance
    open(filename, "r") do f
        loaded_instance = deserialize(f)
        for var in fieldnames(typeof(loaded_instance))
            setfield!(instance, var, getfield(loaded_instance, var))
        end
    end
end

mutable struct PartiallyLoadable <: Loadable
    saveable_variables::Vector{Symbol}
    _savehelper::Loadable

    function PartiallyLoadable(saveable_variables::Vector{Symbol})
        new(saveable_variables, Loadable())
    end
end

function save(instance::PartiallyLoadable, filename::String)
    for var in instance.saveable_variables
        setfield!(instance._savehelper, var, getfield(instance, var))
    end
    save(instance._savehelper, filename)
end

function load(instance::PartiallyLoadable, filename::String)
    load(instance._savehelper, filename)
    for var in instance.saveable_variables
        setfield!(instance, var, getfield(instance._savehelper, var))
    end
end