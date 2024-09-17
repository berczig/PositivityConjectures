module EscherCoreGeneratorAbstractJulia

# Include the CoreGenerator module
include("CoreGeneratorJulia.jl")

import .CoreGeneratorJulia: CoreGenerator
using .CoreGeneratorJulia

# Define the EscherCoreGeneratorAbstract type


struct EscherCoreGeneratorAbstract <: CoreGenerator
    # No additional fields or methods are defined, similar to the Python class
end

end  # module EscherCoreGeneratorAbstract