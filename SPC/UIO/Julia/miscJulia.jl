module miscJulia

using Combinatorics: permutations

# COMBINATORICS FUNCTIONS
function getPermutationsOfN(n::Int)
    return permutations(1:n)
end

function count(iterable)
    return sum(1 for _ in iterable)
end

end  # module miscJulia