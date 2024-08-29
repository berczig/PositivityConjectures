using Base.Iterators: permutations

# COMBINATORIC FUNCTIONS
# function getPermutationsOfN(n::Int)
#     return permutations(1:n)
# end

function getPermutationsOfN(n::Int)
    return permutations(1:n)
end

function count(iterable)
    return sum(1 for _ in iterable)
end