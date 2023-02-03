import numpy as np
from numba import njit
import numba
import os
from itertools import permutations, combinations
from collections import defaultdict
import time
import pickle


@njit
def catalan_recurse(catalan_sequences, partial_sequence, n, curr_index):
    """
    The catalan_recurse function takes in a list of Catalan sequences (catalan_sequences),
    a partial Catalan sequence (partial_sequence), the desired length of the Catalan 
    sequence (n), and an index (curr_index). 
    
    It returns the list of Catalan sequences and the updated index.
    """
    prev_value = partial_sequence[-1] if len(partial_sequence) > 0 else 0
    partial_sequence = np.append(partial_sequence, prev_value)
    for i in range(prev_value, len(partial_sequence)):
        partial_sequence[-1] = i
        if len(partial_sequence) < n:
            catalan_sequences, curr_index = catalan_recurse(catalan_sequences, partial_sequence, n, curr_index)
        else:
            catalan_sequences[curr_index] = partial_sequence
            curr_index += 1
    return catalan_sequences, curr_index 

def catalan(n):
    """
    Returns a list of all length n Catalan sequences
    That is, all sequences of the form a_1, ..., a_n such that 
        a_{i+1} >= a_i for all i = 1 ... n-1, and
        a_i < i for all i = 1 ... n.

    This is a very slow implementation. Currently no need to improve,
    because this is not the bottleneck.
    
    Input: 
        n: the length of the sequence

    Output:
        catalan_sequences: a list containing all Catalan sequences, 
            as defined above
    """

    def catalan_number(n):
        """
        Calculate the n-th Catalan number using the recursive formula
        """

        if (n == 0 or n == 1):
            return 1
 
        # Table to store results of subproblems
        catalan_numbers_array = [0 for i in range(n + 1)]
    
        # Initialize first two values in table
        catalan_numbers_array[0] = 1
        catalan_numbers_array[1] = 1
    
        # Fill entries in catalan[] using recursive formula
        for i in range(2, n + 1):
            catalan_numbers_array[i] = 0
            for j in range(i):
                catalan_numbers_array[i] += catalan_numbers_array[j] * catalan_numbers_array[i-j-1]
    
        # Return last entry
        return catalan_numbers_array[n]

    catalan_sequences, _ = catalan_recurse(np.zeros(shape = [catalan_number(n),n],\
                                     dtype=np.int32), np.array([], dtype=np.int32), n, curr_index = 0)
    return catalan_sequences


@njit
def find_adjacency_matrix(input_sequence):
    """
    Given an input sequence of a unit interval graph of length n, finds the 
    corresponding adjacency matrix.
    
    The rule is: for i<j,
    - if i >= input_sequence[j] then i,j intersect and we set adjmat[i,j] = adjmat[j,i] = -1
    - otherwise interval i comes before interval j and we set adjmat[i,j] = adjmat[j,i] = j

    The diagonal entries are all -1.

    Input:
        a sequence encoding a unit interval graph
    Output:
        the "adjacency matrix" of this interval graph, with the above rules

    """
    n = len(input_sequence)
    adjmat = np.zeros(shape = (n,n), dtype=np.int32)
    for i in range(n):
        adjmat[i,i] = -1
        for j in range(i+1,n):
            if i >= input_sequence[j]:
                #they intersect
                adjmat[i,j] = -1
                adjmat[j,i] = -1
            else:
                adjmat[i,j] = j
                adjmat[j,i] = j
    return adjmat



@njit
def check_if_order_correct(graph, permutation):
    """
    This function takes in two arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    permutation: a 1D NumPy array representing a permutation of the vertices of the given graph.

    It returns a boolean value indicating whether the given permutation is correctly ordered or not. 
    A permutation is correctly ordered if for every pair of adjacent vertices i and j in the 
    permutation, if j is connected to i in the graph, then j should come before i in the permutation.
    """
    n = len(permutation)
    for i in range(n-1):
        #w_i+1 is not before w_i
        if graph[permutation[i], permutation[i+1]] == permutation[i]:
            return False
    return True


@njit
def check_if_connected_interval(graph, permutation):
    """
    This function takes in two arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    permutation: a 1D NumPy array representing a permutation of the vertices of the given graph.

    It returns a boolean value indicating whether the given permutation forms a connected 
    interval or not. A permutation forms a connected interval if for every vertex i in the 
    permutation, there exists at least one vertex j before it in the permutation such that 
    j is connected to i in the graph.
    """
    n = len(permutation)
    for i in range(1,n):
        #w_i+1 intersects w_1 or ... or w_i
        this_i_works = False
        for j in range(i):
            if graph[permutation[i], permutation[j]] == -1:
                this_i_works = True
                break
        if not this_i_works:
            return False
    return True


@njit
def correct(graph, permutation):
    """
    This function takes in two arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    permutation: a 1D NumPy array representing a permutation of the vertices of the given graph.

    It returns a boolean value indicating whether the given permutation is both 
    correctly ordered and forms a connected interval.
    """    
    order_correct = check_if_order_correct(graph, permutation)

    if not order_correct:
        return False
    
    connected_interval = check_if_connected_interval(graph, permutation)

    if not connected_interval:
        return False
    
    return True



def find_all_correct_sequences(input_sequence, all_permutations):
    """
    This function takes in two arguments:

    input_sequence: a 1D NumPy array representing the input sequence.
    all_permutations: a list of 1D NumPy arrays representing all 
    possible permutations of the input sequence.

    It returns a 1D NumPy array containing all permutations that are both 
    correctly ordered and form a connected interval, sorted in ascending order. 
    The adjacency matrix of the graph is first constructed from the input sequence, 
    and then each permutation is checked for correctness using the correct() function.
    """
    graph = find_adjacency_matrix(input_sequence)
    correct_permutations = set()


    for permutation in all_permutations:
        if correct(graph, permutation):
            correct_permutations.add(permutation)
    return np.array(sorted(correct_permutations))


@njit
def check_if_order_correct_sublist(graph, permutation, sublist):
    """
    This function takes in three arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    permutation: a 1D NumPy array representing a permutation of the vertices of the given graph.
    sublist: a 1D NumPy array of integers representing a sublist of indices in the permutation.

    It returns a boolean value indicating whether the given sublist of the permutation is 
    correctly ordered or not. A permutation is correctly ordered if for every pair of adjacent 
    vertices i and j in the permutation, if j is connected to i in the graph, then j should come 
    before i in the permutation.
    """
    for i in range(len(sublist)-1):
        #w_i+1 is not before w_i
        if graph[permutation[sublist[i]], permutation[sublist[i+1]]] == permutation[sublist[i]]:
            return False
    return True

@njit
def check_if_connected_interval_sublist(graph, permutation, sublist):
    """
    This function takes in three arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    permutation: a 1D NumPy array representing a permutation of the vertices of the given graph.
    sublist: a 1D NumPy array of integers representing a sublist of indices in the permutation.

    It returns a boolean value indicating whether the given sublist of the permutation 
    forms a connected interval or not. A permutation forms a connected interval if for 
    every vertex i in the permutation, there exists at least one vertex j before it in 
    the permutation such that j is connected to i in the graph.
    """
    for i in range(1,len(sublist)):
        #w_i+1 intersects w_1 or ... or w_i
        this_i_works = False
        for j in range(i):
            if graph[permutation[sublist[i]], permutation[sublist[j]]] == -1:
                this_i_works = True
                break
        if not this_i_works:
            return False
    return True

@njit
def correct_sublist(graph, permutation, sublist):
    """
    This function takes in three arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    permutation: a 1D NumPy array representing a permutation of the vertices of the given graph.
    sublist: a 1D NumPy array of integers representing a sublist of indices in the permutation.

    It returns a boolean value indicating whether the given sublist of the permutation 
    is both correctly ordered and forms a connected interval.
    """

    order_correct = check_if_order_correct_sublist(graph, permutation, sublist)

    if not order_correct:
        return False
    
    connected_interval = check_if_connected_interval_sublist(graph, permutation, sublist)

    if not connected_interval:
        return False
    
    return True


def find_all_correct_l_k_sequences(input_sequence, a, b, all_permutations):
    """
    This function takes in four arguments:

    input_sequence: a 1D NumPy array representing the input sequence.
    a: an integer representing the number of elements in the first sublist.
    b: an integer representing the number of elements in the second sublist.
    all_permutations: a list of 1D NumPy arrays representing all possible permutations of the input sequence.

    It returns a 1D NumPy array containing all permutations such that the first a elements and the 
    last b elements of the permutation are both correctly ordered and form a connected interval, 
    sorted in ascending order. The adjacency matrix of the graph is first constructed from the input 
    sequence, and then each permutation is checked for correctness using the correct_sublist() function.    
    """
    n = len(input_sequence)

    graph = find_adjacency_matrix(input_sequence)
    correct_permutations = set()

    for permutation in all_permutations:
        if correct_sublist(graph, permutation, np.arange(a)) and correct_sublist(graph, permutation, np.arange(a,a+b)):
            correct_permutations.add(permutation)
    return np.array(sorted(correct_permutations))


@njit
def find_critical_pairs(graph, correct_sequence, last_how_many, how_many_critical_pairs, critical_points):
    """
    This function takes in five arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    correct_sequence: a 1D NumPy array representing a correctly ordered and connected 
                        interval permutation of the vertices of the given graph.
    last_how_many: an integer representing the number of elements in the last sublist.
    how_many_critical_pairs: an integer representing the number of critical pairs to be found.
    critical_points: a 1D NumPy array representing the initial critical points.

    It returns a 1D NumPy array containing the critical points, which are the vertices 
    that participate in the given number of critical pairs in the given correctly ordered 
    and connected interval permutation. A critical pair is a pair of vertices i and j such 
    that i comes before j in the permutation and i is connected to j in the graph.
    """
    n = len(correct_sequence)
    counter = 0
    done = False
    for i in range(n-last_how_many-1,0,-1):
        if graph[correct_sequence[i],correct_sequence[i-1]] == -1 and (not done):
            counter += 1
            
            critical_points = np.concatenate((np.array([correct_sequence[i]]), critical_points))
            critical_points = np.concatenate((np.array([correct_sequence[i-1]]), critical_points))
            
            if counter == how_many_critical_pairs:
                done = True
            #print(critical_points, counter, how_many_critical_pairs, done)

    return critical_points


def add_extra_point(correct_sequence,last_how_many, critical_points):
    """
    This function takes in three arguments:

    correct_sequence: a 1D NumPy array representing a correctly ordered and connected 
    interval permutation of the vertices of a given graph.
    last_how_many: an integer representing the number of elements in the last sublist.
    critical_points: a 1D NumPy array representing the current critical points.

    It returns a 1D NumPy array containing the critical points, with an additional vertex 
    added if it is not already present in the critical points. The additional vertex is 
    the largest vertex in the first l-1 elements of the given correctly ordered and connected 
    interval permutation.
    """
    largest_entry_in_first_l_minus_one = np.max([correct_sequence[0:-last_how_many-1]])
    #print(largest_entry_in_first_l_minus_one)
    if not (largest_entry_in_first_l_minus_one in critical_points):
        critical_points = np.concatenate((np.array([largest_entry_in_first_l_minus_one]), critical_points))
    return critical_points


def find_critical_points(graph, correct_sequence, last_how_many, how_many_critical_pairs):
    """
    This function takes in four arguments:

    graph: a 2D NumPy array representing an adjacency matrix of a graph.
    correct_sequence: a 1D NumPy array representing a correctly ordered and connected 
                            interval permutation of the vertices of the given graph.
    last_how_many: an integer representing the number of elements in the last sublist.
    how_many_critical_pairs: an integer representing the number of critical pairs to be found.

    It returns a 1D NumPy array containing the critical points, which are the vertices that 
    participate in the given number of critical pairs in the given correctly ordered and 
    connected interval permutation. The initial critical points are the last l elements of 
    the permutation. The function also adds an extra vertex to the critical points if it is 
    not already present. The additional vertex is the largest vertex in the first l-1 elements 
    of the given correctly ordered and connected interval permutation. The critical points are 
    then found using the find_critical_pairs() function.
    """

    critical_points = np.array(correct_sequence[-last_how_many-1:])
    critical_points = add_extra_point(correct_sequence,last_how_many, critical_points)
    critical_points = find_critical_pairs(graph, correct_sequence, last_how_many, how_many_critical_pairs, critical_points)

    return critical_points
    
@njit  
def get_fingerprint_adjacency_matrix(critical_points, graph):
    """
    This function takes in two arguments:

    critical_points: a 1D NumPy array representing the critical points of a correctly 
            ordered and connected interval permutation of the vertices of a given graph.
    graph: a 2D NumPy array representing an adjacency matrix of the given graph.

    It returns a 2D NumPy array representing the adjacency matrix of the graph formed 
    by the critical points. The edge between two vertices i and j in the new graph is 
        -1 if i is connected to j in the original graph, 
        i if j is before i in the original graph, and 
        j if i is before j in the original graph.
    """
    m = len(critical_points)
    new_adj_mat = np.zeros(shape=(m,m), dtype=np.int32)
    for i in range(m):
        for j in range(i+1,m):
            if graph[critical_points[i], critical_points[j]] == -1:
                new_adj_mat[i,j] = -1
            elif graph[critical_points[i], critical_points[j]] == critical_points[j]:
                new_adj_mat[i,j] = j
            else:
                new_adj_mat[i,j] = i
            new_adj_mat[j,i] = new_adj_mat[i,j]
    return new_adj_mat


def calculate_fingerprint(correct_sequence, sequence, last_how_many, how_many_critical_pairs):
    """
    This function takes in four arguments:

    correct_sequence: a 1D NumPy array representing a correctly ordered and connected 
                                interval permutation of the vertices of a given graph.
    sequence: a 1D NumPy array representing the input sequence used to construct the given graph.
    last_how_many: an integer representing the number of elements in the last sublist.
    how_many_critical_pairs: an integer representing the number of critical pairs to be found.

    It returns a tuple containing:

    a 1D NumPy array representing the fingerprint of the given correctly ordered and 
    connected interval permutation. The fingerprint is a sequence of values indicating 
    the relationships between the critical points in the permutation.

    a 1D NumPy array representing the critical points of the given correctly ordered and 
    connected interval permutation.

    The fingerprint is calculated by first constructing the adjacency matrix of the graph formed 
    by the critical points using the get_fingerprint_adjacency_matrix() function, and then 
    flattening it into a 1D array. The critical points are obtained by calling the find_critical_points() 
    function on the original adjacency matrix constructed from the input sequence.
    """
    graph = find_adjacency_matrix(sequence)
    critical_points = find_critical_points(graph, correct_sequence, last_how_many, how_many_critical_pairs)
    new_adj_mat = get_fingerprint_adjacency_matrix(critical_points, graph)

    fingerprint = np.array([new_adj_mat[i,j] for i in range(len(critical_points)) for j in range(i+1,len(critical_points))])
    return tuple(fingerprint), critical_points


def create_gurobi_file(lp_file_name, catalan_sequences, all_types_counter, final_matrix, coefficient_list):
    """
    The create_gurobi_file() function generates a Gurobi optimization problem in the form of 
    a .lp file. It takes in five arguments:

    lp_file_name: a string representing the name of the .lp file to be created.
    catalan_sequences: a list of all the Catalan sequences used in the optimization problem.
    all_types_counter: an integer representing the total number of different types of fingerprints.
    final_matrix: a 2D NumPy array representing the frequency of each fingerprint type in each Catalan sequence.
    coefficient_list: a list of the coefficients for each Catalan sequence.

    The function first opens the .lp file specified by lp_file_name and writes the objective function 
    to minimize the sum of all the error terms. It then writes the constraints for the optimization problem, 
    using the final_matrix array to specify the frequency of each fingerprint type in each Catalan sequence 
    and the coefficient_list to specify the corresponding coefficient. It also defines the variables as 
    binary and closes the file when all lines have been written.
    """

    f = open(lp_file_name, 'w')
    f.write("Minimize\n")
    for i in range(len(catalan_sequences)):
        f.write("e" + str(i) + " + ")
    f.write("z\nSubject To\nz = 0\n")
    for i in range(len(catalan_sequences)):
        for j in range(all_types_counter):
            if final_matrix[i,j] != 0:
                f.write(str(final_matrix[i,j]) + " x" + str(j) + " + ")
        f.write("z - e" + str(i) + " <= " + str(coefficient_list[i]) + "\n")
    
    for i in range(len(catalan_sequences)):
        for j in range(all_types_counter):
            if final_matrix[i,j] != 0:
                f.write(str(final_matrix[i,j]) + " x" + str(j) + " + ")
        f.write("z + e" + str(i) + " >= " + str(coefficient_list[i]) + "\n")

    f.write("Bounds\n")
    for i in range(len(catalan_sequences)):
        f.write("e" + str(i) + " >= 0\n")

    f.write("Binary\n")
    for j in range(all_types_counter):
        f.write("x"+str(j) + " ")
    f.write("\nEnd")
    f.close()


def decode_gurobi_solution(reverse_all_types, catalan_sequences, coefficient_list, dict_list):
    """
    The decode_gurobi_solution() function is used to decode the solution to an optimization 
    problem that was solved using the Gurobi solver. It takes in four arguments:

    reverse_all_types: a dictionary that maps integer IDs to fingerprints (which are tuples).
    catalan_sequences: a list of all the Catalan sequences used in the optimization problem.
    coefficient_list: a list of the coefficients for each Catalan sequence.
    dict_list: a list of dictionaries, where each dictionary maps fingerprints to their 
                                            frequency in a particular Catalan sequence.

    The function reads the solution from the file matrixsol.sol and decodes it by extracting 
    the integer IDs of the selected fingerprints. It then checks if the solution is valid by 
    comparing the computed sum of the frequencies of the selected fingerprints to the corresponding 
    coefficient for each Catalan sequence.

    The function first opens the file matrixsol.sol and reads each line. If a line starts with x, 
    it processes the line to extract the integer ID of the selected fingerprint. It adds the integer 
    ID to the solution_types list and closes the file when all lines have been read.

    Next, the function iterates through each Catalan sequence and its corresponding coefficient and 
    dictionary of fingerprint frequencies. It computes the sum of the frequencies of the selected 
    fingerprints in the dictionary and compares it to the coefficient. If the sums do not match, 
    it prints an error message; otherwise, it prints a success message.
    """
    solution_types = []
    f = open('matrixsol.sol', 'r')
    for line in f:
        if line[0] == "x":
            line_stripped = line.strip()
            line_parts = line_stripped.split()
            if line_parts[1] == "1":
                type_index = int(line_parts[0][1:])
                print(reverse_all_types[type_index])
                solution_types.append(type_index)
    print(solution_types)
    f.close()
    print()
    print(len(solution_types))
    print()

    #check that it really works

    something_is_wrong = False
    for i, sequence in enumerate(catalan_sequences):
        coeff = coefficient_list[i]
        types_count = dict_list[i]
        mysum = 0
        for item in solution_types:
            mysum += types_count[reverse_all_types[item]]
        if coeff != mysum:
            print(i, coeff, mysum)
            print("WRONG!")
            something_is_wrong = True
    if not something_is_wrong:
        print("Yep, this solution works!")

def save_variables_to_files(catalan_sequences, all_types_counter, coefficient_list, reverse_all_types, dict_list):
    with open("catalan_sequences.pickle", "wb") as f:
        pickle.dump(catalan_sequences, f)

    with open("all_types_counter.pickle", "wb") as f:
        pickle.dump(all_types_counter, f)

    with open("coefficient_list.pickle", "wb") as f:
        pickle.dump(coefficient_list, f)

    with open("reverse_all_types.pickle", "wb") as f:
        pickle.dump(reverse_all_types, f)

    with open("dict_list.pickle", "wb") as f:
        pickle.dump(dict_list, f)

    #To load files, do this:
    #with open("catalan_sequences.pickle", "rb") as f:
    #    catalan_sequences = pickle.load(f)


def calculate_coeff_values(dict_index, x, x_val, y, y_val, dict_list):
    """
    The function calculate_coeff_values takes in five arguments:

    dict_index: an integer representing the index of a dictionary in the list dict_list
    x: an integer representing an index in a tuple
    x_val: an integer representing a value that the element at index x in the tuple should have
    y: an integer representing an index in a tuple
    y_val: an integer representing a value that the element at index y in the tuple should have
    dict_list: a list of dictionaries, where each dictionary has keys that are tuples and 
                                                                values that are integers

    The function returns an integer representing the sum of the values in the dictionary 
    at index dict_index of dict_list, where the key tuples have their element at index x 
    equal to x_val and their element at index y equal to y_val.
    """
    types_count = dict_list[dict_index]
    summand = 0
    for dict_type in types_count.keys():
        if dict_type[x] == x_val and dict_type[y] == y_val:
            #print(dict_type, types_count[dict_type])
            summand += types_count[dict_type]
    return summand

def get_all_fingerprint_values(all_types):
    """
    This function get_all_fingerprint_values takes in a dictionary all_types and 
    returns a list of sets. Each set in the list contains all the possible values 
    that can appear in the corresponding position of a fingerprint in the dictionary.

    The dictionary all_types maps each fingerprint to a unique integer. A fingerprint 
    is a tuple of integers that represent the adjacency relationships between certain 
    points in a graph. The function returns a list of sets, where the i-th set in the 
    list contains all the values that can appear in the i-th position of a fingerprint 
    in the dictionary.

    For example, if all_types is a dictionary that maps the following fingerprints to 
    integers:

    (1, 2, 3) -> 0
    (2, 3, 4) -> 1
    (2, 3, 5) -> 2
    (1, 3, 5) -> 3

    Then get_all_fingerprint_values(all_types) would return [{1, 2}, {2, 3, 4, 5}, {3, 5}], 
    since the first position can contain 1 or 2, the second position can contain 2, 3, 4, or 5, 
    and the third position can contain 3 or 5.
    """

    max_fingerprint_length = -1
    for fingerprint in all_types.keys():
        if len(fingerprint) > max_fingerprint_length:
            max_fingerprint_length = len(fingerprint)
    
    allowed_values = [set() for _ in range(max_fingerprint_length)]
    for fingerprint in all_types.keys():
        for i, item in enumerate(fingerprint):
            allowed_values[i].add(item)
    
    return allowed_values


def calculate_fingerprint_value_vectors(fingerprint_permissible_values, dict_list):
    """
    The function calculate_fingerprint_value_vectors takes as input two lists: 
    fingerprint_permissible_values and dict_list. The first list is a list of sets, 
    where each set contains the values that are allowed at a certain index of a fingerprint. 
    The second input, dict_list, is a list of dictionaries, where each dictionary represents 
    a count of the occurrences of fingerprints of a certain type.

    The function calculates the fingerprint value vectors by iterating over all pairs of 
    indices first_index and second_index in fingerprint_permissible_values, as well as over 
    all pairs of values first_value and second_value in the sets at those indices. For each 
    such pair of indices and values, the function calculates a fingerprint vector by calling 
    the function calculate_coeff_values for each dictionary in dict_list, passing the current 
    indices and values as arguments to calculate_coeff_values. The resulting fingerprint vector 
    is then added to a dictionary with a key consisting of the current indices and values, and 
    the dictionary is returned at the end of the function.
    """
    fingerprint_vectors = dict()
    for first_index in range(len(fingerprint_permissible_values)):
        for second_index in range(first_index+1, len(fingerprint_permissible_values)):
            for first_value in fingerprint_permissible_values[first_index]:
                for second_value in fingerprint_permissible_values[second_index]:
                    fingerprint_vector = [calculate_coeff_values(i, first_index, first_value, second_index, second_value, dict_list) \
                            for i in range(len(dict_list))]
                    fingerprint_vectors[(first_index, second_index, first_value, second_value)] = np.array(fingerprint_vector, dtype=np.int32)
    return fingerprint_vectors


@njit
def is_disjoint_support_pair(vec1, vec2, all_types):
    for fingerprint in all_types:
        if fingerprint[vec1[0]] == vec1[2] and fingerprint[vec1[1]] == vec1[3]:
            if fingerprint[vec2[0]] == vec2[2] and fingerprint[vec2[1]] == vec2[3]:
                return False
    return True


def calculate_fingerprint_vector_disjointness_graph(fingerprint_value_vectors, all_types):
    
    all_types_array = np.array(list(all_types.keys()), dtype=np.int32)
    key_list = list(fingerprint_value_vectors.keys())
    disjointness_adjmat = np.zeros(shape=(len(key_list), len(key_list)))
    for index_pair in combinations(list(range(len(key_list))), 2):
        if index_pair[0]%100 == 0 and np.random.rand() < 0.0001:
            print(index_pair[0])
        keys = [key_list[item] for item in index_pair]
        if is_disjoint_support_pair(*[key for key in keys], all_types_array):
            disjointness_adjmat[index_pair[0], index_pair[1]] = 1
            disjointness_adjmat[index_pair[1], index_pair[0]] = 1
    return disjointness_adjmat



@njit
def sum_of_pair_is_correct_solution(first_vec, second_vec, coefficient_list):
    for i in range(len(first_vec)):
        if first_vec[i] + second_vec[i] != coefficient_list[i]:
            return False
    return True

@njit
def sum_of_triple_is_correct_solution(first_vec, second_vec, third_vec, coefficient_list):
    for i in range(len(first_vec)):
        if first_vec[i] + second_vec[i] + third_vec[i] != coefficient_list[i]:
            return False
    return True

def try_pairs_triples_of_fingerprint_value_vectors(fingerprint_value_vectors, coefficient_list, fingerprint_vector_disjointness_graph):
    """
    The function try_pairs_triples_of_fingerprint_value_vectors() takes in two arguments:

    fingerprint_value_vectors: a dictionary where the keys are tuples representing 
    pairs or triples of fingerprint type values, and the values are lists representing 
    the corresponding coefficient values for each Catalan sequence.

    coefficient_list: a list representing the coefficient values for each Catalan sequence.

    The function iterates through all pairs/triples of keys in fingerprint_value_vectors and 
    calculates the sum of the corresponding value vectors. If this sum is equal to 
    coefficient_list, it prints a message indicating that a solution has been found 
    and prints the pair of keys that resulted in the sum.

    The purpose of this function is to find a pair of fingerprint values that, when summed, 
    result in the coefficient values for each Catalan sequence.
    """
    
    key_list = list(fingerprint_value_vectors.keys())
    print(len(key_list))
    for index_pair in combinations(list(range(len(key_list))), 2):
        if index_pair[0]%100 == 0 and np.random.rand() < 0.0001:
            print(index_pair[0])
        if fingerprint_vector_disjointness_graph[index_pair[0], index_pair[1]] == 1:
            keys = [key_list[item] for item in index_pair]

        if sum_of_pair_is_correct_solution(*[fingerprint_value_vectors[key] for key in keys], coefficient_list):
            print("Solution found!!!")
            print(keys)

    for index1 in range(len(key_list)):
        for index2 in range(index1+1, len(key_list)):
            index_triple = [index1, index2, 0]
            if index_triple[0]%100 == 0 and np.random.rand() < 0.0000001:
                print(index_triple[0])

            if fingerprint_vector_disjointness_graph[index_triple[0], index_triple[1]] == 1:
                for index3 in range(index2+1, len(key_list)):
                    index_triple[2] = index3
                    if fingerprint_vector_disjointness_graph[index_triple[0], index_triple[2]] == 1 \
                     and fingerprint_vector_disjointness_graph[index_triple[2], index_triple[1]] == 1:
                        keys = [key_list[item] for item in index_triple]

                        if sum_of_triple_is_correct_solution(*[fingerprint_value_vectors[key] for key in keys], coefficient_list):
                            print("Solution found!!!")
                            print(keys)





def main():
    # Initialize variables
    a_val = 5
    b_val = 3
    num_critical_pairs = 2
    n = a_val + b_val
    catalan_sequences = catalan(n)  # Generate all Catalan sequences of length n
    all_types = dict()  # Dictionary to store the fingerprints and their corresponding integer IDs
    reverse_all_types = dict()  # Dictionary to store the integer IDs and their corresponding fingerprints
    all_types_counter = 0  # Counter to assign integer IDs to new fingerprints
    dict_list = []  # List to store the frequency of each fingerprint in each Catalan sequence
    coefficient_list = []  # List to store the coefficients for each Catalan sequence
    all_permutations = list(permutations(list(range(n))))  # Generate all permutations of the vertices of the graph
    all_permutations = [tuple(perm) for perm in all_permutations]  # Convert permutations to tuples

    # Iterate through each Catalan sequence
    for sequence in catalan_sequences:
        print(sequence)
        # Find all correct l+k sequences for the current Catalan sequence
        correct_sequences = find_all_correct_sequences(sequence, all_permutations)
        # Find all correct (l,k)-sequences for the current Catalan sequence
        correct_l_k_sequences = find_all_correct_l_k_sequences(sequence, a_val, b_val, all_permutations)
        #print("Number of correct (l,k)-sequences: " + str(len(correct_l_k_sequences)))
        #print("Number of correct l+k sequences: " + str(len(correct_sequences)))
        coeff = len(correct_l_k_sequences) - len(correct_sequences)
        coefficient_list.append(coeff)
        #print("The coefficient is: " + str(coeff))
        types_count = defaultdict(int)
        for correct_sequence in correct_l_k_sequences:
            fingerprint, _ = calculate_fingerprint(correct_sequence, sequence, last_how_many=b_val, how_many_critical_pairs=num_critical_pairs)
            types_count[fingerprint] += 1
            if not (fingerprint in all_types):
                all_types[fingerprint] = all_types_counter
                reverse_all_types[all_types_counter] = fingerprint
                all_types_counter += 1
            
        dict_list.append(types_count)

        

    coefficient_list = np.array(coefficient_list)
    final_matrix = np.zeros(shape=(len(catalan_sequences), all_types_counter), dtype=np.int32)
    for i in range(len(catalan_sequences)):
        for item in dict_list[i].keys():
            final_matrix[i,all_types[item]] = dict_list[i][item] 
    file_name = "matrix_" + str(a_val) + "_" + str(b_val) + ".npz"

    #Save the final matrix to a file using lzma compression
    np.savez_compressed(file_name, final_matrix, compress='lzma')
    print("Matrix calculation completed! Saved it to file.")
    print(f"Size of file: {os.path.getsize(file_name)} bytes")
    # Here is how to load the contents of "matrix.npz" into a NumPy array 
    # final_matrix = np.load(file_name)['arr_0']

    #Save all other important variables to file, using simple pickling
    save_variables_to_files(catalan_sequences, all_types_counter, coefficient_list, reverse_all_types, dict_list)

    print("The size of the matrix is: ")
    print(all_types_counter, len(catalan_sequences))

    #Uncomment to create gurobi file. Solve the gurobi file with the command
    # gurobi_cl ResultFile="matrixsol.sol" matrixlp.lp 
    create_gurobi_file("matrixlp.lp", catalan_sequences, all_types_counter, final_matrix, coefficient_list)

    #Uncomment to print out all the types in the gurobi solution
    #decode_gurobi_solution(reverse_all_types, catalan_sequences, coefficient_list, dict_list)

    fingerprint_permissible_values = get_all_fingerprint_values(all_types)
    fingerprint_value_vectors = calculate_fingerprint_value_vectors(fingerprint_permissible_values, dict_list)

    print("Calculating disjointness graph")
    fingerprint_vector_disjointness_graph = calculate_fingerprint_vector_disjointness_graph(fingerprint_value_vectors,all_types)

    print("Trying fingerprint type vector pairs now.")
    try_pairs_triples_of_fingerprint_value_vectors(fingerprint_value_vectors, coefficient_list, fingerprint_vector_disjointness_graph)

    """
    for i in range(3000):
        summand1 = calculate_coeff_values(i, 10, 4, 13, 5, dict_list)
        summand2 = calculate_coeff_values(i, 3, 0, 8, 1, dict_list)
        print(coefficient_list[i], summand1 + summand2)
    """

    """
    print("\n\nGergo's solution: \n")
    for dict_type in all_types.keys():
        if dict_type[10] == 4 and dict_type[13] == 5:
            print(all_types[dict_type], end=", ")
        if dict_type[3] == 0 and dict_type[8] == 1:
            print(all_types[dict_type], end=", ")
    """

    """
        summand1 = 0
        summand2 = 0
        for dict_type in types_count.keys():
            if dict_type[10] == 4 and dict_type[13] == 5:
                print(dict_type, types_count[dict_type])
                summand1 += types_count[dict_type]
            if dict_type[3] == 0 and dict_type[8] == 1:
                print(dict_type, types_count[dict_type])
                summand2 += types_count[dict_type]
        
        if coeff == summand1 + summand2:
            print(str(coeff) + " = " + str(summand1) + " + " + str(summand2))
        else:
            print(str(coeff) + " != " + str(summand1) + " + " + str(summand2))
            print("WRONG!")
            break
        print()
    """


tic = time.time()
main()
print("Execution time: " + str(time.time() - tic))
