NO_SUM = 0
EMPTY_SUM = 1
SUM_WITHOUT = 2
SUM_ONLY = 3
SUM_BOTH = 4

import numpy as np

def isThereSubsetSum(arr, sum):
     ##
     ## sum > 0 and zeros only at the end of arr!
     ##


    # d[i][j]=0: not possible to sum to j
    # d[i][j]=1: possible as empty sum
    # d[i][j]=2: possible to sum to j, but without arr[i-1]
    # d[i][j]=3: possible to sum to j and you need arr[i-1]
    # d[i][j]=4: possible to sum to j with and without arr[i-1]

    n = len(arr)
    dp =([[False for i in range(sum + 1)]
            for i in range(len(arr) + 1)])
    # dp[i][j] = true iff there is a subset in arr[0], ..., arr[i-1] that sums to j
     
    for i in range(n + 1):
        dp[i][0] = EMPTY_SUM
         
    for i in range(1, sum + 1):
         dp[0][i]= NO_SUM
             
   
    for i in range(1, n + 1):
        for j in range(1, sum + 1):
            #print((i, j), arr[i-1])
            if j<arr[i-1]:
                if dp[i-1][j] == NO_SUM:
                    dp[i][j] = NO_SUM
                else:
                    dp[i][j] = SUM_WITHOUT

            if j>= arr[i-1]:
                if dp[i-1][j] > NO_SUM and dp[i - 1][j-arr[i-1]] == NO_SUM:
                    dp[i][j] = SUM_WITHOUT
                elif dp[i-1][j] > NO_SUM and dp[i - 1][j-arr[i-1]] > NO_SUM:
                    dp[i][j] = SUM_BOTH

                elif dp[i-1][j] == NO_SUM and dp[i - 1][j-arr[i-1]] > NO_SUM:
                    dp[i][j] = SUM_ONLY
                elif dp[i-1][j] == NO_SUM and dp[i - 1][j-arr[i-1]] == NO_SUM:
                    dp[i][j] = NO_SUM
            #print(dp[i][j])

                #dp[i][j] = (dp[i-1][j] or
                                #dp[i - 1][j-arr[i-1]])
     
    return dp


def checkSum(arr, indices, target_sum):
    return sum([arr[i] for i in indices]) == target_sum

def getSolutions(arr, dp, sum):
    yield from getSolutionsRec(arr, dp, [], len(arr), sum)

def getSolutionsRec(arr, dp, solution:list, i, j):
    val = dp[i][j]
    #print(i, j, val)

    if val == NO_SUM:
        assert False == True, "how"

    elif val == EMPTY_SUM:
        yield solution
        
    elif val == SUM_WITHOUT:
        yield from getSolutionsRec(arr, dp, solution, i-1, j)

    elif val == SUM_ONLY:
        solution.append(i-1)
        yield from getSolutionsRec(arr, dp, solution, i-1, j - arr[i-1])

    elif val == SUM_BOTH:
        solution_copy = solution[:]

        # take the value
        solution.append(i-1)
        yield from getSolutionsRec(arr, dp, solution, i-1, j - arr[i-1])
        solution = solution_copy[:]

        # dont take it, exclude it
        yield from getSolutionsRec(arr, dp, solution, i-1, j)
        solution = solution_copy[:]

def solveCombinedSubsetSumProblem(matrix, target_sums):
    n_sums = len(target_sums)
    
    # after most non-zero entries in row (given that the coef is non-zero)
    A = list(zip(target_sums, matrix))
    A.sort(key= lambda x: (-100000 if x[0] != 0 else 0) -sum([1 for y in x[1] if y != 0]) )
    matrix = [x[1] for x in A]
    target_sums = [x[0] for x in A]

    # rearrange number such that the first row has all zero values in the end
    matrix = np.array(matrix)
    print("{} different addition problems with {} numbers".format(*matrix.shape))
    first_row = matrix[0]
    non_zero_indices = [i for i in range(len(first_row)) if first_row[i] != 0]
    zero_indices = [i for i in range(len(first_row)) if first_row[i] == 0]
    matrix = np.concatenate((matrix[:, non_zero_indices], matrix[:, zero_indices]), axis=1).tolist()

    if True:
        print("number matrix with sum:")
        #for i, x in enumerate(matrix):
            #print(target_sums[i], x)
    print("first_row:", matrix[0], target_sums[0])
    row_1_matrix = isThereSubsetSum(matrix[0], target_sums[0])
    global_solutions = []
    first_row_solutions = 0
    for i, solution in enumerate(getSolutions(matrix[0], row_1_matrix, target_sums[0])):
        if i % 10000 == 0:
            print("checking sol number {}".format(i))
        first_row_solutions += 1
        sol_works_for_all = True
        for i in range(1, n_sums):
            row = matrix[i]
            if checkSum(row, solution, target_sums[i]) == False:
                sol_works_for_all = False
                break
        if sol_works_for_all:
            global_solutions.append(solution)
    print("{} solutions for the first row".format(first_row_solutions))
    return global_solutions

def loadCSVFile(csv_file):
    target_sums = []
    numbers = []
    with open(csv_file, "r") as file:
        for raw_line in file.readlines():
            line = raw_line.strip().split(",")
            target_sums.append( int(line[0]) )
            numbers.append( [int(x) for x in line[1:]] )
    return numbers, target_sums

def solveOnCSVFile(csv_file):
    target_sums = []
    numbers = []
    with open(csv_file, "r") as file:
        for raw_line in file.readlines():
            line = raw_line.strip().split(",")
            target_sums.append( int(line[0]) )
            numbers.append( [int(x) for x in line[1:]] )
    solutions = solveCombinedSubsetSumProblem(numbers, target_sums)
    print("global solutions:")
    for sol in solutions:
        print(">", sol)

def test1():
    matrix = [[0, 0, 2,5,6,6,3,1,5,3], [0, 0, 2,5,6,6,4,1,5,3], [0, 0, 2,1,0,4, 1, 1, 1, 1]]
    scoresums = [21, 22, 7]
    print("global solutions:")
    for sol in solveCombinedSubsetSumProblem(matrix, scoresums):
        print(sol)

def checkforimpossible(numbers, target_sums, threshold = 0):
    exclude_columns = []
    for i in range(len(target_sums)):
        row = numbers[i]
        for j in range(len(row)):
            if row[j] > target_sums[i] + threshold:
                if j not in exclude_columns:
                    exclude_columns.append(j)

    print("exclude_columns:", exclude_columns)
    numbers = np.array(numbers)

    include_columns = []
    for i in range(len(numbers[0])):
        if i not in exclude_columns:
            include_columns.append(i)
    print("from {} columns we will only include {}:".format(len(numbers[0]), len(include_columns)))
    print("include columns:", include_columns)
    print("exclude columns:", exclude_columns)

    X = numbers[:, include_columns]
    overshoot = np.sum(X, axis=1) - np.array(target_sums)
    print("Sum of remaining occurences minus the target value (true coefficient):", list(overshoot))
    print("Sum of all the negative numbers:", np.sum(overshoot[overshoot < 0]))

def explain(path, threshold):
    coretype_occurences, coefficients = loadCSVFile(path)
    print("We have {} different core types".format(len(coretype_occurences[0])))
    #for i, c_occur in enumerate(coretype_occurences):
        #if coefficients[i] != 0:
            #print("The {}th UIO has the coefficient {} and we have the following coretype occurences:".format(i+1, coefficients[i]))
            #print(c_occur)
    checkforimpossible(coretype_occurences, coefficients, threshold=threshold)


#test1()
solveOnCSVFile("SPC/Saves,Tests/subsetsum/subsetsum3_2_1_symmetric.txt")
#checkforimpossible(*loadCSVFile("SPC/Saves,Tests/subsetsum/subsetsum5_4.txt"))
#explain("SPC/Saves,Tests/subsetsum/subsetsum5_4.txt")
#explain("SPC/Saves,Tests/subsetsum/subsetsum3_2_1_symmetric.txt", 0)
# unter 2/3/4 kommen nur noch 2/4
# über 4 muss 2/3/4 sein