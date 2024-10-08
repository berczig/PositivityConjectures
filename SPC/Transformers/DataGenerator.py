# Description: This file contains the functions to read the data from the csv file and tokenize the data.
# It also contains the functions to encode the tokenized data into integers, and provide training data for GPTMiniFloat
# The rows of input file have the following format:
# #  "(a_1,\ldots, a_n)",y  
# where a_1,\ldots, a_n are floats and y is a float. 
# Output of read_and_tokenize_training_data_from_csv(file_path) is the pair (X,Y) where X is the list of tokenized input sequences (tokenization uses characteras in scientific notation) and Y is the tokenized list of y's.
# Output of encode_tokenized_data(X_tokenized_data) is the training pair (X,Y) for MiniGPT, it assigns integer indeces to each token in X and Y.


import torch
import csv

file_path = "SPC/Transformers/uio_data_n=9.csv"


import pandas as pd

def csv_to_dataframe(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    return df

# Read the file and create the training data
def getTrainingDataFromFile(file_path, partition):
    df = csv_to_dataframe(file_path)
    Xuio = df['encodings'].apply(eval).tolist()  # Convert string representation of list back to list
    Yuio = df[str(partition)]

    Xuio = [list(map(int, x)) for x in Xuio]
    
    Xuio = torch.tensor(Xuio)
    Yuio = torch.tensor(Yuio).unsqueeze(1)
    
    print(Xuio.shape, Yuio.shape)
    print(Xuio, Yuio)
    
    return Xuio, Yuio


# Function whose input is a float number x, the output is the list of characters in the scientific notation of x.

def float_to_scientific_notation_list(x):
    # Step 2: Convert the float number to scientific notation string
    scientific_notation_str = "{:.6e}".format(x)
    
    # Step 3: Convert the string to a list of characters
    char_list = list(scientific_notation_str)
    
    # Step 4: Return the list of characters
    return char_list


chars = ['0','1','2','3','4','5','6','7','8','9','e','.','-','+','*'] # character vocabulary
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# Print integer values of characters

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
#data = torch.tensor(encode(text), dtype=torch.long)


# Read the file and create the tokenized training data. The input file lines have the format:
# "(float1,float2,...,float_n)",float 
# The output is [float_to_scientific_notation_list(float1), float_to_scientific_notation_list(float2), ..., float_to_scientific_notation_list(float_n)], float_to_scientific_notation_list(float)


def read_and_tokenize_training_data_from_csv(file_path, partition):
    X_tokenized_data = []
    Y_tokenized_data = []
    Y_float = []
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Convert the 'encodings' column from string representation of list back to list
    df['encodings'] = df['encodings'].apply(eval)
    
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        X_row = []
        
        # Extract the tuple of floats and the single float
        tuple_part = row['encodings']  # 'encodings' column is already converted to list
        single_float_part = row[str(partition)]  

        # Convert the tuple part to a list of floats
        float_tuple = [float(x) for x in tuple_part]
        
        # Convert the single float part to a float
        single_float = float(single_float_part)
        
        # Tokenize the floats
        tokenized_tuple = [float_to_scientific_notation_list(f) for f in float_tuple]
        tokenized_single_float = float_to_scientific_notation_list(single_float)
        
        # Append a '*' and then elements of tokenized_tuple individually to the list, we keep the Y data as a float
        for f in tokenized_tuple:
            X_row.extend(f)
            X_row.append('*')
        Y_float.append(single_float)
        X_tokenized_data.append(X_row)
        
        # Append the tokenized data for the partition columns
        for col in df.columns[partition[0]:partition[1]]:
            partition_value = row[col]
            Y_tokenized_data.append(float_to_scientific_notation_list(float(partition_value)))

    return X_tokenized_data, Y_tokenized_data, Y_float


# Encode the tokenized data into integers
def encoded_tokenized_data(X_tokenized_data, Y_float):
    
    X_encoded_data = []
    for row in X_tokenized_data:
        X_encoded_data.append([stoi[c] for c in row])
    return X_encoded_data, Y_float

# Function to generate all UIOs of a given length, but we do not use this because we 
# generate training data with Julia and C++, and we read in from the file

from eschers import UIO
from eschers import generate_all_uios

def getUIOTrainingVectors(UIO_length):
    coeffs = []
    Xuio = torch.tensor(generate_all_uios(UIO_length))
    for encod in Xuio:
        uio = UIO(encod) 
        coeffs.append(uio.getCoeffientByEscher(3,2,1)) 
    Yuio = torch.tensor(coeffs).unsqueeze(1)
    print(Xuio.shape, Yuio.shape)
    print(Xuio, Yuio)
    return Xuio, Yuio