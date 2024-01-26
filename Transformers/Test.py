import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_vectors(n_vectors, length, start, end):
    vectors = []
    for _ in range(n_vectors):
        vector = torch.sort(torch.randint(start, end+1, (length,)))[0]
        vectors.append(vector)
    return torch.stack(vectors)


UIOs = generate_vectors(100, 11, 1, 10)
n = int(0.9*len(UIOs)) # first 90% will be train, rest val
UIOtrain_data = UIOs[:n]
UIOval_data = UIOs[n:]
print(UIOtrain_data)

# data loading
batch_size = 10
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = UIOtrain_data if split == 'train' else UIOval_data
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i][:-1] for i in ix])
    y = torch.stack([data[i][-1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

print(get_batch('train'))

true_values = torch.tensor([2, 4, 5, 4, 5], dtype=torch.float32)
predicted_values = torch.tensor([1.5, 3.5, 4.8, 4.2, 5.2], dtype=torch.float32)

# Create the MSE loss criterion
mse_loss = nn.MSELoss()

# Calculate the MSE loss
loss_value = mse_loss(predicted_values, true_values)

print("Mean Squared Error (MSE) Loss:", loss_value.item())
