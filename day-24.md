## Day 24

- TensortDataset() allows us to store our features (X) and target labels (y) as tensors making them easy to manage
- DataLoader helps us efficiently manage data loading durig training

```
import torch
from torch.utils.data import TensorDataset

X = animals.iloc[:, 1:-1].to_numpy()  
y = animals.iloc[:, -1].to_numpy()

# Create a dataset
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# Print the first sample
input_sample, label_sample = dataset[0]
print('Input sample:', input_sample)
print('Label sample:', label_sample)
```

- **Using DataLoader:**

```python
from torch.utils.data import DataLoader

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size = 2, shuffle=True)

# Iterate over the dataloader
for batch_inputs, batch_labels in dataloader:
    print('batch_inputs:', batch_inputs)
    print('batch_labels:', batch_labels)
```

- Training a neural network:
    - create a model
    - choose a loss function
    - define a dataset
    - set an optimizer
    - run a training loop
        - calculate the loss(forward pass)
        - compute gradients(backprop)
        - updating model parameters
    - for regression we will use a linear layer as the final output istead of sigmoid or softmax
- Using the MSELoss
    
    ```python
    y_pred = np.array([3, 5.0, 2.5, 7.0])  
    y = np.array([3.0, 4.5, 2.0, 8.0])     
    
    # Calculate MSE using NumPy
    mse_numpy = np.mean((y_pred - y)**2)
    
    # Create the MSELoss function in PyTorch
    criterion = nn.MSELoss()
    
    # Calculate MSE using PyTorch
    mse_pytorch = criterion(torch.tensor(y_pred),torch.tensor(y))
    
    print("MSE (NumPy):", mse_numpy)
    print("MSE (PyTorch):", mse_pytorch)
    ```
    
- Writing a training loop: The following variables have been created: `num_epochs`, containing the number of epochs (set to 5); `dataloader`, containing the dataloader; `model`, containing the neural network; `criterion`, containing the loss function, `nn.MSELoss()`; `optimizer`, containing the SGD optimizer.

```python
# Loop over the number of epochs and then the dataloader
for i in range(num_epochs):
  for data in dataloader:
    # Set the gradients to zero
    optimizer.zero_grad()
    # Run a forward pass
    feature, target = data
    prediction = model(feature)
    # Compute the loss
    loss = criterion(prediction, target)
    # Compute the gradients
    loss.backward()
    # Update the model's parameters
    optimizer.step()
```

### disadvantages of few activation functions

- some activation functions can shrink gradients too much and make the training inefficient
- limitations of sigmoid and softmax:
    - the output is bounded between 0 and 1
    - it is very small for large and small values of x
    - it causes saturation leading to vanishing gradient
    - durig backprop this becomes a problem as each gradient depends on the previous one
    - when gradients are extremely small, they fail to update the weights effectively
    - hence both these fuctions are nnot ideal for hidden layers and are best used for the last layer only

### ReLU activation

- f(x) = max(0,x)
- A good rule of thumb is to use ReLU as the default activation function in your models (except for the last layer).

```python
# Create a ReLU function with PyTorch
relu_pytorch = nn.ReLU()

x_pos = torch.tensor(2.0)
x_neg = torch.tensor(-3.0)

# Apply the ReLU function to the tensors
output_pos = relu_pytorch(x_pos)
output_neg = relu_pytorch(x_neg)

print("ReLU applied to positive value:", output_pos)
print("ReLU applied to negative value:", output_neg
```

### Leaky ReLU

- for positive inputs it behaves like ReLU
- for negative inputs, it multiplies the input by a small coefficient (0.01 is the default in pytorch)
- this ensures gradients for negative inputs is non zero, preventing neurons from completely stopping learning , which can happen with ReLU
- Leaky ReLU is another very popular activation function found in modern architecture. By never setting the gradients to zero, it allows every parameter of the model to keep learning.

```python
# Create a leaky relu function in PyTorch
leaky_relu_pytorch = nn.LeakyReLU(negative_slope = 0.05)

x = torch.tensor(-2.0)
# Call the above function on the tensor x
output = leaky_relu_pytorch(x)
print(output)
```

### Updating weights with SGD

- training a neural network means solving an optimization problem by minimizing the loss function and adjusting the model parameters
- sgd takes two arguments:
    - learning rate: controls the step size, if its too low the traininng is slow
    - momentum: adds inertia to help optimizer moves smoothly and avoid getting stuck at `local minima`, if its too small, optimizer get stuck
- step size decreases near zero as the gradient gets smaller as step size is the gradient multiplied by the learning rate
- since the functionn is less steep near zero, the gradient and thus the step size, gets smaller
- Momentum and learning rate are critical to the training of your neural network. A good rule of thumb is to start with a learning rate of 0.001 and a momentum of 0.95.