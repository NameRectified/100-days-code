
## Day 23

### Introduction to deep learning with pytorch

- The fundamental model structure in deep learning is a network of inputs, hidden layers and outputs.
- They  require far more data compared to other machine learning models to derive patterns
- Tensor is similar to an array or matrix and is the building block of neural networks
- Two tensors are compatible(for addn or subtracn) if their shapes or dimesions are equal.
- element wise multiplication `*` and matrix multiplicaton `@`

```python
import torch

temperatures = [[72, 75, 78], [70, 73, 76]]

# Create a tensor from temperatures
temp_tensor = torch.tensor(temperatures)

print(temp_tensor) 

adjustment = torch.tensor([[2, 2, 2], [2, 2, 2]])

# Display the shape of the adjustment tensor
print("Adjustment shape:", adjustment.shape)

# Display the type of the adjustment tensor
print("Adjustment type:", adjustment.dtype)

print("Temperatures shape:", temperatures.shape)
print("Temperatures type:", temperatures.dtype)

adjustment = torch.tensor([[2, 2, 2], [2, 2, 2]])

# Add the temperatures and adjustment tensors
corrected_temperatures = adjustment + temperatures
print("Corrected temperatures:", corrected_temperatures)
```

### first neural network

- It is going to be a fully connected neural network with no hiddenn layers(equivalent to a linear model)
- When designing a neural network, the input and output dimesions are predefined, number of neurons in the innput layer is the number of features in our dataset and number of neurons in the output layer is number of classes we want to predict
- each linear layer has a set of `weights` (.weights) and `biases` (.biases)
- weight reflects the importance of each feature and bias provides the neurons with a baseline output

```python
import torch
import torch.nn as nn

input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])

# Create a Linear layer
linear_layer = nn.Linear(
                         in_features=3, 
                         out_features=2
                         )

# Pass input_tensor through the linear layer
output = linear_layer(input_tensor)

print(output)
```

- we can stack layers using `nn.Sequential`  and layers within it are hidden layers
- We can add as many hidden layers as we want as long as the input dimesion of a layer is equal to the output dimension of the previous layer
- a layer is fully connected when each neuron links to all neurons in the previous layer
- each neuron performs a linear operation using all neurons from the previous layer and hence a single neuron has N+1 learnable parameters, N outputs from previous layer and 1 from bias
- Increasing the hiddenn layers increases the number of parameters in the model also known as model capacity. Higher capacity models can handle more complex datasets but may take lonnger to train and risk overfitting, while too few might limit learninng capacity
- to calculate the model capacity or total number of parameters: (no. of neurons * inputs or outputs from previous layer) + no. of neuron(as each neuron has 1 bias) = no.of nuerons * (inputs/outputs +1)
- we can also calculate the number of elements in tensor using .numel()

```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(9, 4),
                      nn.Linear(4, 2),
                      nn.Linear(2, 1))

total = 0

# Calculate the number of parameters in the model
for p in model.parameters():
  total += p.numel()
  
print(f"The number of parameters in the model is {total}")
```

- multilayer neural network

```python
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[2, 3, 6, 7, 9, 3, 2, 1]])

# Create a container for stacking linear layers
model = nn.Sequential(nn.Linear(8, 4),
                nn.Linear(4, 1)
                )

output = model(input_tensor)
print(output)
```

### Activation functions

- they help us add non linearity to. the network,  sigmoid(nn.Sigmoid) for binary classification and softmax(nn.Softmax) for multiclass classification
- non linearity helps the network learn more complex interactions and relationships between inputs and targets
- The output of the last liear layer is called `pre-activation`  output, which will be passed to activation functions to get the transformed output
- A neural network with linear layers and one sigmoid layer as the last step behaves like logistic regression (A logistic regression model is essentially a single-layer neural network with a sigmoid activation.)
- `dim=-1` in softmax indicates the last dimension as the last linear layer’s output
- Create a sigmoid function and apply it on `input_tensor` to generate a probability for a binary classification task.

```python
input_tensor = torch.tensor([[2.4]])

# Create a sigmoid function and apply it on input_tensor
sigmoid = nn.Sigmoid()
probability = sigmoid(input_tensor)
print(probability)
```

- Create a softmax function and apply it on `input_tensor` to generate a probability for a multi-class classification task.

```python
input_tensor = torch.tensor([[1.0, -6.0, 2.5, -0.3, 1.2, 0.8]])

# Create a softmax function and apply it on input_tensor
softmax = nn.Softmax()
probabilities = softmax(input_tensor)
print(probabilities)
```

### forward pass

- When the input data flows throught a neural network in the forward direction to produce predictions or outputs, calculations are performed at each layer, and are passed to the next layer until the final output is generated
- Q) create a neural network that takes a **1x8** tensor as input and outputs a single value for binary classification.
    - Pass the output of the linear layer to a **sigmoid** to produce a probability.

```python
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# Implement a small neural network for binary classification
model = nn.Sequential(
  nn.Linear(8,1),
  nn.Sigmoid()
)

output = model(input_tensor)
print(output)
```

- The sigmoid output is always between 0 and 1.
- **multi-class** classification with **four** outputs

```python
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Update network below to perform a multi-class classification with four labels
model = nn.Sequential(
  nn.Linear(11, 20),
  nn.Linear(20, 12),
  nn.Linear(12, 6),
  nn.Linear(6, 4),
  nn.Softmax(dim=-1) # for reggresion just remove this line and 
  #change output for previous layer to 1
)

output = model(input_tensor)
print(output)
```

### loss functions

- Loss functions tell us how good our model is at making predictions during training
- we use one hot encoding to convert an integer into a tensor of zeros and ones
- cross entropy loss is used for classification
- loss function takes:
    - scores - model predictions before the final softmax function
    - one hot target - one hot encoded ground truth label
- loss function outputs:
    - loss -  a single float
- Creating one-hot encoded labels

```python
import torch.nn.functional as F
y = 1
num_classes = 3

# Create the one-hot encoded vector using NumPy
one_hot_numpy = np.array([0, 1, 0])

# Create the one-hot encoded vector using PyTorch
one_hot_pytorch = F.one_hot(torch.tensor(y), num_classes=num_classes)

print("One-hot vector using NumPy:", one_hot_numpy)
print("One-hot vector using PyTorch:", one_hot_pytorch)
```

- Calculating cross entropy loss

```python
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y
one_hot_label = F.one_hot(torch.tensor(y), num_classes=4)

# Create the cross entropy loss function
criterion = CrossEntropyLoss()

loss = criterion(scores.double(), one_hot_label.double())
print(loss)
```

### Using derivatives to update model parameters

- derivative represents the slope
- where the derivative is zero, that is the loss fuction’s minimum
- We compute loss by comparing the predictions to the target value
- layer weights and biases are randomly intialized when a model is created and update them during training using a backward pass or backpropogation
- derivatives(gradients) help minimize the loss and to tune layer weights and biases

### updating model parameters manually

- access each layer gradient
- multiply by learning rate
- subtract this product from the weight

- for non-convex functions we use gradient descent
- .parameters() returns an iterable of all model parameters
- **Accessing the model parameters**

```python
model = nn.Sequential(nn.Linear(16, 8),
                      nn.Linear(8, 2)
                     )

# Access the weight of the first linear layer
weight_0 = model[0].weight
print("Weight of the first layer:", weight_0)

# Access the bias of the second linear layer
bias_1 = model[1].bias
print("Bias of the second layer:", bias_1)

# Updating the weights manually
weight0 = model[0].weight
weight1 = model[1].weight
weight2 = model[2].weight

# Access the gradients of the weight of each linear layer
grads0 = weight0.grad
grads1 = weight1.grad
grads2 = weight2.grad

# Update the weights using the learning rate and the gradients
weight0 = weight0 - grads0 * lr
weight1 = weight1 - grads1 * lr
weight2 = weight2 - grads2 * lr

```

- Using the PyTorch optimizer

```python
# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss = criterion(pred, target)
loss.backward()

# Update the model's parameters using the optimizer
optimizer.step()
```