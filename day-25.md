## Day 25

### Layer initialization

- A layer’s weight are initialized to small values
- in torch we can initialize weights using `nn.init`

```python
layer0 = nn.Linear(16, 32)
layer1 = nn.Linear(32, 64)

# Use uniform initialization for layer0 and layer1 weights
nn.init.uniform_(layer0.weight)
nn.init.uniform_(layer1.weight)

model = nn.Sequential(layer0, layer1)
```

### Transfer learning

- Reusing a model trained on a first task for another similar second task
- `fine-tuning` a type of transfer learning in which we load the weights from a previous model, but train the model with smaller learning rate
    - train part of the network(we freeze some of the layers)
    - rule of thumb: freeze early layers of network and fine-tune layers closers to the output layer, this can be achieved by assigning parameters `requires_grad = False`
    - **Freeze layers of a model**
    
    ```python
    for name, param in model.named_parameters():
      
        # Check for first layer's weight
        if name == '0.weight':
       
            # Freeze this weight
            param.requires_grad = False
            
        # Check for second layer's weight
        if name == '1.weight':
          
            # Freeze this weight
            param.requires_grad = False
    ```
    

### Evaluating model performance

- training data: adjusts model parameters like weights and biases
- validation data: tunes hyperparameters like momentum, learning rate
- test: evaluates final model performance
- we will track loss and accuracy during training and validation
- calculating training loss:
    - for each epoch:
        - sum the loss across all the batches of the dataloader
        - compute the mean training loss at the end of the epoch
- when a model overfits, training loss keeps decreasing but validation loss starts to increase
- calculate accuracy with torchmetrics

### Writing the evaluation loop

```python
# Set the model to evaluation mode
model.eval()
validation_loss = 0.0

with torch.no_grad(): # in validation we dont wish to update gradient and hence no grad
  
  for features, labels in validationloader:
    
      outputs = model(features)
      loss = criterion(outputs, labels)
      
      # Sum the current loss to the validation_loss variable
      # make sure to apply .item() to turn the loss tensor into a numerical value
      validation_loss += loss.item()
# Calculate the mean loss value
validation_loss_epoch = validation_loss / len(validationloader)
print(validation_loss_epoch)

# Set the model back to training mode
model.train()
```

### Calculating accuracy using torchmetrics

```python
# Create accuracy metric
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
for features, labels in dataloader:
    outputs = model(features)
  
    # Calculate accuracy over the batch
    # argmax is used to convert the one hot encoded prediction into 
    # class indices before passing them to the metric
    metric.update(outputs, labels.argmax(dim=-1))
    
# Calculate accuracy over the whole epoch
accuracy = metric.compute()
print(f"Accuracy on all data: {accuracy}")

# Reset metric for the next epoch
metric.reset()
plot_errors(model, dataloader) #The plot_errors function will highlight misclassified samples, helping you analyze model errors.
```

### Fighting overfitting

- overfitting: model does not generalize to unseen data
    - model memorizes training data
    - performs well on training data but poorly on validation data
- causes (problem and solutionn):
    - dataset is not large eough: get more data/ use data augmentation
    - model has too much capacity: reduce model size/ add dropout
    - weights are too large: then decay the weight
- strategies:
    - reducing model size or adding dropout layer
    - using weight decay to force parameters to remain small
    - obtaining new data or augmenting data
- regularisation using a dropout layer:
    - randomly zeroes out elements of the input tensor during training
    - they are generally added after the activation function
    - it behaves differently during trainnig and evaluation, durign trainnig it randomly deactivates neurons while during evaluation it is disabled ensuring all neurons are active for stable predictions
- regularizationn with weight decay:
    - it is added to the optimizer with the `weight_decay` parameter that is set to a small value eg 0.0001
    - weight decay add penalty to the loss fuction and encourages smaller weights
    - during backpropogation, this penalty is subtracted from the gradient, preventing excessive weight growth
    - the higher the weight decay, the stronger the realization, making overfitting less likely
- data augmentation:
    - it is commonly applied to image data, which ca be rotated and scaled, so that different views of the same face become available as “new points”
- Experimenting with dropout: Dropout is a powerful tool against overfitting. Real-world models use hundreds of these layers to boost performance on unseen data.

```python
# Model with Dropout
model = nn.Sequential(
    nn.Linear(8, 6),
    nn.Linear(6, 4),
    nn.Dropout(p=0.5))

# Forward pass in training mode (Dropout active)
model.train()
output_train = model(features)

# Forward pass in evaluation mode (Dropout disabled)
model.eval()
output_eval = model(features)

# Print results
print("Output in train mode:", output_train)
print("Output in eval mode:", output_eval)
```

### Improving model performance

- steps to maximize performance:
    1. overfit the training dataset: this allows us to know if the problem is solvable
        1. modify the trainig loop to overfit a single datapoint. it is useful to start with a single datapoint before overfitting the entire dataset
        2. we modify the training loop to repeatedly train on a single example rather than iterating over the entire dataloader
        3. it should quickly reach near-zero loss and 100% accuracy on that datapoint
        4. then we scale up to the entire training set
    2. reduce overfitting: set a performance baseline
        1. use the strategies discussed above to reduce overfitting
        2. keep track of each hyperparameter and validation accuracy for each set of experimets
    3. fine tune the hyperparameters:
        1. gridsearch tests parameters at fixed intervals
        2. randomsearch (more efficient) selects them within a given range
    
    Implementing random search
    
    ```python
    values = []
    for idx in range(10):
        # Randomly sample a learning rate factor between 2 and 4
        factor = np.random.uniform(2,4)
        lr = 10 ** -factor
        
        # Randomly select a momentum between 0.85 and 0.99
        momentum = np.random.uniform(0.85, 0.99)
        
        values.append((lr, momentum))
           
    plot_hyperparameter_search(values) #
    ```
    