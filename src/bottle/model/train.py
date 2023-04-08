
import torch

def get_criterion_optimizer(model, lr=1e-3, weight_decay=0):
    """
    This function returns a criterion and optimizer for a PyTorch model using cross-entropy loss and
    Adam optimizer with specified learning rate and weight decay.
    
    :param model: The neural network model that we want to train
    :param lr: lr stands for learning rate, which is a hyperparameter that controls the step size at
    each iteration while moving toward a minimum of a loss function during training of a neural network.
    It determines how much the model weights are updated in response to the estimated error each time
    the model is updated
    :param weight_decay: Weight decay is a regularization technique used to prevent overfitting in
    neural networks. It adds a penalty term to the loss function during training, which encourages the
    model to have smaller weights. This helps to prevent the model from becoming too complex and
    overfitting the training data. The weight decay parameter controls, defaults to 0 (optional)
    :return: The function `get_criterion_optimizer` returns a tuple containing the `criterion` and
    `optimizer` objects. The `criterion` is an instance of the `torch.nn.CrossEntropyLoss` class, and
    the `optimizer` is an instance of the `torch.optim.Adam` class, initialized with the model's
    parameters, a learning rate `lr`, and a weight decay `weight_decay`.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return criterion, optimizer

def train_model(model, train_dataset, device, lr=1e-3, weight_decay=0, epochs=10, debug_mode=False):
    """
    This function trains a given model on a given dataset for a specified number of epochs using a
    specified device, learning rate, weight decay, and criterion optimizer.
    
    :param model: The neural network model that we want to train
    :param train_dataset: The dataset object containing the training data
    :param device: The device on which the model and data will be loaded, such as 'cpu' or 'cuda'
    :param lr: learning rate, which controls the step size at each iteration while moving toward a
    minimum of a loss function
    :param weight_decay: Weight decay is a regularization technique used to prevent overfitting in
    machine learning models. It adds a penalty term to the loss function that encourages the model to
    have smaller weights. The weight decay parameter controls the strength of this penalty term. A
    higher weight decay value will result in a stronger regularization effect, defaults to 0 (optional)
    :param epochs: The number of times the entire training dataset will be passed through the model
    during training, defaults to 10 (optional)
    :param debug_mode: debug_mode is a boolean parameter that determines whether to print the loss after
    each epoch during training. If set to True, the loss will be printed; if set to False, the loss will
    not be printed, defaults to False (optional)
    :return: a list of training loss data.
    """
    criterion, optimizer = get_criterion_optimizer(model, lr, weight_decay)
    train_loss_data = []

    for epoch in range(epochs):
        train_iterable = iter(train_dataset)
        #Load in the data in batches using the train_dataset object
        for i, (images, labels) in enumerate(train_iterable):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if debug_mode:
            print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, epochs, loss.item()))

    train_loss_data += [loss.item()]
    
    return train_loss_data

def test_model(model, test_dataset, device, debug_mode=False):
    """
    The function takes a model, test dataset, and device as inputs, and evaluates the accuracy of the
    model on the test dataset.
    
    :param model: The neural network model that has been trained and is being tested on the test dataset
    :param test_dataset: The test_dataset parameter is a dataset object containing the test data, which
    is used to evaluate the performance of the model. It typically contains a set of images and their
    corresponding labels
    :param device: The device parameter specifies whether the computations should be performed on the
    CPU or GPU. It is used to move the data and model to the specified device before performing any
    computations
    :param debug_mode: debug_mode is a boolean parameter that is used to control whether or not to print
    the accuracy of the model during testing. If debug_mode is set to True, the function will print the
    accuracy of the model on the test dataset. If debug_mode is set to False, the function will not
    print anything, defaults to False (optional)
    """
    test_iterable = iter(test_dataset)

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_iterable:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if debug_mode:
            print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))
            
def measure_accuracy(model, classes, dataset):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        