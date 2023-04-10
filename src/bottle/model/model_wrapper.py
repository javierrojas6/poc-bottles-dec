
import torch
from datetime import datetime
from PIL import Image
import torchvision.transforms as T


class ModelWrapper():
    model = None
    learning_rate = 1e-3
    weight_decay = 0
    epochs = 10

    def __init__(self, model):
        self.model = model

    def _get_criterion_optimizer(self, lr=1e-3, weight_decay=0):
        """
        This function returns a cross-entropy loss criterion and an Adam optimizer with specified learning
        rate and weight decay for a given PyTorch model.

        :param lr: Learning rate, which determines the step size at each iteration while moving toward a
        minimum of a loss function during training
        :param weight_decay: Weight decay is a regularization technique used to prevent overfitting in
        neural networks. It adds a penalty term to the loss function that encourages the weights to be
        small. This penalty term is proportional to the square of the magnitude of the weights. The weight
        decay parameter controls the strength of this penalty term, defaults to 0 (optional)
        :return: The function `get_criterion_optimizer` returns a tuple containing the `criterion` and
        `optimizer` objects. The `criterion` is an instance of the `torch.nn.CrossEntropyLoss` class, and
        the `optimizer` is an instance of the `torch.optim.Adam` class, initialized with the learning rate
        `lr` and weight decay `weight_decay` parameters, and the parameters of
        """
        self.learning_rate = lr
        self.weight_decay = weight_decay

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        return criterion, optimizer

    def train(self, train_dataset, device, lr=1e-3, weight_decay=0, epochs=10, debug_mode=False):
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
        criterion, optimizer = self._get_criterion_optimizer(lr=lr, weight_decay=weight_decay)
        train_loss_data = []

        if debug_mode:
            print('starting training, this will take a moment...')

        for epoch in range(epochs):
            train_iterable = iter(train_dataset)
            # Load in the data in batches using the train_dataset object
            for i, (images, labels) in enumerate(train_iterable):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_data += [loss.item()]

            if debug_mode:
                print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, epochs, loss.item()))

        return train_loss_data

    def evaluate(self, test_dataset, device, debug_mode=False):
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
            all_true_labels = []
            all_predicted_labels = []

            for images, labels in test_iterable:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)

                all_true_labels += list(labels.to('cpu').numpy())
                all_predicted_labels += list(predicted.to('cpu').numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if debug_mode:
                print('Accuracy of the network on the {} train images: {} %'.format(total, 100 * correct / total))

        return all_true_labels, all_predicted_labels

    def predict(self, x, transforms = []):
        """
        This function takes an image file path, applies transforms (if provided), and returns the predicted
        class label using a PyTorch model.
        
        :param x: The file path of the image to be predicted
        :param transforms: `transforms` is a list of image transformations that can be applied to the input
        image before it is passed through the model for prediction. If no transforms are provided, the
        default transform applied is `T.ToTensor()`, which converts the image to a PyTorch tensor. Other
        common transforms include resizing
        :return: The function `predict` takes an image file path `x` and a list of image transforms
        `transforms` as input, applies the transforms to the image, and returns a list of predicted class
        labels for the image using a pre-trained PyTorch model. Specifically, it returns a list containing a
        single element, which is the predicted class label for the input image.
        """
        img = Image.open(x)

        if len(transforms) == 0: # no transforms
            transforms = T.Compose([T.ToTensor()])

        img = transforms(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output.data, 1)
            output = list(predicted.numpy())

        return output

    def measure_accuracy(self, classes, dataset):
        """
        This function measures the accuracy of a model's predictions for each class in a given dataset.

        :param classes: A list of class names for the dataset. For example, if the dataset contains
        images of cats and dogs, the classes list would be ['cat', 'dog']
        :param dataset: The dataset parameter is a dataset object that contains the images and labels
        for the dataset that we want to measure the accuracy of
        """
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in dataset:
                images, labels = data
                outputs = self.model(images)
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

    def save(self, file_name='model', device='cpu', ext='pth'):
        """
        This function saves the state dictionary of a PyTorch model to a file with a generated name based
        on the current date and time.

        :param file_name: The name of the file to be saved. If no name is provided, the default name will
        be 'model', defaults to model (optional)
        """
        date_number = datetime.now().strftime('%Y%m%d%H%M%S')

        generated_name = '.'.join([f'{file_name}-{date_number}', device, ext])

        torch.save(self.model.state_dict(), generated_name)

    def load(self, path):
        """
        This function loads a saved state dictionary of a PyTorch model from a specified path.

        :param path: The path parameter is a string that represents the file path where the model's
        state dictionary is saved. The load method loads the state dictionary from the specified file
        path and updates the model's parameters with the loaded values
        """
        self.model.load_state_dict(torch.load(path))
