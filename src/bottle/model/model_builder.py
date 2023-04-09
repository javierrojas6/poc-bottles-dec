import torch
import torchvision
import sklearn

def build_bottle_detection_model(device):
    """
    This function builds and returns a pre-trained Mask R-CNN ResNet50 FPN model for bottle detection,
    which is then moved to the specified device.
    
    :param device: The "device" parameter is used to specify whether the model should be trained and run
    on the CPU or GPU. It is a string that can take the values "cpu" or "cuda" depending on the hardware
    available
    :return: a pre-trained Mask R-CNN model based on ResNet-50 architecture for detecting objects and
    generating segmentation masks. The model is being moved to the specified device (CPU or GPU) before
    returning.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    return model.to(device)


def build_bottle_cap_state_detection_model(device, pretrained=True):
    """
    This function builds a bottle cap state detection model using a pre-trained ResNet18 model from
    PyTorch's vision library.
    
    :param device: The device parameter specifies the device on which the model will be loaded and run.
    It can be either "cpu" or "cuda" depending on whether you want to use the CPU or GPU for computation
    :param pretrained: pretrained is a boolean parameter that specifies whether to load the pre-trained
    weights for the ResNet18 model or not. If set to True, the pre-trained weights will be loaded, and
    if set to False, the model will be initialized with random weights, defaults to True (optional)
    :return: a ResNet18 model for detecting the state of bottle caps, which is loaded from the PyTorch
    vision library. The model is also moved to the specified device (CPU or GPU) before being returned.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=pretrained)
    return model.to(device)

def evaluate_classification(y_true, y_pred, labels=None):
    """
    The function `evaluate_classification` calculates various classification metrics for a given set of
    true and predicted labels.
    
    :param y_true: The true labels of the classification problem
    :param y_pred: The predicted labels or classes for a classification problem
    :param labels: The list of labels to include in the report. If None, all labels are included
    :return: a dictionary containing various evaluation metrics for a classification problem, such as
    confusion matrix, accuracy score, AUC, average precision score, balanced accuracy score, F1 score,
    F-beta score, Matthews correlation coefficient, precision-recall curve, precision score, recall
    score, and top-k accuracy score.
    """
    average = 'micro'

    return {
        'confusion_matrix': sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels),
        'accuracy_score': sklearn.metrics.accuracy_score(y_true, y_pred),
        'auc': sklearn.metrics.accuracy_score(y_true, y_pred),
        'average_precision_score': sklearn.metrics.average_precision_score(y_true, y_pred),
        'balanced_accuracy_score': sklearn.metrics.balanced_accuracy_score(y_true, y_pred),
        # 'brier_score_loss': sklearn.metrics.brier_score_loss(y_true, y_pred),
        'f1_score': sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average=average),
        'fbeta_score': sklearn.metrics.fbeta_score(y_true, y_pred, beta=0.5, labels=labels, average=average),
        'matthews_corrcoef': sklearn.metrics.matthews_corrcoef(y_true, y_pred),
        'precision_recall_curve': sklearn.metrics.precision_recall_curve(y_true, y_pred),
        'precision_score': sklearn.metrics.precision_score(y_true, y_pred, labels=labels, average=average),
        'recall_score': sklearn.metrics.recall_score(y_true, y_pred, labels=labels, average=average),
        'top_k_accuracy_score': sklearn.metrics.top_k_accuracy_score(y_true, y_pred),
    }
