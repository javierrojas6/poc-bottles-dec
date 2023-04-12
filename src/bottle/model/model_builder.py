import torch
import torchvision
import sklearn

BOTTLE_DETECTION_MODEL = ('resnet50', 'v0.15.1')
# BOTTLE_CAP_STATE_DETECTION_MODEL = ('resnet34', 'v0.15.1', 'ResNet34_Weights.DEFAULT')
BOTTLE_CAP_STATE_DETECTION_MODEL = ('resnet18', 'v0.10.0', 'ResNet18_Weights.DEFAULT')


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
    This function builds and loads a pre-trained PyTorch model for detecting the state of bottle caps on
    a specified device.
    
    :param device: The device parameter specifies the device on which the model will be loaded and run.
    It can be either "cpu" or "cuda" depending on whether you want to use the CPU or GPU for computation
    :param pretrained: A boolean value indicating whether to load the pre-trained weights for the model
    or not. If set to True, the function will load the pre-trained weights for the model. If set to
    False, the function will load the model without any pre-trained weights, defaults to True (optional)
    :return: The function `build_bottle_cap_state_detection_model` returns a PyTorch model for detecting
    the state of a bottle cap, loaded from a pre-trained model if `pretrained=True`. The model is loaded
    onto the specified `device`.
    """
    name, version, weights = BOTTLE_CAP_STATE_DETECTION_MODEL
    if pretrained: model = torch.hub.load(f'pytorch/vision:{version}', name, weights=weights)
    else: model = torch.hub.load(f'pytorch/vision:{version}', name)
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
