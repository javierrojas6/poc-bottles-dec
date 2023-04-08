import torch
import torchvision

def build_bottle_detection_model(device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    return model.to(device)
    
def build_bottle_cap_state_detection_model(device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    return model.to(device)
