from models import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_dope = DopeNetwork()
model_MobileNet = DopeMobileNet()
model_Resnet50 = DopeResNet50()
model_EfficientNet_B0 = DopeEfficientNet_B0()

print (f"Dope: {count_parameters(model_dope)}")
print (f"DopeMobileNetV2: {count_parameters(model_MobileNet)}")
print (f"DopeResnet50: {count_parameters(model_Resnet50)}")
print (f"DopeEfficientNet_B0: {count_parameters(model_EfficientNet_B0 )}")