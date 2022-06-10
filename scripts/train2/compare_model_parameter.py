from models import *


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    


model_dope = DopeNetwork()

model_dope_s3 = DopeNetworkStage3()
model_MobileNet = DopeMobileNet()
model_Resnet50 = DopeResNet50()
model_EfficientNet_B0 = DopeEfficientNet_B0()

count_parameters(model_dope)
count_parameters(model_dope_s3)
count_parameters(model_MobileNet)
# print(model_dope_s3)

# print (f"Dope: {count_parameters(model_dope)}")
# print (f"Dope_s3: {count_parameters(model_dope_s3)}")
# print (f"DopeMobileNetV2: {count_parameters(model_MobileNet)}")
# print (f"DopeResnet50: {count_parameters(model_Resnet50)}")
# print (f"DopeEfficientNet_B0: {count_parameters(model_EfficientNet_B0 )}")