import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

import sys
# Import the  module
from pathlib import Path
# from os import path
import os 
#sys.path.insert(0, path.join(path.dirname(__file__), "../../"))
sys.path.append(str(Path(os.getcwd()).parent))
from scripts.train2.models import *


# model = DopeMobileNet()
# model = torch.nn.DataParallel(model)
# # cudnn.benchmark = True
device = torch.device('cpu')
model.load_state_dict(torch.load("/home/airobot/dopeData/weights/redtea_mobile88.pth",map_location=device), strict=False)
# model.load_state_dict(torch.load("/home/airobot/dopeData/weights/redtea_mobile88.pth"), strict=False)
# model = torch.nn.Module.load_state_dict(torch.load("/home/airobot/dopeData/weights/redtea_mobile88.pth"), strict=False)
# model.load_state_dict(torch.load("/home/airobot/dopeData/weights/net_Milktea_mobile_final.pth"), strict=False)
model.eval()

example = torch.rand(1, 3, 400, 400)
out = model(example)
print(len(out))

# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("model_mobile_full400.pt")


# print(traced_script_module.code)
# print(traced_script_module.graph)
# # traced_script_module .save('model_scripted.pt') 
# optimized_traced_model = optimize_for_mobile(traced_script_module)
# # # optimized_traced_model._save_for_lite_interpreter("app/src/main/assets/model.ptl")
# optimized_traced_model._save_for_lite_interpreter("model_mobile_cpu.pt")


#sample 
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
# model.eval()
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# optimized_traced_model = optimize_for_mobile(traced_script_module)
# optimized_traced_model._save_for_lite_interpreter("model.pt")