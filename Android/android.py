import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

import sys
# Import the  module
from pathlib import Path
from os import path
#sys.path.insert(0, path.join(path.dirname(__file__), "../../"))
sys.path.append(str(Path(path.dirname(__file__)).parent))
from scripts.train2.models import *


model = DopeMobileNet()
model.load_state_dict(torch.load("/home/airobot/dopeData/weights/redtea_mobile88.pth"))
model.eval()
example = torch.rand(1, 3, 400, 400)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
# optimized_traced_model._save_for_lite_interpreter("app/src/main/assets/model.ptl")
optimized_traced_model._save_for_lite_interpreter("model.ptl")