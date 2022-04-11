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
import torch.utils.bundled_inputs
import torch.utils.mobile_optimizer
import torch.backends._nnapi.prepare
import torchvision.models.quantization.mobilenet


def make_mobilenetv2_nnapi(output_dir_path, quantize_mode):
    quantize_core, quantize_iface = {
        "none": (False, False),
        "core": (True, False),
        "full": (True, True),
    }[quantize_mode]

    # model = torchvision.models.quantization.mobilenet.mobilenet_v2(pretrained=True, quantize=quantize_core)
    model = DopeMobileNet()
    model.load_state_dict(torch.load("/home/airobot/dopeData/weights/redtea_mobile88.pth",map_location=device), strict=False)
    model.eval()

    # Fuse BatchNorm operators in the floating point model.
    # (Quantized models already have this done.)
    # Remove dropout for this inference-only use case.

    # if not quantize_core:
    #     model.fuse_model()
    # assert type(model.classifier[0]) == torch.nn.Dropout
    # model.classifier[0] = torch.nn.Identity()

    input_float = torch.zeros(1, 3, 400, 400)
    input_tensor = input_float

    # If we're doing a quantized model, we need to trace only the quantized core.
    # So capture the quantizer and dequantizer, use them to prepare the input,
    # and replace them with identity modules so we can trace without them.
    if quantize_core:
        quantizer = model.quant
        dequantizer = model.dequant
        model.quant = torch.nn.Identity()
        model.dequant = torch.nn.Identity()
        input_tensor = quantizer(input_float)

    # Many NNAPI backends prefer NHWC tensors, so convert our input to channels_last,
    # and set the "nnapi_nhwc" attribute for the converter.
    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    input_tensor.nnapi_nhwc = True

    # Trace the model.  NNAPI conversion only works with TorchScript models,
    # and traced models are more likely to convert successfully than scripted.
    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensor)
    nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)

    # If we're not using a quantized interface, wrap a quant/dequant around the core.
    if quantize_core and not quantize_iface:
        nnapi_model = torch.nn.Sequential(quantizer, nnapi_model, dequantizer)
        model.quant = quantizer
        model.dequant = dequantizer
        # Switch back to float input for benchmarking.
        input_tensor = input_float.contiguous(memory_format=torch.channels_last)

    # Optimize the CPU model to make CPU-vs-NNAPI benchmarks fair.
    model = torch.utils.mobile_optimizer.optimize_for_mobile(torch.jit.script(model))

    # Bundle sample inputs with the models for easier benchmarking.
    # This step is optional.
    class BundleWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod
        def forward(self, arg):
            return self.mod(arg)
    nnapi_model = torch.jit.script(BundleWrapper(nnapi_model))
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        nnapi_model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])

    # Save both models.
    model._save_for_lite_interpreter(str(output_dir_path / ("mobilenetv2-quant_{}-cpu.pt".format(quantize_mode))))
    nnapi_model._save_for_lite_interpreter(str(output_dir_path / ("mobilenetv2-quant_{}-nnapi.pt".format(quantize_mode))))



model = DopeMobileNet()
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

make_mobilenetv2_nnapi(Path(os.environ["HOME"]) / "mobilenetv2-nnapi", "none")
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