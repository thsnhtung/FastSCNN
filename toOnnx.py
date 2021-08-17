from models.fast_scnn import get_fast_scnn
import torch
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

model = get_fast_scnn(dataset= 'simulation', aux= False)
model.eval()
x = torch.randn(1, 3, 160, 320, requires_grad=True)

torch_out = model(x)
print(torch_out[0].size())

torch.onnx.export(model, x, 'out_model.onnx', export_params=True, opset_version=12, do_constant_folding=True, 
                   input_names = ['input'], output_names = ['output'])
                    