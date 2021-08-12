from networks import *
import os, sys, math, argparse
gpu_id = 1
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
model = define_G(1, 1, 64, 'resnet_9blocks').to(device)
batch_size = 1
input_c = 1
width, height = 1024, 1024
rand_tensor = torch.rand((batch_size, input_c, height, width)).to(device)
output = model(rand_tensor)
print(output.size())