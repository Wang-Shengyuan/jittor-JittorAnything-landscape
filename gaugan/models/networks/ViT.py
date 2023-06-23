import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer

class ViTEncoder(BaseNetwork):
    
    def __init__(self, opt):
        super().__init__()
        

    def execute(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = nn.interpolate(x, size=(256, 256), mode='bilinear')


