from ringnet.models import Base3DObjectRingsExtractor
import torch
from torch.utils.data import DataLoader

from common.models import MLP, EfficientNetExtractor, ResNetExtractor
import numpy as np
import os
import json
import argparse


parser = argparse.ArgumentParser()
# Directory 
parser.add_argument('--obj-path', type=str, required=True, help='Path to 3D object set')
parser.add_argument('--sketch-path', type=str, required=True, help='Path to sketch test set')
parser.add_argument('--output-path', type=str,
                    default='./predict', help='Path to output folder')

#Tuning weight
parser.add_argument('--view-cnn-backbone', type=str, default='efficientnet_b2',
                    choices=[*ResNetExtractor.arch.keys(), *EfficientNetExtractor.arch.keys()],  help='Model for ringview feature extraction')
parser.add_argument('--skt-cnn-backbone', type=str, default='efficientnet_b2',
                    choices=[*ResNetExtractor.arch.keys(), *EfficientNetExtractor.arch.keys()],  help='Model for sketch feature extraction')

parser.add_argument('--latent-dim', type=int, default=128,
                    help='Latent dimensions of common embedding space')

parser.add_argument('--obj-weight', type=str, required=True, help='Path to 3D object weight')
parser.add_argument('--skt-weight', type=str, required=True, help='Path to sketch weight')



args = parser.parse_args()

# Initiate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = args.latent_dim
obj_weight=args.obj_weight
skt_weight=args.skt_weight

# Load Model
## Object extraction
obj_kwargs, obj_state = torch.load(args.obj_weight)
obj_extractor = Base3DObjectRingsExtractor(**obj_kwargs)
obj_embedder = MLP(obj_extractor, latent_dim=latent_dim)
obj_embedder=obj_embedder.load_state_dict(obj_state).to(device)

## sketch extraction
query_kwargs, query_state = torch.load(args.skt_weight)
skt_cnn_backbone = str(query_kwargs['version'])
if skt_cnn_backbone.startswith('resnet'):
    query_extractor = ResNetExtractor(skt_cnn_backbone)
elif skt_cnn_backbone.startswith('efficientnet'):
    query_extractor = EfficientNetExtractor(skt_cnn_backbone)
else:
    raise NotImplementedError
query_embedder = MLP(query_extractor, latent_dim=latent_dim)
query_embedder=query_embedder.load_state_dict(query_state).to(device)

# Load data




