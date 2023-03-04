from common.test import test_loop
from pointcloud.curvenet import CurveNet
from pointcloud.dataset import SHREC23_PointCloudData_ImageQuery
from pointcloud.pointmlp import PointMLP, PointMLPElite
from ringnet.dataset import SHREC23_Rings_RenderOnly_ImageQuery
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
parser.add_argument('--info-json', type=str, required=True, help='Path to model infomation json')
parser.add_argument('--obj-data-path', type=str, required=True, help='Path to 3D objects folder')
parser.add_argument('--skt-data-path', type=str, required=True, help='Path to 3D sketches folder')
parser.add_argument('--test-csv-path', type=str, required=True, help='Path to CSV file of sketch id')
parser.add_argument('--output-path', type=str,
                    default='./predict', help='Path to output folder')

#Tuning weight
parser.add_argument('--latent-dim', type=int, default=128,
                    help='Latent dimensions of common embedding space')

parser.add_argument('--obj-weight', type=str, required=True, help='Path to object weight')
parser.add_argument('--skt-weight', type=str, required=True, help='Path to sketch weight')
parser.add_argument('--output-path', type=str,
                    default='./predict', help='Path to output folder')


args = parser.parse_args()
#Info json
with open(args.info_json) as json_file:
    arg_dict = json.load(json_file)

# Initiate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = arg_dict['batch_size']
latent_dim = args.latent_dim
obj_weight=args.obj_weight
skt_weight=args.skt_weight
output_path=args.output_path
num_workers=arg_dict['num_workers']
pcl_model=arg_dict['pcl_model']


# Load Model
## Object extraction
obj_state = torch.load(args.obj_weight)
if pcl_model == 'curvenet':
    obj_extractor = CurveNet(device=device)
elif pcl_model == 'pointmlp':
    obj_extractor = PointMLP(device=device)
elif pcl_model == 'pointmlpelite':
    obj_extractor = PointMLPElite(device=device)
else:
    raise NotImplementedError

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

# Load data (need run test)
test_ds = SHREC23_PointCloudData_ImageQuery(obj_data_path=args.obj_data_path,
                                            csv_data_path=args.test_csv_path,
                                            skt_root=args.skt_data_path)

test_dl = DataLoader(test_ds, batch_size=batch_size,
                    shuffle=False, num_workers=args.num_workers, collate_fn=test_ds.collate_fn)



# Inference (need empty target id check)
test_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
            obj_input='pointclouds', query_input='query_ims',
            dl=test_dl,
            dimension=latent_dim,
            output_path = output_path,
            device=device)

# json to csv or rewrite inference base on test_loop
