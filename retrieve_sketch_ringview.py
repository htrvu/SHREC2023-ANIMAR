import torch
from torch.utils.data import DataLoader

from ringnet.dataset import SHREC23_Test_Rings_Objects
from ringnet.models import Base3DObjectRingsExtractor

from common.predict import predict
from common.models import MLP, EfficientNetExtractor, ResNetExtractor
from common.dataset import SHREC23_Test_SketchesData

import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser()

# Directory 
parser.add_argument('--info-json', type=str, required=True, help='Path to model infomation json')
parser.add_argument('--rings-path', type=str, required=True, help='Path to parent folder of ringviews')
parser.add_argument('--obj-csv-path', type=str, required=True, help='Path to CSV file of objects')
parser.add_argument('--skt-data-path', type=str, required=True, help='Path to 3D sketches folder')
parser.add_argument('--skt-csv-path', type=str, required=True, help='Path to CSV file of sketch in test set')
parser.add_argument('--obj-weight', type=str, required=True, help='Path to object weight')
parser.add_argument('--skt-weight', type=str, required=True, help='Path to sketch weight')
parser.add_argument('--output-path', type=str,
                    default='./predicts', help='Path to output folder')

args = parser.parse_args()

#Info json
with open(args.info_json) as json_file:
    arg_dict = json.load(json_file)

# Output folder
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
folders = os.listdir(output_path)
new_id = 0
if len(folders) > 0:
    for folder in folders:
        if folder.startswith('pcl'):
            continue
        new_id = max(new_id, int(folder.split('ringview_predict_')[-1]))
    new_id += 1
sub_output_path = os.path.join(output_path, f'ringview_predict_{new_id}')
os.makedirs(sub_output_path)

# Initiate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = arg_dict['batch_size']
latent_dim = arg_dict['latent_dim']
obj_weight=args.obj_weight
skt_weight=args.skt_weight
output_path=args.output_path
num_workers=arg_dict['num_workers']
ring_ids = [int(id) for id in arg_dict['used_rings'].split(',')]

# Load Model
## Object extraction
obj_kwargs, obj_state = torch.load(args.obj_weight)
obj_extractor = Base3DObjectRingsExtractor(**obj_kwargs)
obj_embedder = MLP(obj_extractor, latent_dim=latent_dim)
obj_embedder.load_state_dict(obj_state)
obj_embedder = obj_embedder.to(device)

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
query_embedder.load_state_dict(query_state)
query_embedder = query_embedder.to(device)

# Load data (need run test)
obj_ds = SHREC23_Test_Rings_Objects(args.obj_csv_path, args.rings_path, ring_ids)

skt_ds = SHREC23_Test_SketchesData(skt_data_path=args.skt_data_path,
                                                csv_data_path=args.skt_csv_path)

obj_dl = DataLoader(obj_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=obj_ds.collate_fn)
skt_dl = DataLoader(skt_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=skt_ds.collate_fn)


# Inference
metrics_results = predict(obj_embedder=obj_embedder, query_embedder=query_embedder,
        obj_input='object_ims', query_input='query_ims',
        obj_dl=obj_dl,
        query_dl=skt_dl,
        dimension=latent_dim,
        output_path=sub_output_path,
        device=device)