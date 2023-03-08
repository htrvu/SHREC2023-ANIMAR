import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from common.predict import predict

from pointcloud.dataset import SHREC23_PointCloudData_TextQuery, SHREC23_Test_PointCloudData_Objects
from pointcloud.curvenet import CurveNet

from common.models import BertExtractor, MLP
from common.test import test_loop
from common.train import train_loop
from pointcloud.pointmlp import PointMLP, PointMLPElite
from utils.plot_logs import plot_logs

parser = argparse.ArgumentParser()
parser.add_argument('--pcl-model', type=str,
                    default='curvenet', choices=['curvenet', 'pointmlp', 'pointmlpelite'], help='Model for point cloud feature extraction')
parser.add_argument('--info-json', type=str, required=True, help='Path to model infomation json')
parser.add_argument('--output-path', type=str, default='./prompt', help='Path to output folder')
parser.add_argument('--obj-data-path', type=str, required=True, help='Path to 3D objects folder')
parser.add_argument('--obj-csv-path', type=str, required=True, help='Path to CSV file of objects')
parser.add_argument('--txt-csv-path', type=str, help='Path to CSV file of prompts')
parser.add_argument('--obj-weight', type=str, required=True, help='Path to 3D object weight')
parser.add_argument('--txt-weight', type=str, required=True, help='Path to prompt weight')

args = parser.parse_args()

#Info json
with open(args.info_json) as json_file:
    arg_dict = json.load(json_file)

batch_size = arg_dict['batch_size']
latent_dim = arg_dict['latent_dim']

# Initialize
## Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obj_weight=args.obj_weight
query_weight=args.query_weight

## Storage
# output folder
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
folders = os.listdir(output_path)
new_id = 0
if len(folders) > 0:
    for folder in folders:
        if not folder.startswith('pcl_predict_'):
            continue
        new_id = max(new_id, int(folder.split('pcl_predict_')[-1]))
    new_id += 1
output_path = os.path.join(output_path, f'pcl_predict_{new_id}')
os.makedirs(output_path)

# Load Model
## Get weight
### For Object Extraction
obj_state = torch.load(args.obj_weight)[0]
### For Text Extraction
query_state= torch.load(args.txt_weight)[0]

## Construct model
### For Object Extraction
if arg_dict['pcl_model'] == 'curvenet':
    obj_extractor = CurveNet(device=device)
elif arg_dict['pcl_model'] == 'pointmlp':
    obj_extractor = PointMLP(device=device)
elif arg_dict['pcl_model'] == 'pointmlpelite':
    obj_extractor = PointMLPElite(device=device)
else:
    raise NotImplementedError

## For Text Extraction
query_extractor = BertExtractor() # OOM, so freeze for baseline

## Apply weights
### For Object Extraction
obj_embedder = MLP(obj_extractor, latent_dim=latent_dim)
obj_embedder.load_state_dict(obj_state)
obj_embedder = obj_embedder.to(device)
### For Text Extraction
query_embedder = MLP(query_extractor, latent_dim=latent_dim)
query_embedder.load_state_dict(query_state)
query_embedder = query_embedder.to(device)

# Load data
obj_ds = SHREC23_Test_PointCloudData_Objects(obj_data_path=args.obj_data_path,
                                             csv_data_path=args.obj_csv_path)

txt_ds = SHREC23_Test_SketchesData(skt_data_path=args.skt_data_path,
                                    csv_data_path=args.skt_csv_path)

## Initialize dataloader
obj_dl = DataLoader(obj_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=obj_ds.collate_fn)
txt_dl = DataLoader(txt_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=txt_ds.collate_fn)

# Predict
predict(obj_embedder=obj_embedder, query_embedder=query_embedder,
        obj_input='pointclouds', query_input='tokens',
        obj_dl=obj_dl,
        query_dl=txt_dl,
        dimension=latent_dim,
        output_path=output_path,
        device=device)