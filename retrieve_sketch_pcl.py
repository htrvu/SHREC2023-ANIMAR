import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory

from pointcloud.dataset import SHREC23_PointCloudData_ImageQuery
from pointcloud.curvenet import CurveNet
from pointcloud.pointmlp import PointMLP, PointMLPElite

from common.models import ResNetExtractor, EfficientNetExtractor, MLP
from common.test import test_loop
from common.train import train_loop

from utils.plot_logs import plot_logs

import os
import json
import argparse


'''
python retrieve_sketch_pcl.py \
    --pcl-model pointmlp \
    --cnn-backbone efficientnet_v2_s \
    --output-path exps \
    --obj-data-path data/SketchANIMAR2023/3D_Model_References/References \
    --skt-data-path data/SketchANIMAR2023/Train/CroppedSketchQuery_Train \
    --test-csv-path data/csv/test_skt.csv \
    --batch-size 4 \
    --latent-dim 128 \
    --obj-weight exps/pointmlp_effiv2_200epoch/weights/best_obj_embedder.pth \
    --skt-weight exps/pointmlp_effiv2_200epoch/weights/best_query_embedder.pth
'''

parser = argparse.ArgumentParser()
parser.add_argument('--pcl-model', type=str,
                    default='curvenet', choices=['curvenet', 'pointmlp', 'pointmlpelite'], help='Model for point cloud feature extraction')
parser.add_argument('--cnn-backbone', type=str, default='efficientnet_b2',
                    choices=[*ResNetExtractor.arch.keys(), *EfficientNetExtractor.arch.keys()],  help='Model for sketch feature extraction')
parser.add_argument('--output-path', type=str, default='./exps_test', help='Path to output folder')

parser.add_argument('--obj-data-path', type=str, required=True, help='Path to 3D objects folder')
parser.add_argument('--skt-data-path', type=str, required=True, help='Path to 3D sketches folder')
parser.add_argument('--test-csv-path', type=str, required=True, help='Path to CSV file of mapping object and sketch in test set')

parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
parser.add_argument('--num-workers', type=int, default=1, help='Num of workers')

parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimensions of common embedding space')

parser.add_argument('--obj-weight', type=str, required=True, help='Path to 3D object weight')
parser.add_argument('--skt-weight', type=str, required=True, help='Path to sketch weight')

args = parser.parse_args()

# Initialize
## Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = args.latent_dim
obj_weight=args.obj_weight
skt_weight=args.skt_weight
batch_size = args.batch_size
## Storage
# output folder
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
folders = os.listdir(output_path)
new_id = 0
if len(folders) > 0:
    for folder in folders:
        if folder.startswith('ringview'):
            continue
        new_id = max(new_id, int(folder.split('pcl_exp_')[-1]))
    new_id += 1
output_path = os.path.join(output_path, f'pcl_exp_{new_id}')
os.makedirs(output_path)

# Load Model
## Get weight
### For Object Extraction
obj_state, *other = torch.load(args.obj_weight)
### For Sketch Extraction
query_kwargs, query_state, *other = torch.load(args.skt_weight)

## Construct model
### For Object Extraction
if args.pcl_model == 'curvenet':
    obj_extractor = CurveNet(device=device)
elif args.pcl_model == 'pointmlp':
    obj_extractor = PointMLP(device=device)
elif args.pcl_model == 'pointmlpelite':
    obj_extractor = PointMLPElite(device=device)
else:
    raise NotImplementedError
### For Sketch Extraction
cnn_backbone = str(query_kwargs['version'])
if cnn_backbone.startswith('resnet'):
    query_extractor = ResNetExtractor(cnn_backbone)
elif cnn_backbone.startswith('efficientnet'):
    query_extractor = EfficientNetExtractor(cnn_backbone)
else:
    raise NotImplementedError

## Apply weights
### For Object Extraction
obj_embedder = MLP(obj_extractor, latent_dim=latent_dim)
obj_embedder.load_state_dict(obj_state)
obj_embedder = obj_embedder.to(device)
### For Sketch Extraction
query_embedder = MLP(query_extractor, latent_dim=latent_dim)
query_embedder.load_state_dict(query_state)
query_embedder = query_embedder.to(device)


# Load Data
## Load test data
test_ds = SHREC23_PointCloudData_ImageQuery(obj_data_path=args.obj_data_path,
                                            csv_data_path=args.test_csv_path,
                                            skt_root=args.skt_data_path)
## Initialize dataloader for test
test_dl = DataLoader(test_ds, batch_size=batch_size,
                    shuffle=False, num_workers=args.num_workers, collate_fn=test_ds.collate_fn)

metrics_results = test_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
        obj_input='pointclouds', query_input='query_ims',
        dl=test_dl,
        dimension=latent_dim,
        output_path = output_path,
        device=device)