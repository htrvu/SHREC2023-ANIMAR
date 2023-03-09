import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory

from pointcloud.dataset import SHREC23_Test_PointCloudData_Objects
from pointcloud.curvenet import CurveNet
from pointcloud.pointmlp import PointMLP, PointMLPElite

from common.dataset import SHREC23_Test_SketchesData
from common.models import ResNetExtractor, EfficientNetExtractor, MLP
from common.predict import predict

import os
import json
import argparse

'''
python retrieve_sketch_pcl.py \
    --pcl-model pointmlp \
    --cnn-backbone efficientnet_v2_s\
    --output-path exps \
    --obj-data-path /kaggle/input/shrec23/SketchANIMAR2023/3D_Model_References/References \
    --skt-data-path /kaggle/input/shrec23/SketchANIMAR2023/Train/CroppedSketchQuery_Train \
    --skt-csv-path /kaggle/input/shrec23/csv/test_skt.csv \
    --batch-size 4 \
    --latent-dim 128 \
    --obj-weight exps/pcl_exp_0/weights/best_obj_embedder.pth \
    --skt-weight pcl_exp_0/weights/best_query_embedder.pth

'''

parser = argparse.ArgumentParser()
parser.add_argument('--info-json', type=str, required=True, help='Path to model infomation json')
parser.add_argument('--output-path', type=str, default='./predicts', help='Path to output folder')
parser.add_argument('--obj-data-path', type=str, required=True, help='Path to 3D objects folder')
parser.add_argument('--obj-csv-path', type=str, required=True, help='Path to CSV file of objects')
parser.add_argument('--skt-data-path', type=str, required=True, help='Path to 3D sketches folder')
parser.add_argument('--skt-csv-path', type=str, help='Path to CSV test file of sketches')
parser.add_argument('--obj-weight', type=str, required=True, help='Path to 3D object weight')
parser.add_argument('--skt-weight', type=str, required=True, help='Path to sketch weight')

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
skt_weight=args.skt_weight

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
### For Sketch Extraction
query_kwargs, query_state, *other = torch.load(args.skt_weight)

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

## For Sketch Extraction
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
obj_ds = SHREC23_Test_PointCloudData_Objects(obj_data_path=args.obj_data_path,
                                             csv_data_path=args.obj_csv_path)

skt_ds = SHREC23_Test_SketchesData(skt_data_path=args.skt_data_path,
                                    csv_data_path=args.skt_csv_path)

## Initialize dataloader
obj_dl = DataLoader(obj_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=obj_ds.collate_fn)
skt_dl = DataLoader(skt_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=skt_ds.collate_fn)

predict(obj_embedder=obj_embedder, query_embedder=query_embedder,
        obj_input='pointclouds', query_input='query_ims',
        obj_dl=obj_dl,
        query_dl=skt_dl,
        dimension=latent_dim,
        output_path=output_path,
        device=device)