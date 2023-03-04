from common.test import test_loop
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
parser.add_argument('--rings-path', type=str, required=True, help='Path to parent folder of ringviews')
parser.add_argument('--skt-data-path', type=str, required=True, help='Path to 3D sketches folder')
parser.add_argument('--test-csv-path', type=str, required=True, help='Path to CSV file of sketch in test set')
parser.add_argument('--obj-weight', type=str, required=True, help='Path to object weight')
parser.add_argument('--skt-weight', type=str, required=True, help='Path to sketch weight')
parser.add_argument('--output-path', type=str,
                    default='./predict', help='Path to output folder')

#Tuning weight
parser.add_argument('--used-rings', type=str, default='0,1,2,3,4,5,6', help='Rings to be used for training')




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
        new_id = max(new_id, int(folder.split('ringview_exp_')[-1]))
    new_id += 1
sub_output_path = os.path.join(output_path, f'ringview_exp_{new_id}')
os.makedirs(sub_output_path)

# Initiate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = arg_dict['batch_size']
latent_dim = arg_dict['latent_dim']
obj_weight=args.obj_weight
skt_weight=args.skt_weight
output_path=args.output_path
num_workers=arg_dict['num_workers']
#ring id (need to review this arg)
ring_ids = [int(id) for id in args.used_rings.split(',')]


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

# Load data (need run test)
test_ds = SHREC23_Rings_RenderOnly_ImageQuery(
        args.test_csv_path, args.rings_path, args.skt_data_path, ring_ids)

test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, collate_fn=test_ds.collate_fn)



# Inference (need empty target id check)
test_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
                                obj_input='object_ims', query_input='query_ims',
                                dl=test_dl,
                                dimension=latent_dim,
                                device=device,
                                output_path=output_path)

# json to csv or rewrite inference base on test_loop
