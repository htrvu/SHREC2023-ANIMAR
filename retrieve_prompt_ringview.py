import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from common.dataset import SHREC23_Test_TextData
from common.predict import predict

from common.models import BertExtractor, MLP
from ringnet.dataset import SHREC23_Test_Rings_Objects
from ringnet.models import Base3DObjectRingsExtractor

'''
python retrieve_prompt_ringview.py \
    --info-json exps/ringview_exp_13/args.json \
    --rings-path data/TextANIMAR2023/3D_Model_References/generated_sketches \
    --obj-csv-path data/TextANIMAR2023/3D_Model_References/References/References.csv \
    --txt-csv-path data/TextANIMAR2023/Test/TextQuery_Test.csv \
    --obj-weight exps/ringview_exp_13/weights/best_obj_embedder.pth \
    --txt-weight exps/ringview_exp_13/weights/best_query_embedder.pth

'''

parser = argparse.ArgumentParser()
parser.add_argument('--info-json', type=str, required=True, help='Path to model infomation json')
parser.add_argument('--rings-path', type=str, required=True, help='Path to parent folder of ringviews')
parser.add_argument('--output-path', type=str, default='./prompt', help='Path to output folder')
parser.add_argument('--obj-csv-path', type=str, required=True, help='Path to CSV file of objects')
parser.add_argument('--txt-csv-path', type=str, help='Path to CSV file of prompts')
parser.add_argument('--obj-weight', type=str, required=True, help='Path to 3D object weight')
parser.add_argument('--txt-weight', type=str, required=True, help='Path to prompt weight')

args = parser.parse_args()

#Info json
with open(args.info_json) as json_file:
    arg_dict = json.load(json_file)

# Initialize
## Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obj_weight=args.obj_weight
query_weight=args.txt_weight
batch_size = arg_dict['batch_size']
latent_dim = arg_dict['latent_dim']
obj_weight=args.obj_weight
txt_weight=args.txt_weight
output_path=args.output_path
num_workers=arg_dict['num_workers']
ring_ids = [int(id) for id in arg_dict['used_rings'].split(',')]

## Storage
# output folder
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
folders = os.listdir(output_path)
new_id = 0
if len(folders) > 0:
    for folder in folders:
        if not folder.startswith('ringview_predict_'):
            continue
        new_id = max(new_id, int(folder.split('ringview_predict_')[-1]))
    new_id += 1
output_path = os.path.join(output_path, f'ringview_predict_{new_id}')
os.makedirs(output_path)

# Load Model
## Get weight
### For Object Extraction
obj_kwargs, obj_state = torch.load(args.obj_weight)
### For Text Extraction
query_state= torch.load(args.txt_weight)[0]

## Construct model
### For Object Extraction
obj_extractor = Base3DObjectRingsExtractor(**obj_kwargs)


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
obj_ds = SHREC23_Test_Rings_Objects(args.obj_csv_path, args.rings_path, ring_ids)


txt_ds = SHREC23_Test_TextData(csv_data_path=args.txt_csv_path)

## Initialize dataloader
obj_dl = DataLoader(obj_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=obj_ds.collate_fn)
txt_dl = DataLoader(txt_ds, batch_size=batch_size,
                    shuffle=False, num_workers=arg_dict['num_workers'], collate_fn=txt_ds.collate_fn)

# Inference
predict(obj_embedder=obj_embedder, query_embedder=query_embedder,
        obj_input='object_ims', query_input='tokens',
        obj_dl=obj_dl,
        query_dl=txt_dl,
        dimension=latent_dim,
        output_path=output_path,
        device=device)