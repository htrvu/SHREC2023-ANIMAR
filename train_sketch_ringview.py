import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory

from ringnet.dataset import SHREC23_Rings_RenderOnly_ImageQuery
from ringnet.models import Base3DObjectRingsExtractor
from common.models import ResNetExtractor, EfficientNetExtractor, MLP

from common.test import test_loop
from common.train import train_loop

from utils.plot_logs import plot_logs

import numpy as np
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--view-cnn-backbone', type=str, default='efficientnet_b2',
                    choices=[*ResNetExtractor.arch.keys(), *EfficientNetExtractor.arch.keys()],  help='Model for ringview feature extraction')
parser.add_argument('--skt-cnn-backbone', type=str, default='efficientnet_b2',
                    choices=[*ResNetExtractor.arch.keys(), *EfficientNetExtractor.arch.keys()],  help='Model for sketch feature extraction')

parser.add_argument('--rings-path', type=str, required=True, help='Path to parent folder of ringviews')
parser.add_argument('--num-rings', type=int, default=6, help='Num of rings (from ring1 to)')
parser.add_argument('--skt-data-path', type=str, required=True, help='Path to 3D sketches folder')
parser.add_argument('--train-csv-path', type=str, required=True, help='Path to CSV file of mapping object and sketch in training set')
parser.add_argument('--test-csv-path', type=str, required=True, help='Path to CSV file of mapping object and sketch in test set')

parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
parser.add_argument('--num-workers', type=int, default=1, help='Num of workers')
parser.add_argument('--lr-obj', type=float, default=1e-4, help='Learning rate for object\'s network')
parser.add_argument('--lr-skt', type=float, default=1e-4, help='Learning rate for sketch\'s network')
parser.add_argument('--use-cbm', type=bool, default=False, help='Use cross batch memory in training')

parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimensions of common embedding space')

parser.add_argument('--view-seq-embedder', type=str, default='bilstm', choices=['bilstm', 'mha'], help='Embedder for all views of a ring')
parser.add_argument('--num-rings-mhas', type=int, default=1, help='Num of MHA layers for all rings self-attention')
parser.add_argument('--num-heads', type=int, default=4, help='Num of heads in each MHA layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate of MHA layers')

parser.add_argument('--output-path', type=str, default='./exps', help='Path to output folder')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = args.batch_size
latent_dim = args.latent_dim
epoch = args.epochs

# output folder
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
output_path = os.path.join(output_path, f'ringview_exp_{new_id}')
os.makedirs(output_path)
weights_path = os.path.join(output_path, 'weights')
os.mkdir(weights_path)

# object extraction model
view_cnn_backbone = args.view_cnn_backbone
obj_extractor = Base3DObjectRingsExtractor(
    view_cnn_backbone = view_cnn_backbone,
    num_rings = args.num_rings,
    view_seq_embedder = args.view_seq_embedder,
    num_mhas = args.num_rings_mhas,
    num_heads = args.num_heads,
    dropout = args.dropout,
)
obj_embedder = MLP(obj_extractor,latent_dim=latent_dim).to(device)

# sketch extraction model
skt_cnn_backbone = args.skt_cnn_backbone
if skt_cnn_backbone.startswith('resnet'):
    query_extractor = ResNetExtractor(skt_cnn_backbone)
elif skt_cnn_backbone.startswith('efficientnet'):
    query_extractor = EfficientNetExtractor(skt_cnn_backbone)
else:
    raise NotImplementedError
query_embedder = MLP(query_extractor, latent_dim=latent_dim).to(device)

# datasets
# 'data/SketchANIMAR2023/3D_Model_References/generated_models'
train_ds = SHREC23_Rings_RenderOnly_ImageQuery(
        args.train_csv_path, args.rings_path, args.skt_data_path, 1 + np.arange(args.num_rings))

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_ds.collate_fn)

test_ds = SHREC23_Rings_RenderOnly_ImageQuery(
        args.test_csv_path, args.rings_path, args.skt_data_path, 1 + np.arange(args.num_rings))

test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_ds.collate_fn)

contra_loss = NTXentLoss()
cbm_query = CrossBatchMemory(contra_loss, latent_dim, 128)
cbm_object = CrossBatchMemory(contra_loss, latent_dim, 128)

# Set optimizers
optimizer1 = torch.optim.AdamW(obj_embedder.parameters(), lr=args.lr_obj, weight_decay=0.0001)
optimizer2 = torch.optim.AdamW(query_embedder.parameters(), lr=args.lr_skt, weight_decay=0.0001)

training_losses = []
eval_results = []
best_mAP = 0
for e in range(epoch):
    print(f'Epoch {e+1}/{epoch}:')
    loss = train_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
               obj_input='object_ims', query_input='query_ims',
               cbm_query=cbm_query, cbm_object=cbm_object,
               obj_optimizer=optimizer1, query_optimizer=optimizer2,
               dl=train_dl,
               device=device)
    print(f'Loss: {loss:.4f}')
    training_losses.append(loss)

    metrics_results = test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='object_ims', query_input='query_ims',
              dl=test_dl,
              dimension=latent_dim,
              device=device,
              output_path=output_path)

    if metrics_results['mAP'] > best_mAP:
        best_mAP = metrics_results['mAP']
        # save weights
        torch.save(obj_embedder.state_dict(), os.path.join(weights_path, 'best_obj_embedder.pth'))
        torch.save(query_embedder.state_dict(), os.path.join(weights_path, 'best_query_embedder.pth'))
    eval_results.append(metrics_results)

torch.save(obj_embedder.state_dict(), os.path.join(weights_path, 'last_obj_embedder.pth'))
torch.save(query_embedder.state_dict(), os.path.join(weights_path, 'last_query_embedder.pth'))

with open(os.path.join(output_path, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f)

# plot metrics
NNs = []
P10s = []
NDCGs = []
mAPs = []
for res in eval_results:
    NNs.append(res['NN'])
    P10s.append(res['P@10'])
    NDCGs.append(res['NDCG'])
    mAPs.append(res['mAP'])
plot_logs(training_losses, NNs, P10s, NDCGs, mAPs, f'{output_path}/results.png')