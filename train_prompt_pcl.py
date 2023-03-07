import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory

from pointcloud.dataset import SHREC23_PointCloudData_TextQuery
from pointcloud.curvenet import CurveNet

from common.models import BertExtractor, MLP
from common.test import test_loop
from common.train import train_loop

parser = argparse.ArgumentParser()
parser.add_argument('--pcl-model', type=str,
                    default='curvenet', choices=['curvenet', 'pointmlp', 'pointmlpelite'], help='Model for point cloud feature extraction')

parser.add_argument('--obj-data-path', type=str,
                    required=True, help='Path to 3D objects folder')
parser.add_argument('--skt-data-path', type=str,
                    required=True, help='Path to 3D sketches folder')
parser.add_argument('--train-csv-path', type=str, required=True,
                    help='Path to CSV file of mapping object and sketch in training set')
parser.add_argument('--test-csv-path', type=str, required=True,
                    help='Path to CSV file of mapping object and sketch in test set')

parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
parser.add_argument('--num-workers', type=int,
                    default=1, help='Num of workers')
parser.add_argument('--lr-obj', type=float, default=1e-4,
                    help='Learning rate for object\'s network')
parser.add_argument('--lr-skt', type=float, default=1e-4,
                    help='Learning rate for sketch\'s network')
parser.add_argument('--use-cbm', default=False, action='store_true',
                    help='Use cross batch memory in training')
parser.add_argument('--reduce-lr', default=False, action='store_true',
                    help='Use cross batch memory in training')

parser.add_argument('--latent-dim', type=int, default=128,
                    help='Latent dimensions of common embedding space')

parser.add_argument('--output-path', type=str,
                    default='./exps', help='Path to output folder')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init
batch_size = args.batch_size
latent_dim = args.latent_dim
epoch = args.epochs



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obj_extractor = CurveNet(device=device)

obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)
query_extractor = BertExtractor() # OOM, so freeze for baseline
query_embedder = MLP(query_extractor,latent_dim=latent_dim).to(device)

train_ds = SHREC23_PointCloudData_TextQuery(obj_data_path='data/SketchANIMAR2023/3D_Model_References/References',
                                             csv_data_path='data/csv/train_tex.csv')

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_ds.collate_fn)

test_ds = SHREC23_PointCloudData_TextQuery(obj_data_path='data/SketchANIMAR2023/3D_Model_References/References',
                                             csv_data_path='data/csv/test_tex.csv')

test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=test_ds.collate_fn)

contra_loss = NTXentLoss()
cbm_query = CrossBatchMemory(contra_loss, latent_dim, 128)
cbm_object = CrossBatchMemory(contra_loss, latent_dim, 128)

# Set optimizers
optimizer1 = torch.optim.Adam(obj_embedder.parameters(), lr=0.00001, weight_decay=0.0001)
optimizer2 = torch.optim.Adam(query_embedder.parameters(), lr=0.00001, weight_decay=0.0001)

for e in range(epoch):
    print(f'Epoch {e+1}/{epoch}:')
    loss = train_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
               obj_input='pointclouds', query_input='tokens',
               cbm_query=cbm_query, cbm_object=cbm_object,
               obj_optimizer=optimizer1, query_optimizer=optimizer2,
               dl=train_dl,
               device=device)
    
    print(f'Loss: {loss:.4f}')

    test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='pointclouds', query_input='tokens',
              dl=test_dl,
              dimension=latent_dim,
              device=device)