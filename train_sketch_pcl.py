import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from torch.optim.lr_scheduler import StepLR

from pointcloud.dataset import SHREC23_PointCloudData_ImageQuery
from pointcloud.curvenet import CurveNet
from pointcloud.pointmlp import PointMLP, PointMLPElite

from common.models import ResNetExtractor, EfficientNetExtractor, MLP
from common.test import test_loop
from common.train import train_loop, val_loop

from utils.plot_logs import plot_logs

import os
import json
import argparse
# from utils.seed import seed_everything

# seed_everything()

parser = argparse.ArgumentParser()
parser.add_argument('--pcl-model', type=str,
                    default='curvenet', choices=['curvenet', 'pointmlp', 'pointmlpelite'], help='Model for point cloud feature extraction')
parser.add_argument('--cnn-backbone', type=str, default='efficientnet_b2',
                    choices=[*ResNetExtractor.arch.keys(), *EfficientNetExtractor.arch.keys()],  help='Model for sketch feature extraction')

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
        if not folder.startswith('pcl_exp_'):
            continue
        new_id = max(new_id, int(folder.split('pcl_exp_')[-1]))
    new_id += 1
output_path = os.path.join(output_path, f'pcl_exp_{new_id}')
os.makedirs(output_path)
weights_path = os.path.join(output_path, 'weights')
os.mkdir(weights_path)

# object extraction model
if args.pcl_model == 'curvenet':
    obj_extractor = CurveNet(device=device)
elif args.pcl_model == 'pointmlp':
    obj_extractor = PointMLP(device=device)
elif args.pcl_model == 'pointmlpelite':
    obj_extractor = PointMLPElite(device=device)
else:
    raise NotImplementedError

obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)

# sketch extraction model
cnn_backbone = args.cnn_backbone
if cnn_backbone.startswith('resnet'):
    query_extractor = ResNetExtractor(cnn_backbone)
elif cnn_backbone.startswith('efficientnet'):
    query_extractor = EfficientNetExtractor(cnn_backbone)
else:
    raise NotImplementedError

query_embedder = MLP(query_extractor, latent_dim=latent_dim).to(device)

# datasets
train_ds = SHREC23_PointCloudData_ImageQuery(obj_data_path=args.obj_data_path,
                                             csv_data_path=args.train_csv_path,
                                             skt_root=args.skt_data_path)

train_dl = DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, num_workers=args.num_workers, collate_fn=train_ds.collate_fn)

test_ds = SHREC23_PointCloudData_ImageQuery(obj_data_path=args.obj_data_path,
                                            csv_data_path=args.test_csv_path,
                                            skt_root=args.skt_data_path)

test_dl = DataLoader(test_ds, batch_size=batch_size,
                     shuffle=False, num_workers=args.num_workers, collate_fn=test_ds.collate_fn)

contra_loss = NTXentLoss()
cbm_query = CrossBatchMemory(contra_loss, latent_dim, 128)
cbm_object = CrossBatchMemory(contra_loss, latent_dim, 128)

# Set optimizers
optimizer1 = torch.optim.AdamW(
    obj_embedder.parameters(), lr=args.lr_obj, weight_decay=0.0001)
optimizer2 = torch.optim.AdamW(
    query_embedder.parameters(), lr=args.lr_skt, weight_decay=0.0001)

# Set Scheduler
if args.reduce_lr:
    obj_scheduler = StepLR(optimizer1, step_size=15, gamma=0.1)
    query_scheduler = StepLR(optimizer2, step_size=15, gamma=0.1)

    prev_obj_lr, prev_query_lr = optimizer1.param_groups[
        0]['lr'], optimizer2.param_groups[0]['lr']

training_losses = []
val_losses = []
eval_results = []
# best_NDCG = 0
best_loss = float('inf')

for e in range(epoch):
    print(f'Epoch {e+1}/{epoch}:')
    
    train_loss = train_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
                            obj_input='pointclouds', query_input='query_ims',
                            cbm_query=cbm_query, cbm_object=cbm_object,
                            obj_optimizer=optimizer1, query_optimizer=optimizer2,
                            dl=train_dl,
                            device=device,
                            use_cross_batch_mem=args.use_cbm)
    print(f'Training loss: {train_loss:.4f}')
    training_losses.append(train_loss)

    val_loss = val_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
                        obj_input='pointclouds', query_input='query_ims',
                        cbm_query=cbm_query, cbm_object=cbm_object,
                        dl=test_dl,
                        device=device,
                        use_cross_batch_mem=args.use_cbm)
    print(f'Val loss: {val_loss:.4f}')
    val_losses.append(val_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss
        print('Saving best weights...')
        # save weights
        torch.save([obj_embedder.state_dict()], os.path.join(
            weights_path, 'best_obj_embedder.pth'))
        torch.save([query_extractor.kwargs, query_embedder.state_dict()],
                   os.path.join(weights_path, 'best_query_embedder.pth'))
        
    if args.reduce_lr:
        obj_scheduler.step()
        query_scheduler.step()

        # Print the loss and learning rate for each epoch
        curr_obj_lr, curr_query_lr = optimizer1.param_groups[
            0]['lr'], optimizer2.param_groups[0]['lr']

        if curr_obj_lr != prev_obj_lr:
            print('Object learning rate changed from {:.7f} to {:.7f}'.format(
                prev_obj_lr, curr_obj_lr))
            prev_obj_lr = curr_obj_lr

        if curr_query_lr != prev_query_lr:
            print('Query learning rate changed from {:.7f} to {:.7f}'.format(
                prev_query_lr, curr_query_lr))
            prev_query_lr = curr_query_lr

    metrics_results = test_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
            obj_input='pointclouds', query_input='query_ims',
            dl=test_dl,
            dimension=latent_dim,
            output_path = output_path,
            device=device)
    
    eval_results.append(metrics_results)

torch.save([obj_embedder.state_dict()], os.path.join(
    weights_path, 'last_obj_embedder.pth'))
torch.save([query_embedder.state_dict()],
           os.path.join(weights_path, 'last_query_embedder.pth'))

obj_embedder.load_state_dict(torch.load(os.path.join(weights_path, 'best_obj_embedder.pth'))[0])
query_embedder.load_state_dict(torch.load(os.path.join(weights_path, 'best_query_embedder.pth'))[0])
print('Best weights result (it maybe has some randomness):')
metrics_results = test_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
        obj_input='pointclouds', query_input='query_ims',
        dl=test_dl,
        dimension=latent_dim,
        output_path=output_path,
        device=device)

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
plot_logs(training_losses, val_losses, NNs, P10s, NDCGs,
          mAPs, f'{output_path}/results.png')
