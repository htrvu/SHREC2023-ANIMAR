import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from torch.optim.lr_scheduler import StepLR

from ringnet.dataset import SHREC23_Rings_RenderOnly_TextQuery
from ringnet.models import Base3DObjectRingsExtractor

from common.models import BertExtractor, MLP, EfficientNetExtractor, ResNetExtractor,ClipTextExtractor,ClipVisionExtractor
from common.test import test_loop
from common.train import train_loop, val_loop

from utils.plot_logs import plot_logs

'''
python train_prompt_ringview.py \
    --view-cnn-backbone efficientnet_b2 \
    --text-model distilbert-base-uncased \
    --rings-path data/TextANIMAR2023/3D_Model_References/generated_sketches \
    --used-rings 3,4 \
    --train-csv-path data/csv/train_tex.csv \
    --test-csv-path data/csv/test_tex.csv \
    --batch-size 2 \
    --epochs 100 \
    --latent-dim 256 \
    --output-path exps \
    --view-seq-embedder mha \
    --num-rings-mhas 2 \
    --num-heads 4 \
    --lr-obj 3e-5 \
    --lr-txt 3e-5
'''

parser = argparse.ArgumentParser()
parser.add_argument('--view-cnn-backbone', type=str, default='efficientnet_b2',
                    choices=[*ResNetExtractor.arch.keys(), *EfficientNetExtractor.arch.keys(),*ClipVisionExtractor.arch],  help='Model for ringview feature extraction')
parser.add_argument('--text-model', type=str,
                    default='distilbert-base-uncased', help='Model for text feature extraction')
parser.add_argument('--rings-path', type=str, required=True,
                    help='Path to parent folder of ringviews')
parser.add_argument('--used-rings', type=str,
                    default='0,1,2,3,4,5,6', help='Rings to be used for training')
parser.add_argument('--train-csv-path', type=str, required=True,
                    help='Path to CSV file of mapping object and prompt in training set')
parser.add_argument('--test-csv-path', type=str, required=True,
                    help='Path to CSV file of mapping object and prompt in test set')

parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Num of epochs')
parser.add_argument('--num-workers', type=int,
                    default=1, help='Num of workers')
parser.add_argument('--lr-obj', type=float, default=1e-4,
                    help='Learning rate for object\'s network')
parser.add_argument('--lr-txt', type=float, default=1e-4,
                    help='Learning rate for text\'s network')
parser.add_argument('--use-cbm', default=False, action='store_true',
                    help='Use cross batch memory in training')
parser.add_argument('--reduce-lr', default=False, action='store_true',
                    help='Use cross batch memory in training')

parser.add_argument('--latent-dim', type=int, default=128,
                    help='Latent dimensions of common embedding space')

parser.add_argument('--view-seq-embedder', type=str, default='bilstm',
                    choices=['bilstm', 'mha'], help='Embedder for all views of a ring')
parser.add_argument('--num-rings-mhas', type=int, default=1,
                    help='Num of MHA layers for all rings self-attention')
parser.add_argument('--num-heads', type=int, default=4,
                    help='Num of heads in each MHA layers')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate of MHA layers')

parser.add_argument('--output-path', type=str,
                    default='./prompt', help='Path to output folder')

args = parser.parse_args()

# Init
batch_size = args.batch_size
latent_dim = args.latent_dim
epoch = args.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ring_ids = [int(id) for id in args.used_rings.split(',')]

# Output folder
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
folders = os.listdir(output_path)
new_id = 0
if len(folders) > 0:
    for folder in folders:
        if not folder.startswith('ringview_exp_'):
            continue
        new_id = max(new_id, int(folder.split('ringview_exp_')[-1]))
    new_id += 1
output_path = os.path.join(output_path, f'ringview_exp_{new_id}')
os.makedirs(output_path)
weights_path = os.path.join(output_path, 'weights')
os.mkdir(weights_path)

# Object extraction model
view_cnn_backbone = args.view_cnn_backbone
obj_extractor = Base3DObjectRingsExtractor(
    view_cnn_backbone=view_cnn_backbone,
    num_rings=len(ring_ids),
    view_seq_embedder=args.view_seq_embedder,
    num_mhas=args.num_rings_mhas,
    num_heads=args.num_heads,
    dropout=args.dropout,
)
obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)

# Query model extractor
if args.text_model.startswith('openai'):
    query_extractor = ClipTextExtractor(version=args.text_model,is_frozen=True) # OOM, so freeze for baseline
else:
    query_extractor = BertExtractor(version=args.text_model,is_frozen=True) 
query_embedder = MLP(query_extractor,latent_dim=latent_dim).to(device)

# Data loader

train_ds = SHREC23_Rings_RenderOnly_TextQuery(
        args.train_csv_path, args.rings_path, None, ring_ids)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_ds.collate_fn)

test_ds = SHREC23_Rings_RenderOnly_TextQuery(
        args.train_csv_path, args.rings_path, None, ring_ids)

test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=test_ds.collate_fn)


contra_loss = NTXentLoss()
cbm_query = CrossBatchMemory(contra_loss, latent_dim, 128)
cbm_object = CrossBatchMemory(contra_loss, latent_dim, 128)

# Set optimizers
optimizer1 = torch.optim.AdamW(
    obj_embedder.parameters(), lr=args.lr_obj, weight_decay=0.0001)
optimizer2 = torch.optim.AdamW(
    query_embedder.parameters(), lr=args.lr_txt, weight_decay=0.0001)

# Set Scheduler
if args.reduce_lr:
    obj_scheduler = StepLR(optimizer1, step_size=15, gamma=0.1)
    query_scheduler = StepLR(optimizer2, step_size=15, gamma=0.1)

    prev_obj_lr, prev_query_lr = optimizer1.param_groups[
        0]['lr'], optimizer2.param_groups[0]['lr']
        
# Train
training_losses = []
val_losses = []
eval_results = []
best_loss = float('inf')

for e in range(epoch):
    print(f'Epoch {e+1}/{epoch}:')
    train_loss = train_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
                      obj_input='object_ims', query_input='tokens',
                      cbm_query=cbm_query, cbm_object=cbm_object,
                      obj_optimizer=optimizer1, query_optimizer=optimizer2,
                      dl=train_dl,
                      device=device,
                      use_cross_batch_mem=args.use_cbm)

    print(f'Training Loss: {train_loss:.4f}')
    training_losses.append(train_loss)

    val_loss = val_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
                        obj_input='object_ims', query_input='tokens',
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
        torch.save([obj_extractor.kwargs, obj_embedder.state_dict()], os.path.join(
            weights_path, 'best_obj_embedder.pth'))
        torch.save([query_embedder.state_dict()], os.path.join(
            weights_path, 'best_query_embedder.pth'))


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
                                obj_input='object_ims', query_input='tokens',
                                dl=test_dl,
                                dimension=latent_dim,
                                device=device,
                                output_path=output_path)

    eval_results.append(metrics_results)


torch.save([obj_extractor.kwargs, obj_embedder.state_dict()], os.path.join(
    weights_path, 'last_obj_embedder.pth'))
torch.save([query_embedder.state_dict()], os.path.join(
    weights_path, 'last_query_embedder.pth'))

obj_embedder.load_state_dict(torch.load(os.path.join(weights_path, 'best_obj_embedder.pth'))[1])
query_embedder.load_state_dict(torch.load(os.path.join(weights_path, 'best_query_embedder.pth'))[0])
print('Best weights result:')
metrics_results = test_loop(obj_embedder=obj_embedder, query_embedder=query_embedder,
                                obj_input='object_ims', query_input='tokens',
                                dl=test_dl,
                                dimension=latent_dim,
                                device=device,
                                output_path=output_path)

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
plot_logs(training_losses, val_losses, NNs, P10s, NDCGs, mAPs, f'{output_path}/results.png')
