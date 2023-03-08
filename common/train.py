# https://github.com/KevinMusgrave/pytorch-metric-learning/issues/373
import torch
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.reducers import BaseReducer
import time
from tqdm import tqdm
import numpy as np

class PerModality(BaseReducer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def pos_pair_reduction(self, losses, loss_indices, *args):
        a1, _ = loss_indices
        halfway = self.batch_size // 2
        obj_loss = torch.mean(losses[a1 < halfway])
        skt_loss = torch.mean(losses[a1 >= halfway])
        return (obj_loss + skt_loss) / 2

def gen_labels(batch):
    # gallery_ids = batch['gallery_ids']
    query_ids = batch['query_ids']

    unique_queries = set(query_ids)
    query_id_dict = {query: i for i, query in enumerate(unique_queries)}

    label_objs = []
    label_queries = []
    for i in range(len(query_ids)):
        label_queries.append(query_id_dict[query_ids[i]])
        label_objs.append(query_id_dict[query_ids[i]])
    
    return torch.cat([torch.Tensor(label_objs), torch.Tensor(label_queries)])

def train_loop(obj_embedder, query_embedder, dl, obj_input, query_input, cbm_query, cbm_object, obj_optimizer, query_optimizer, device, use_cross_batch_mem=False):
    start_time = time.time()
    loss_avg = 0
    progress_bar = tqdm(enumerate(dl), total=len(dl))
    obj_embedder.train()
    query_embedder.train()
    for step, batch in progress_bar:

        obj_emb = obj_embedder(batch[obj_input].to(device))
        query_emb = query_embedder(batch[query_input].to(device))
        emb = torch.cat([obj_emb, query_emb])
        emb_len = emb.shape[0] # batch_size * 2 (text + object). If last batch is smaller, this will be smaller than batch_size * 2

        # labels = torch.cat([torch.arange(emb_len//2), torch.arange(emb_len//2)])
        labels = gen_labels(batch)

        if step > 2 and use_cross_batch_mem:
            enqueue_idx = torch.arange(emb_len//2, emb_len)
            # enqueue_mask: A boolean tensor where enqueue_mask[i] is True
            enqueue_mask = torch.zeros(emb_len, dtype=torch.bool)
            enqueue_mask[enqueue_idx] = True
            # enqueue_mask = enqueue_mask[:min(enqueue_mask.shape[0], emb.shape[0]),]
            loss1 = cbm_object(emb, labels, enqueue_mask=enqueue_mask)

            # This is the text-to-video loss
            enqueue_idx = torch.arange(emb_len//2, emb_len)

            enqueue_mask = torch.zeros(emb_len, dtype=torch.bool)
            enqueue_mask[enqueue_idx] = True
            # trim mask

            loss2 = cbm_query(emb, labels, enqueue_mask=enqueue_mask)
            loss = (loss1 + loss2) / 2
        else:
            contra_loss = NTXentLoss(reducer=PerModality(emb_len))
            loss = contra_loss(emb, labels)

        obj_optimizer.zero_grad()
        query_optimizer.zero_grad()
        loss.backward()
        obj_optimizer.step()
        query_optimizer.step()
        loss_avg += loss.item()
        progress_bar.set_description(f"Step: {step}/{len(dl)}, Loss: {loss.item():.4f}, Elapsed: {time.time() - start_time:.4f}")
    return loss_avg / len(dl)


def val_loop(obj_embedder, query_embedder, dl, obj_input, query_input, cbm_query, cbm_object, device, use_cross_batch_mem=False):
    start_time = time.time()
    loss_avg = 0
    progress_bar = tqdm(enumerate(dl), total=len(dl))
    obj_embedder.eval()
    query_embedder.eval()

    for step, batch in progress_bar:

        obj_emb = obj_embedder(batch[obj_input].to(device))
        query_emb = query_embedder(batch[query_input].to(device))
        emb = torch.cat([obj_emb, query_emb])
        emb_len = emb.shape[0] # batch_size * 2 (text + object). If last batch is smaller, this will be smaller than batch_size * 2

        # labels = torch.cat([torch.arange(emb_len//2), torch.arange(emb_len//2)])
        labels = gen_labels(batch)

        if step > 2 and use_cross_batch_mem:
            enqueue_idx = torch.arange(emb_len//2, emb_len)
            # enqueue_mask: A boolean tensor where enqueue_mask[i] is True
            enqueue_mask = torch.zeros(emb_len, dtype=torch.bool)
            enqueue_mask[enqueue_idx] = True
            # enqueue_mask = enqueue_mask[:min(enqueue_mask.shape[0], emb.shape[0]),]
            loss1 = cbm_object(emb, labels, enqueue_mask=enqueue_mask)

            # This is the text-to-video loss
            enqueue_idx = torch.arange(emb_len//2, emb_len)

            enqueue_mask = torch.zeros(emb_len, dtype=torch.bool)
            enqueue_mask[enqueue_idx] = True
            # trim mask

            loss2 = cbm_query(emb, labels, enqueue_mask=enqueue_mask)
            loss = (loss1 + loss2) / 2
        else:
            contra_loss = NTXentLoss(reducer=PerModality(emb_len))
            loss = contra_loss(emb, labels)

        loss_avg += loss.item()
        progress_bar.set_description(f"Step: {step}/{len(dl)}, Loss: {loss.item():.4f}, Elapsed: {time.time() - start_time:.4f}")
    return loss_avg / len(dl)