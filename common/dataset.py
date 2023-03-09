import os 
from PIL import Image
import pandas as pd 
import torch
from torchvision import transforms as tvtf
from torch.utils import data
from transformers import AutoTokenizer
from utils.text_preprocess import preprocess


class SHREC23_Test_SketchesData(data.Dataset):
    def __init__(self, skt_data_path, csv_data_path):
        self.csv_data = pd.read_csv(csv_data_path)
        self.ids = self.csv_data.index
        self.skt_data_path = skt_data_path
        self.render_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        skt_id = self.csv_data.iloc[idx]['ID']
        query_impath = os.path.join(self.skt_data_path, skt_id + '.jpg')
        query_im = Image.open(query_impath).convert('RGB')
        query_im = self.render_transforms(query_im)
        
        return {
            "query_im": query_im,
            "query_id": skt_id,
        }
    
    def collate_fn(self, batch):
        batch = {
            'query_ims': torch.stack([item['query_im'] for item in batch]),
            'query_ids': [item['query_id'] for item in batch],
        }
        return batch
    
class SHREC23_Test_TextData(data.Dataset):
    def __init__(self, csv_data_path):
        self.csv_data = pd.read_csv(csv_data_path,delimiter=';')
        self.ids = self.csv_data.index
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        txt_id = self.csv_data.iloc[idx]['ID']
        query_text = preprocess(self.csv_data.iloc[idx]['Description'])
        print(query_text)
        

        
        return {
            "query_text": query_text,
            "query_id": txt_id,
        }
    
    def collate_fn(self, batch):
        batch = {
            "query_texts": [x['query_text'] for x in batch],
            'query_ids': [item['query_id'] for item in batch],
        }
        batch["tokens"] = self.tokenizer.batch_encode_plus(
            batch["query_texts"], padding="longest", return_tensors="pt"
        )
        return batch