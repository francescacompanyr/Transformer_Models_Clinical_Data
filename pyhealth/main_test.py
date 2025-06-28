import pandas as pd
from typing import *
import numpy as np
from itertools import chain
import random
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset.radiolung import RadioLungDataset
from dataset.splitter import split_v1
from trainer import Trainer

import sys
import pandas as pd

from models.transformer import Transformer, TransformerSeparated, TransformerGroup
from dataset.splitter import split
from trainer import Trainer
from models.base_model import BaseModel
import sys
import wandb
from torch.utils.data import Sampler
import numpy as np
#_, nom_execucio, merged = sys.argv

class FixedRatioBatchSampler(Sampler):
        def __init__(self, labels, batch_size=100, pos_ratio=0.14):
            self.labels = np.array(labels)
            self.batch_size = batch_size
            self.pos_ratio = pos_ratio
            self.n_pos = int(round(batch_size * pos_ratio))
            self.n_neg = batch_size - self.n_pos

            self.pos_indices = np.where(self.labels == 1)[0]
            self.neg_indices = np.where(self.labels == 0)[0]

            self.num_batches = min(
                len(self.pos_indices) // self.n_pos,
                len(self.neg_indices) // self.n_neg
            )

        def __iter__(self):
            pos = np.random.permutation(self.pos_indices)
            neg = np.random.permutation(self.neg_indices)

            for i in range(self.num_batches):
                pos_batch = pos[i*self.n_pos:(i+1)*self.n_pos]
                neg_batch = neg[i*self.n_neg:(i+1)*self.n_neg]
                batch = np.concatenate([pos_batch, neg_batch])
                np.random.shuffle(batch)
                yield batch.tolist()

        def __len__(self):
            return self.num_batches


def collate_fn_dict(batch:List):
    return {key: [d[key] for d in batch] for key in batch[0].keys()}
path_list = ["/ghome/debora/Francesca/Code/tfg/pyhealth/11112024_BDMetaData_CanRuti.xlsx", "/ghome/debora/Francesca/Code/tfg/pyhealth/11112024_BDMetaData_santpau.xlsx", "/ghome/debora/Francesca/Code/tfg/pyhealth/28102024_BDMetaData_delmar.xlsx", "/ghome/debora/Francesca/Code/tfg/pyhealth/28102024_BDMetaData_mutuaterrassa.xlsx"]



if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    import os 


    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    
    info_runing ={"separat": 1, 'pretrained': False, "corpus tokenizer":False, "notes":'Sampler Ratios Disablebert', "emb_dim": 256, "nom dataset": 'RadioLung' }

 

    
    data = RadioLungDataset(path_df=path_list, label_key = 'type'  )

    if info_runing["separat"] == 0:
    
        model = Transformer(dataset=data, 
                            feature_keys=[ "age", "sex", "BMI", "education",
                                "air_pollution", "smoking", "packyear_index", "fh_cancer",
                                "ph_cancer", "COPD", "FVC", "indice", "DLCO"],
                            label_key="type",
                            mode="binary",
                            embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

    elif info_runing["separat"] == 1:
        model = TransformerSeparated(dataset=data, 
                            feature_keys=[ "age", "sex", "BMI", "education",
                                "air_pollution", "smoking", "packyear_index", "fh_cancer",
                                "ph_cancer", "COPD", "FVC", "indice", "DLCO"],
                            label_key="type",
                            mode="binary",
                            embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

    else:
        dic_group = {
            "": [],
            "": [],
            "": [],
        }

        model = TransformerGroup(dataset=data, 
                            feature_keys=[ "age", "sex", "BMI", "education",
                                "air_pollution", "smoking", "packyear_index", "fh_cancer",
                                "ph_cancer", "COPD", "FVC", "indice", "DLCO"],
                            label_key="type",
                            mode="binary", dic_group=dic_group,
                            embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

         
    train_ds, val_ds, test_ds,  train_labels, val_labels, test_labels = split_v1(data, (0.7, 0.3,0),None, "type")
    #train_loader = DataLoader(data, batch_size=128, shuffle=True,  collate_fn=collate_fn_dict,)
    #val_loader = DataLoader(val_ds, batch_size=128, shuffle=True,  collate_fn=collate_fn_dict,)
  


    batch_sampler_val = FixedRatioBatchSampler(
        labels=val_labels,
        batch_size=1000,
        pos_ratio=0.14
    )

    val_loader = DataLoader(batch_size=250,
        dataset=val_ds,
        num_workers=2,  collate_fn=collate_fn_dict,) #batch_sampler=batch_sampler_val,


    batch_sampler_train = FixedRatioBatchSampler( labels=train_labels,
        batch_size=1000,  pos_ratio=0.14)

    train_loader = DataLoader( batch_size=250,
        dataset=train_ds,num_workers =2,  collate_fn=collate_fn_dict,) #batch_sampler=batch_sampler_train, 

    optimizer_params = {"lr": 1e-3}

    df = data.get_df()

    labels = df['type']
    labels = [ x[0] for x in labels]


    trainer = Trainer(model=model, exp_name= 'RADIOLUNG-separat', info_wandb=info_runing  )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=100, optimizer_params=optimizer_params , labels = labels, label_key="type")