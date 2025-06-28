import pandas as pd
from typing import *
import numpy as np
from itertools import chain
import random
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.transformer import Transformer, TransformerSeparated, TransformerGroup
from dataset.splitter import split, split_v1
from trainer import Trainer
from models.base_model import BaseModel
import sys
import wandb
from torch.utils.data import Sampler
import numpy as np

wandb.login(key="7ce219a15db491eb84fa4b5e5da1a6650611d988")

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


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    import os 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    exp_name = 'BORRAR'
    info_runing ={"separat": 0, 'pretrained': False, "corpus tokenizer":True, "notes":'Sampler Ratios Disablebert', 
    "emb_dim": 256, "nom dataset": 'Diabetes', "kfold":True, "contextual":True }
    
    from datasets import load_dataset

    ds = load_dataset("Bena345/cdc-diabetes-health-indicators")

    df = ds['train'].to_pandas()

    from dataset.radiolung import BaseDataset

    data = BaseDataset( path_df = df, label_key="Diabetes_binary", contextual=info_runing["contextual"])

    dic_group = {
            "personals": ["Income", "Education", "Age", "Sex"],
            "metriques": ["PhysActivity", "PhysHlth", "MentHlth", "GenHlth","Smoker", "HvyAlcoholConsump"],
            "analitics": ["HighBP", "HighChol", "Stroke", "BMI"],
        }

    if info_runing["separat"] == 0:
    
        model = Transformer(dataset=data, 
                            feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol", "CholCheck",
                                        "Stroke", "Smoker", "Veggies", "PhysHlth", "HvyAlcoholConsump",
                                        "GenHlth", "MentHlth", "Sex",
                                        "Fruits", "PhysActivity"],
                            mode="binary", dic_group=dic_group,
                            label_key="Diabetes_binary", 
                            embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

    elif info_runing["separat"] == 1:
        model = TransformerSeparated(dataset=data, 
                            feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol", 
                                        "Stroke", "Smoker", "PhysHlth", "HvyAlcoholConsump",
                                        "GenHlth", "MentHlth", "Sex", "PhysActivity"],
                            mode="binary", dic_group=dic_group,
                            label_key="Diabetes_binary", 
                            embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

    else:
        dic_group = {
            "personals": ["Income", "Education", "Age", "Sex"],
            "metriques": ["PhysActivity", "PhysHlth", "MentHlth", "GenHlth","Smoker", "HvyAlcoholConsump"],
            "analitics": ["HighBP", "HighChol", "Stroke", "BMI"],
        } # separacio 1

        dic_group = {
        "personals": ["Income", "Education", "Age", "Sex"],
        "habits": ["PhysActivity", "Smoker", "HvyAlcoholConsump"],
        "self_reported_health": ["PhysHlth", "MentHlth", "GenHlth"],
        "clinics": ["HighBP", "HighChol", "Stroke"],
        "measures": ["BMI"],
    }
        dic_group = {
            "personals": ["Income", "Education", "Age", "Sex"],
            "metriques": ["PhysActivity", "PhysHlth", "MentHlth", "GenHlth","Smoker", "HvyAlcoholConsump"],
            "analitics": ["HighBP", "HighChol", "Stroke", "BMI"],
        } # separacio 1



        model = TransformerGroup(dataset=data, 
                            feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol", 
                                        "Stroke", "Smoker", "PhysHlth", "HvyAlcoholConsump",
                                        "GenHlth", "MentHlth", "Sex", "PhysActivity"],
                            label_key="Diabetes_binary",
                            mode="binary", dic_group=dic_group,
                            embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

         
    train_ds, val_ds, test_ds,  train_labels, val_labels, test_labels = split_v1(data, (0.7, 0.3,0), None ,'Diabetes_binary')
    #train_loader = DataLoader(data, batch_size=128, shuffle=True,  collate_fn=collate_fn_dict,)
    #val_loader = DataLoader(val_ds, batch_size=128, shuffle=True,  collate_fn=collate_fn_dict,)
    labels = df['Diabetes_binary']
    labels = [ x[0] for x in labels]


    batch_sampler_val = FixedRatioBatchSampler(
        labels=val_labels,
        batch_size=2000,
        pos_ratio=0.14
    )

    val_loader = DataLoader(
        dataset=val_ds,  batch_size=1000,
         num_workers=2,  collate_fn=collate_fn_dict,) #batch_sampler=batch_sampler_val,


    batch_sampler_train = FixedRatioBatchSampler( labels=train_labels,
        batch_size=2000,  pos_ratio=0.14)

    train_loader = DataLoader(
        dataset=train_ds,  batch_size=1000,
        num_workers =2,  collate_fn=collate_fn_dict,) #batch_sampler=batch_sampler_train, 

    optimizer_params = {"lr": 1e-4}

    df = data.get_df()

    labels = df['Diabetes_binary']
    labels = [ x[0] for x in labels]


    trainer = Trainer(model=model, exp_name=exp_name, info_wandb=info_runing  )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=2, optimizer_params=optimizer_params , labels = labels, label_key="Diabetes_binary")


    """ 
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    # Nombre de folds
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    from datasets import load_dataset
    ds = load_dataset("Bena345/cdc-diabetes-health-indicators")
    df = ds['train'].to_pandas()
    from dataset.radiolung import BaseDataset
    data = BaseDataset( path_df = df, label_key="Diabetes_binary")
    df = data.get_df()
    labels = df['Diabetes_binary']
    labels = [ x[0] for x in labels]

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n Fold {fold + 1}/{k}")

        train_ds = torch.utils.data.Subset(data, train_idx)
        val_ds = torch.utils.data.Subset(data, val_idx)
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        batch_sampler_train = FixedRatioBatchSampler(train_labels, batch_size=2000, pos_ratio=0.14)
        batch_sampler_val = FixedRatioBatchSampler(val_labels, batch_size=2000, pos_ratio=0.14)
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler_train, collate_fn=collate_fn_dict)
        val_loader = DataLoader(val_ds, batch_sampler=batch_sampler_val,  collate_fn=collate_fn_dict)

        
        if info_runing["separat"] == 0:
            model = Transformer(
                dataset=data,
                feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol", "CholCheck",
                            "Stroke", "Smoker", "Veggies", "PhysHlth", "HvyAlcoholConsump",
                            "GenHlth", "MentHlth", "Sex", "Fruits", "PhysActivity"],
                label_key="Diabetes_binary",
                mode="binary",
                embedding_dim=info_runing["emb_dim"],
                from_categorical_pretrained=info_runing["pretrained"],
                corpus_tokenizer=info_runing["corpus tokenizer"]
            )

        elif info_runing["separat"] == 1:
            model = TransformerSeparated(
                dataset=data,
                feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol",
                            "Stroke", "Smoker", "PhysHlth", "HvyAlcoholConsump",
                            "GenHlth", "MentHlth", "Sex", "PhysActivity"],
                label_key="Diabetes_binary",
                mode="binary",
                embedding_dim=info_runing["emb_dim"],
                from_categorical_pretrained=info_runing["pretrained"],
                corpus_tokenizer=info_runing["corpus tokenizer"]
            )

        else:
            dic_group = {
                "personals": ["Income", "Education", "Age", "Sex"],
                "metriques": ["PhysActivity", "PhysHlth", "MentHlth", "GenHlth", "Smoker", "HvyAlcoholConsump"],
                "analitics": ["HighBP", "HighChol", "Stroke", "BMI"]
            }

            model = TransformerGroup(
                dataset=data,
                feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol",
                            "Stroke", "Smoker", "PhysHlth", "HvyAlcoholConsump",
                            "GenHlth", "MentHlth", "Sex", "PhysActivity"],
                label_key="Diabetes_binary",
                mode="binary",
                dic_group=dic_group,
                embedding_dim=info_runing["emb_dim"],
                from_categorical_pretrained=info_runing["pretrained"],
                corpus_tokenizer=info_runing["corpus tokenizer"]
            )
        from collections import Counter
        print(f"Train labels: {Counter(train_labels)}, Val labels: {Counter(val_labels)}")
        # Entrenador
        trainer = Trainer(model=model, exp_name=f'fold_{fold+1}', info_wandb=info_runing)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=100,
            optimizer_params={"lr": 1e-3},
            labels=labels,
            label_key="Diabetes_binary"
        )
"""