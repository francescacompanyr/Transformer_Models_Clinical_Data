from sklearn.model_selection import StratifiedKFold
import torch
import copy
import numpy as np
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
from dataset.radiolung import BaseDataset
from dataset.radiolung import RadioLungDataset


def collate_fn_dict(batch:List):
        return {key: [d[key] for d in batch] for key in batch[0].keys()}

def run_kfold_training(k_folds=5, nom_execuacio ='base'):
    # Path list
    path_list = [
        "/ghome/debora/Francesca/Code/tfg/pyhealth/11112024_BDMetaData_CanRuti.xlsx", 
        "/ghome/debora/Francesca/Code/tfg/pyhealth/11112024_BDMetaData_santpau.xlsx", 
        "/ghome/debora/Francesca/Code/tfg/pyhealth/28102024_BDMetaData_delmar.xlsx", 
        "/ghome/debora/Francesca/Code/tfg/pyhealth/28102024_BDMetaData_mutuaterrassa.xlsx"
    ]

    
    data = RadioLungDataset(path_df=path_list, label_key='type')

    df = data.get_df()
    labels = df['type']
    labels = [x[0] for x in labels] 

    indices = np.arange(len(data))

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_metrics = {
        'train_loss': [], 'train_auc_0': [], 'train_auc_1': [], 'train_f1': [],
        'train_recall_0': [] , 'train_recall_1': [], 'val_recall_0': [] , 'val_recall_1': [],
        'val_loss': [], 'val_auc_0': [], 'val_auc_1': [],  'val_f1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        print(f"\n=== FOLD {fold + 1}/{k_folds} ===")
        
        train_ds = torch.utils.data.Subset(data, train_idx)
        val_ds = torch.utils.data.Subset(data, val_idx)
        
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}")
        print(f"Train pos ratio: {sum(train_labels)/len(train_labels):.3f}")
        print(f"Val pos ratio: {sum(val_labels)/len(val_labels):.3f}")
        
        train_loader = DataLoader(
            dataset=train_ds, 
            batch_size=1000,
            shuffle=True,
            collate_fn=collate_fn_dict
        )
        
        val_loader = DataLoader(
            dataset=val_ds, 
            batch_size=1000,
            collate_fn=collate_fn_dict
        )
        
        if info_runing["separat"] == 0:
            model = Transformer(dataset=data, 
                            feature_keys=["age", "sex", "BMI", "education",
                                "air_pollution", "smoking", "packyear_index", "fh_cancer",
                                "ph_cancer", "COPD", "FVC", "indice", "DLCO"],
                            label_key="type",
                            mode="binary",
                            embedding_dim=info_runing["emb_dim"], 
                            from_categorical_pretrained=info_runing["pretrained"], 
                            corpus_tokenizer=info_runing["corpus tokenizer"])

        elif info_runing["separat"] == 1:
            model = TransformerSeparated(dataset=data, 
                            feature_keys=["age", "sex", "BMI", "education",
                                "air_pollution", "smoking", "packyear_index", "fh_cancer",
                                "ph_cancer", "COPD", "FVC", "indice", "DLCO"],
                            label_key="type",
                            mode="binary",
                            embedding_dim=info_runing["emb_dim"], 
                            from_categorical_pretrained=info_runing["pretrained"], 
                            corpus_tokenizer=info_runing["corpus tokenizer"])
        else:
            dic_group = {
            "personal": ['age','sex', 'education', 'smoking'],
            "condition": ['air_pollution', 'ph_cancer', 'fh_cancer'],
            "analytics": ['BMI','packyear_index', 'COPD','FVC','indice','DLCO'],
              }
            model = TransformerGroup(dataset=data, 
                            feature_keys=["age", "sex", "BMI", "education",
                                "air_pollution", "smoking", "packyear_index", "fh_cancer",
                                "ph_cancer", "COPD", "FVC", "indice", "DLCO"],
                            label_key="type",
                            mode="binary", dic_group=dic_group,
                            embedding_dim=info_runing["emb_dim"], 
                            from_categorical_pretrained=info_runing["pretrained"], 
                            corpus_tokenizer=info_runing["corpus tokenizer"])
    
        info_runing_fold = copy.deepcopy(info_runing)
        info_runing_fold["notes"] = f"{info_runing.get('notes', '')} - Fold {fold+1}"

        monitor = "loss"
        monitor_criterion = "min"

        
        trainer = Trainer(
            model=model, 
            exp_name=f'{nom_execuacio}_{fold+1}', # 
            info_wandb=info_runing_fold
        )
        """       test_loader =  DataLoader(
            dataset=data_test, 
            batch_size=1000,
            shuffle=True,
            collate_fn=collate_fn_dict
        )
        """
        optimizer_params = {"lr": 1e-3}
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader = None,
            epochs=75, 
            optimizer_params=optimizer_params,
            labels=train_labels,
            label_key="type",
            monitor=monitor,
            monitor_criterion="min"
        )
        
        fold_results.append(results)
        
        
        final_train_scores = trainer.evaluate(train_loader, labels=train_labels, label_key="type")
        final_val_scores = trainer.evaluate(val_loader, labels=val_labels, label_key="type")
        print(final_train_scores)

        all_metrics['train_loss'].append(final_train_scores['loss'])
        all_metrics['train_auc_0'].append(final_train_scores['auc0'])
        all_metrics['train_auc_1'].append(final_train_scores['auc1'])
        all_metrics['train_f1'].append(final_train_scores['f1'])
        all_metrics['train_recall_0'].append(final_train_scores['recall0'])
        all_metrics['train_recall_1'].append(final_train_scores['recall1'])     
        all_metrics['val_loss'].append(final_val_scores['loss'])
        all_metrics['val_auc_0'].append(final_val_scores['auc0'])
        all_metrics['val_auc_1'].append(final_val_scores['auc1'])
        all_metrics['val_f1'].append(final_val_scores['f1'])
        all_metrics['val_recall_0'].append(final_val_scores['recall0'])
        all_metrics['val_recall_1'].append(final_val_scores['recall1'])      
        print(f"\nFold {fold+1} Results:")
        print(f"Train AUC: {final_train_scores['auc1']:.4f}")
        print(f"Val AUC: {final_val_scores['auc1']:.4f}")
        
    print(f"\n{'='*50}")
    print(f"K-FOLD CROSS VALIDATION RESULTS ({k_folds} folds)")
    print(f"{'='*50}")
    
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    import wandb
    final_run = wandb.init(
        name=f'kfold_summary_{nom_execuacio}',
        entity="tfg_pyh", 
        project="tfg_summary",
        config={
            "k_folds": k_folds,
            "dataset": info_runing['nom dataset'],
            "final_summary": True
        }
    )
    
    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        summary_metrics[f"{metric_name}_mean"] = np.mean(values)
        summary_metrics[f"{metric_name}_std"] = np.std(values)
    
    final_run.log(summary_metrics)
    final_run.finish()
    
    return all_metrics, fold_results


if __name__ == "__main__":
    info_runing = {
        "separat": 0,
        "emb_dim": 512,
        "pretrained": False,
        "corpus tokenizer": 0,
        "nom dataset": "RadioLung",
        "notes": "K-Fold Cross Validation"
    }

    nom_execuacio = "testfuncionarl"
    
    metrics, results = run_kfold_training(k_folds=5, nom_execuacio = nom_execuacio)