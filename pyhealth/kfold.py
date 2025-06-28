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


def collate_fn_dict(batch:List):
        return {key: [d[key] for d in batch] for key in batch[0].keys()}

def run_kfold_training(k_folds=5, nom_execuacio ='base'):
    # Carregar dataset
    from datasets import load_dataset
    ds = load_dataset("Bena345/cdc-diabetes-health-indicators")
    df = ds['train'].to_pandas()    
    df_test = ds['test'].to_pandas()
    df_positive = df[df['Diabetes_binary'] == 'Diabetic']
    df_negative = df[df['Diabetes_binary'] == 'Non-Diabetic']
    min_len = min((len(df_positive)), len(df_negative))
    print(len(df_positive), len(df_negative),'\n\n')

    
    df_balanced = pd.concat([
        df_positive.sample(min_len, random_state=42),
        df_negative.sample(min_len*1, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)  # Barreja les files

    data = BaseDataset(path_df=df_balanced, label_key="Diabetes_binary", contextual=info_runing["contextual"])
    data_test = BaseDataset(path_df=df_test, label_key="Diabetes_binary", contextual=info_runing["contextual"])
    data_test, _ = train_test_split(data_test, test_size=0.88 ,random_state=42)
    df_full = data.get_df()
    labels = df_full['Diabetes_binary']
    labels = [x[0] for x in labels]


    dic_group = {
                "personals": ["Income", "Education", "Age", "Sex"],
                "metriques": ["PhysActivity", "PhysHlth", "MentHlth", "GenHlth","Smoker", "HvyAlcoholConsump"],
                "analitics": ["HighBP", "HighChol", "Stroke", "BMI"],
            } # separacio 1

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_metrics = {
        'train_loss': [], 'train_auc_0': [], 'train_auc_1': [], 'train_f1': [],
        'train_recall_0': [] , 'train_recall_1': [], 'val_recall_0': [] , 'val_recall_1': [],
        'val_loss': [], 'val_auc_0': [], 'val_auc_1': [],  'val_f1': []
    }
    
    # Iterar per cada fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        print(f"\n=== FOLD {fold + 1}/{k_folds} ===")
        
        train_ds = torch.utils.data.Subset(data, train_idx)
        val_ds = torch.utils.data.Subset(data, val_idx)
        
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}")
        print(f"Train pos ratio: {sum(train_labels)/len(train_labels):.3f}")
        print(f"Val pos ratio: {sum(val_labels)/len(val_labels):.3f}")
        
        # Crear DataLoaders
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
        
        # Crear model nou per cada fold
        if info_runing["separat"] == 0:
            model = Transformer(
                dataset=data, 
                feature_keys=["Income", "Education", "BMI", "Age", "HighBP", "HighChol", "CholCheck",
                             "Stroke", "Smoker", "Veggies", "PhysHlth", "HvyAlcoholConsump",
                             "GenHlth", "MentHlth", "Sex", "Fruits", "PhysActivity"],
                label_key="Diabetes_binary",dic_group=dic_group,
                mode="binary",
                embedding_dim=info_runing["emb_dim"], 
                from_categorical_pretrained=info_runing["pretrained"], 
                corpus_tokenizer=info_runing["corpus tokenizer"]
            )
        elif info_runing["separat"] == 1:
                 model = TransformerSeparated(dataset=data, 
                            feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol", 
                                        "Stroke", "Smoker", "PhysHlth", "HvyAlcoholConsump",
                                        "GenHlth", "MentHlth", "Sex", "PhysActivity"],
                            label_key="Diabetes_binary", dic_group=dic_group,
                            mode="binary",
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
            

        info_runing_fold = copy.deepcopy(info_runing)
        info_runing_fold["notes"] = f"{info_runing.get('notes', '')} - Fold {fold+1}"

        monitor = "loss"
        monitor_criterion = "min"

        
        trainer = Trainer(
            model=model, 
            exp_name=f'{nom_execuacio}_{fold+1}', # 
            info_wandb=info_runing_fold
        )
        test_loader =  DataLoader(
            dataset=data_test, 
            batch_size=1000,
            shuffle=True,
            collate_fn=collate_fn_dict
        )
        # Entrenar
        optimizer_params = {"lr": 1e-4}
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader = test_loader,
            epochs=75, # Reduït per k-fold
            optimizer_params=optimizer_params,
            labels=train_labels,
            label_key="Diabetes_binary",
            monitor=monitor,
            monitor_criterion="min"
        )
        
        # Guardar resultats d'aquest fold
        fold_results.append(results)
        
        # Avaluar al final del fold
        
        final_train_scores = trainer.evaluate(train_loader, labels=train_labels, label_key="Diabetes_binary")
        final_val_scores = trainer.evaluate(val_loader, labels=val_labels, label_key="Diabetes_binary")
        print(final_train_scores)

        # Guardar mètriques finals
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
        
    # Calcular estadístiques finals
    print(f"\n{'='*50}")
    print(f"K-FOLD CROSS VALIDATION RESULTS ({k_folds} folds)")
    print(f"{'='*50}")
    
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Log resultats finals a wandb
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
    
    # Log mitjanes i desviacions
    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        summary_metrics[f"{metric_name}_mean"] = np.mean(values)
        summary_metrics[f"{metric_name}_std"] = np.std(values)
    
    final_run.log(summary_metrics)
    final_run.finish()
    
    return all_metrics, fold_results


if __name__ == "__main__":
    info_runing = {
        "separat": 2,
        "emb_dim": 512,
        "pretrained": False,
        "corpus tokenizer": 2,
        "nom dataset": "Diabetes",
        "notes": "K-Fold Cross Validation",
        "contextual":False
    }

    #### treu info abans de sa nova confihuracio!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    nom_execuacio = "ARQgru-TOKENgrup"
    
    metrics, results = run_kfold_training(k_folds=5, nom_execuacio = nom_execuacio)