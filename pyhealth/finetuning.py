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
import os


def collate_fn_dict(batch:List):
        return {key: [d[key] for d in batch] for key in batch[0].keys()}

def load_pretrained_model(model, checkpoint_path, strict=False, partial_load=False):

    if os.path.exists(checkpoint_path):
        print(f"Carregant model preentrenat des de: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Trobat checkpoint amb 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Trobat checkpoint amb 'state_dict'")
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                state_dict = checkpoint
                print("Utilitzant checkpoint com a state_dict directe")
            
            if state_dict is None:
                print("No s'ha pogut extreure state_dict del checkpoint")
                return model
            
            if not partial_load:
                try:
                    model.load_state_dict(state_dict, strict=strict)
                    print("Model preentrenat carregat correctament (mode estricte)!")
                except Exception as e:
                    if strict:
                        print(f" ERROR en mode estricte: {e}")
                        print("des de zero.")
                        return model
                    else:
                        print(f"Mode estricte fallit, intentant càrrega parcial...")
                        partial_load = True
            
            if partial_load:
                model_dict = model.state_dict()
                
                compatible_dict = {}
                incompatible_keys = []
                
                for key, value in state_dict.items():
                    if key in model_dict:
                        if model_dict[key].shape == value.shape:
                            compatible_dict[key] = value
                        else:
                            incompatible_keys.append(f"{key}: {value.shape} vs {model_dict[key].shape}")
                    else:
                        incompatible_keys.append(f"{key}: no existeix al model actual")
                
                total_checkpoint_keys = len(state_dict)
                total_model_keys = len(model_dict)
                loaded_keys = len(compatible_dict)
                
                print(f" Estadístiques de càrrega:")
                print(f"   Claus al checkpoint: {total_checkpoint_keys}")
                print(f"   Claus al model actual: {total_model_keys}")
                print(f"   Claus carregades: {loaded_keys}")
                print(f"   % carregat: {loaded_keys/total_model_keys*100:.1f}%")
                
                if loaded_keys > 0:
                    model_dict.update(compatible_dict)
                    model.load_state_dict(model_dict)
                    print(" Càrrega parcial completada!")
                    
                    if loaded_keys > 0:
                        print(f"   Algunes claus carregades: {list(compatible_dict.keys())}...")
                    if incompatible_keys:
                        print(f"   Algunes claus incompatibles: {incompatible_keys}...")
                else:
                    print(" Cap clau compatible trobada")
                    print("S'entrenarà des de zero.")
                    return model
            
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f"   Epoch del checkpoint: {checkpoint['epoch']}")
                if 'best_val_loss' in checkpoint:
                    print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
                if 'best_val_auc' in checkpoint:
                    print(f"   Best val AUC: {checkpoint['best_val_auc']:.4f}")
                    
        except Exception as e:
            print(f"ERROR carregant el checkpoint: {e}")
            print("S'entrenarà des de zero.")
            return model
            
    else:
        print(f"ADVERTÈNCIA: No s'ha trobat el checkpoint a {checkpoint_path}")
        print("S'entrenarà des de zero.")
    
    return model

def run_kfold_training(k_folds=5, nom_execuacio='base', finetune_config=None):

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
    
    is_finetuning = finetune_config is not None and 'checkpoint_path' in finetune_config
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        print(f"\n=== FOLD {fold + 1}/{k_folds} ===")
        if is_finetuning:
            print(f"Mode: FINE-TUNING des de {finetune_config['checkpoint_path']}")
        else:
            print("Mode: TRAINING des de zero")
        
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
    
        if is_finetuning:
            model = load_pretrained_model(model, finetune_config['checkpoint_path'])
            
            if finetune_config.get('freeze_embeddings', False):
                print("Congelant capes d'embedding...")
                for name, param in model.named_parameters():
                    if 'embedding' in name.lower() or 'embed' in name.lower():
                        param.requires_grad = False
                        print(f"  Congelat: {name}")
        
        info_runing_fold = copy.deepcopy(info_runing)
        fold_suffix = f"Fold {fold+1}"
        if is_finetuning:
            fold_suffix += " (Fine-tuning)"
        info_runing_fold["notes"] = f"{info_runing.get('notes', '')} - {fold_suffix}"

        monitor = "loss"
        monitor_criterion = "min"
        
        trainer = Trainer(
            model=model, 
            exp_name=f'{nom_execuacio}_{fold+1}', 
            info_wandb=info_runing_fold
        )
        
        if is_finetuning and 'learning_rate' in finetune_config:
            optimizer_params = {"lr": finetune_config['learning_rate']}
            print(f"Utilitzant learning rate per fine-tuning: {finetune_config['learning_rate']}")
        else:
            optimizer_params = {"lr": 1e-3}
        
        if is_finetuning and 'epochs' in finetune_config:
            epochs = finetune_config['epochs']
            print(f"Utilitzant {epochs} epochs per fine-tuning")
        else:
            epochs = 75
        
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=None,
            epochs=epochs,
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
    mode_str = "FINE-TUNING" if is_finetuning else "TRAINING"
    print(f"K-FOLD CROSS VALIDATION RESULTS ({k_folds} folds) - {mode_str}")
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
            "final_summary": True,
            "is_finetuning": is_finetuning,
            "finetune_config": finetune_config if is_finetuning else None
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
    
    print("=== TRAINING DES DE ZERO ===")
    metrics_scratch, results_scratch = run_kfold_training(
        k_folds=5, 
        nom_execuacio=f"{nom_execuacio}_scratch"
    )
    
    finetune_config = {
        'checkpoint_path': '/ghome/debora/Francesca/Results/output/ARQgru-TOKENgrup_1_20250619-234116/last.ckpt', 
        'freeze_embeddings': False,  
        'learning_rate': 1e-4,  
        'epochs': 75  
    }
    
    print("\n=== FINE-TUNING ===")
    metrics_finetune, results_finetune = run_kfold_training(
        k_folds=5, 
        nom_execuacio=f"{nom_execuacio}_finetune",
        finetune_config=finetune_config
    )
    
    print(f"\n{'='*60}")
    print("COMPARACIÓ DE RESULTATS")
    print(f"{'='*60}")
    print("TRAINING DES DE ZERO:")
    print(f"  Val AUC: {np.mean(metrics_scratch['val_auc_1']):.4f} ± {np.std(metrics_scratch['val_auc_1']):.4f}")
    print(f"  Val F1:  {np.mean(metrics_scratch['val_f1']):.4f} ± {np.std(metrics_scratch['val_f1']):.4f}")
    
    print("FINE-TUNING:")
    print(f"  Val AUC: {np.mean(metrics_finetune['val_auc_1']):.4f} ± {np.std(metrics_finetune['val_auc_1']):.4f}")
    print(f"  Val F1:  {np.mean(metrics_finetune['val_f1']):.4f} ± {np.std(metrics_finetune['val_f1']):.4f}")









