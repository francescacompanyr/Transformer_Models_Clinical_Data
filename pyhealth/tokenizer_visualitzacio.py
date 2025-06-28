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
from datasets import load_dataset

ds = load_dataset("Bena345/cdc-diabetes-health-indicators")

df = ds['train'].to_pandas()

from dataset.radiolung import BaseDataset

data = BaseDataset( path_df = df, label_key="Diabetes_binary")

info_runing ={"separat": 2 ,'pretrained': False, "corpus tokenizer":True, "notes":'Sampler Ratios Disablebert', 
    "emb_dim": 256, "nom dataset": 'Diabetes', "kfold":True }

"""
  feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol", "CholCheck",
                                    "Stroke", "Smoker", "Veggies", "PhysHlth", "HvyAlcoholConsump",
                                    "GenHlth", "MentHlth", "Sex",
                                    "Fruits", "PhysActivity"],
"""

if info_runing["separat"] == 0:

    model = Transformer(dataset=data, 
                        feature_keys=[  "Education",  "Age", 
                
                                    "GenHlth"],
                        label_key="Diabetes_binary",
                        mode="binary",
                        embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

elif info_runing["separat"] == 1:
    model = TransformerSeparated(dataset=data, 
                        feature_keys=[ "Income", "Education", "BMI", "Age", "HighBP", "HighChol", 
                                    "Stroke", "Smoker", "PhysHlth", "HvyAlcoholConsump",
                                    "GenHlth", "MentHlth", "Sex", "PhysActivity"],
                        label_key="Diabetes_binary",
                        mode="binary",
                        embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

else:
    dic_group = {
        "personals": ["Income", "Education", "Age", "Sex"],
        "metriques": ["PhysActivity", "PhysHlth", "MentHlth", "GenHlth","Smoker", "HvyAlcoholConsump"],
        "analitics": ["HighBP", "HighChol", "Stroke", "BMI"],
    }

    model = TransformerGroup(dataset=data, 
                        feature_keys=[  "Education",  "Age", "HighBP", "HighChol", 
                                    "Stroke", "Smoker",  "HvyAlcoholConsump",
                                    "GenHlth", "MentHlth", "Sex", "PhysActivity"],
                        label_key="Diabetes_binary",
                        mode="binary", dic_group=dic_group,
                        embedding_dim=info_runing["emb_dim"], from_categorical_pretrained = info_runing["pretrained"], corpus_tokenizer = info_runing["corpus tokenizer"])

print(model)


feature_tokenizers = model.get_feature_tokenizers()

df = data.get_df()

print(f"Dataset shape: {df.shape}")
print(f"Features analitzades: {len(model.feature_keys)}")
print(f"Label key: {model.label_key}")
print()

token_analysis = {}
max_examples=15

for feature_key in model.feature_keys:
    
    print(f"\n{'='*20} FEATURE: {feature_key} {'='*20}")
    
    # Obtenir el tokenizer per aquesta feature
    tokenizer = feature_tokenizers[feature_key]
    
    # Obtenir tots els tokens possibles per aquesta feature
    all_possible_tokens = model.dataset.get_all_tokens(key=feature_key)
    
    print(f" Nombre total de tokens possibles: {len(all_possible_tokens)}")
    print(f"Tokens disponibles: {all_possible_tokens}")
    
    # Obtenir els valors reals de la columna
    column_values = df[feature_key].dropna().unique()
    print(f" Valors únics en el dataset: {len(column_values)}")
    print(f"💾 Tipus de dades: {df[feature_key].dtype}")
    # Mostrar alguns exemples de com es mapegen els valors als tokens
    print(f"\n🔄 MAPEJAT VALOR -> TOKEN:")
    print("-" * 40)
    
    examples_shown = 0
    token_to_value_map = {}
    value_to_token_map = {}
    
    for value in sorted(column_values):
        if examples_shown >= max_examples and len(column_values) > max_examples:
            print(f"... i {len(column_values) - max_examples} valors més")
            break
            
        try:
            if hasattr(model.dataset, 'value_to_token'):
                token = model.dataset.value_to_token(feature_key, value)
            else:
                token = str(value)
            
            if token in all_possible_tokens:
                token_id = tokenizer.token_to_id(token) if hasattr(tokenizer, 'token_to_id') else all_possible_tokens.index(token)
                print(f"  📈 {value:>15} -> '{token}' (ID: {token_id})")
                
                token_to_value_map[token] = value
                value_to_token_map[value] = token
            else:
                print(f"  {value:>15} -> TOKEN NO TROBAT!")
                
        except Exception as e:
            print(f" {value:>15} -> Error: {e}")
        
        examples_shown += 1