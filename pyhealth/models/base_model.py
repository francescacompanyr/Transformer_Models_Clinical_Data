from abc import ABC
from typing import List, Dict, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.decomposition import PCA
from tokenizer import Tokenizer

VALID_MODE = ["binary", "multiclass", "multilabel", "regression"]


class BaseModel(ABC, nn.Module):
    def __init__( self, dataset, feature_keys: List[str], label_key: str, mode: Optional[str] = None,   dic_group: dict = None,):
        super(BaseModel, self).__init__()
    
        self.dic_grup = dic_group
        self.dataset = dataset
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.mode = mode

        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))
        return

    @property
    def device(self):
        return self._dummy_param.device

    def get_feature_tokenizers(self) -> Dict[str, Tokenizer]:
        
        feature_tokenizers = {}
        for feature_key in self.feature_keys:
            print(feature_key)
            feature_tokenizers[feature_key] = Tokenizer(tokens=self.dataset.get_all_tokens(key=feature_key), )
        
        return feature_tokenizers

    @staticmethod
    def get_embedding_layers( feature_tokenizers: Dict[str, Tokenizer], embedding_dim: int,   ) :
       
        embedding_layers = nn.ModuleDict()
        for key, tokenizer in feature_tokenizers.items():
            embedding_layers[key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                embedding_dim,)
            
        return embedding_layers

    @staticmethod
    def padding2d(batch):
        batch_max_length = max([len(x) for x in batch])
        # get mask
        mask = torch.zeros(len(batch), batch_max_length, dtype=torch.bool)
        for i, x in enumerate(batch):
            mask[i, : len(x)] = 1
        # level-2 padding
        batch = [x + [[0.0] * len(x[0])] * (batch_max_length - len(x)) for x in batch]

        return batch, mask

    @staticmethod
    def padding3d(batch):

        batch_max_length_level2 = max([len(x) for x in batch])
        batch_max_length_level3 = max(
            [max([len(x) for x in visits]) for visits in batch] )
        # the most inner vector length
        vec_len = len(batch[0][0][0])
        # get mask
        mask = torch.zeros(   len(batch),   batch_max_length_level2,   batch_max_length_level3,   dtype=torch.bool, )
        for i, visits in enumerate(batch):
            for j, x in enumerate(visits):
                mask[i, j, : len(x)] = 1

        # level-2 padding
        batch = [ x + [[[0.0] * vec_len]] * (batch_max_length_level2 - len(x)) for x in batch ]
        # level-3 padding
        batch = [ [x + [[0.0] * vec_len] * (batch_max_length_level3 - len(x)) for x in visits] for visits in batch  ]
        return batch, mask

    def add_feature_transform_layer(self, feature_key: str, info):
        if info["type"] == str:        
            tokenizer = Tokenizer( tokens=self.dataset.get_all_tokens(key=feature_key),  )
            self.feat_tokenizers[feature_key] = tokenizer
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,)
        elif info["type"] in [float, int]:
            self.linear_layers[feature_key] = nn.Linear(info["len"], self.embedding_dim)
        else:
            raise ValueError("Unsupported feature type: {}".format(info["type"]))

    def add_feature_transform_dataset(self, info):
        all_tokens = []
        for feature_key, dic_info in info.items():
             if dic_info["type"] == str:
                tokens = self.dataset.get_all_tokens(key=feature_key)
                all_tokens.extend(tokens)
        
        tokenizer = Tokenizer(tokens=all_tokens)
        self.tokenizer = tokenizer  
        for feature_key, dic_info in info.items():
            if dic_info["type"] == str:
                self.feat_tokenizers[feature_key] = tokenizer
                self.embeddings[feature_key] = nn.Embedding(
                    tokenizer.get_vocabulary_size(), self.embedding_dim,)

            elif dic_info["type"] in [float, int]:
                self.linear_layers[feature_key] = nn.Linear(dic_info["len"], self.embedding_dim)
            else:
                raise ValueError("Unsupported feature type: {}".format(dic_info["type"]))

    def add_feature_transform_grup(self, info):
        print(info)
        group_tokenizers = {}
        
        for group_name, feature_list in self.dic_group.items():
            group_tokens = []
            for feature_key in feature_list:
                print(feature_key)
                if feature_key in info and info[feature_key]["type"] == str:
                    tokens = self.dataset.get_all_tokens(key=feature_key)
                    group_tokens.extend(tokens)

            if group_tokens:
                print('entraaa')
                print(group_tokens, group_name)
                group_tokenizers[group_name] = Tokenizer(tokens=group_tokens)
        
        for feature_key, dic_info in info.items():
            feature_group = None
            for group_name, feature_list in self.dic_group.items():
                if feature_key in feature_list:
                    feature_group = group_name
                    break
            
            if dic_info["type"] == str:
                # Usar el tokenizer del grupo correspondiente
                if feature_group and feature_group in group_tokenizers:
                    tokenizer = group_tokenizers[feature_group]
                    self.feat_tokenizers[feature_key] = tokenizer
                    self.embeddings[feature_key] = nn.Embedding(
                        tokenizer.get_vocabulary_size(), self.embedding_dim
                    )
                else:
                    # Fallback: crear tokenizer individual si no pertenece a ningún grupo
                    tokens = self.dataset.get_all_tokens(key=feature_key)
                    tokenizer = Tokenizer(tokens=tokens)
                    self.feat_tokenizers[feature_key] = tokenizer
                    self.embeddings[feature_key] = nn.Embedding(
                        tokenizer.get_vocabulary_size(), self.embedding_dim
                    )
            
            elif dic_info["type"] in [float, int]:
                self.linear_layers[feature_key] = nn.Linear(dic_info["len"], self.embedding_dim)
            
            else:
                raise ValueError("Unsupported feature type: {}".format(dic_info["type"]))
    

            


    def get_label_tokenizer(self) -> Tokenizer:

        label_tokenizer = Tokenizer(
            self.dataset.get_all_tokens(key=self.label_key), )
        return label_tokenizer

    def get_output_size(self, label_tokenizer: Tokenizer) -> int:
        output_size = label_tokenizer.get_vocabulary_size()
        if self.mode == "binary":
            assert output_size == 2
            output_size = 1
        return output_size

    def get_loss_function(self) -> Callable:
        if self.mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "multiclass":
            return F.cross_entropy
        elif self.mode == "regression":
            return F.mse_loss
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def prepare_labels(self, labels: Union[List[str], List[List[str]]],label_tokenizer: Tokenizer, ) -> torch.Tensor:

        if self.mode in ["binary"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.FloatTensor(labels).unsqueeze(-1)
        elif self.mode in ["multiclass"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.LongTensor(labels)
        else:
            raise NotImplementedError
        labels = labels.to(self.device)
        return labels

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        if self.mode in ["binary"]:
            y_prob = torch.sigmoid(logits)
        elif self.mode in ["multiclass"]:
            y_prob = F.softmax(logits, dim=-1)
        elif self.mode in ["multilabel"]:
            y_prob = torch.sigmoid(logits)
        else:
            raise NotImplementedError
        return y_prob
