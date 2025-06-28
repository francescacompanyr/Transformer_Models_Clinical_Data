import math
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from models.base_model import BaseModel
from tokenizer import Tokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
 
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
    
        self.linear_layers = nn.ModuleList( [nn.Linear(d_model, d_model, bias=False) for _ in range(3)] )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

        self.attn_gradients = None
        self.attn_map = None

    # helper functions for interpretability
    def get_attn_map(self):
        return self.attn_map 
    
    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad 

    # register_hook option allows us to save the gradients in backwarding
    def forward(self, query, key, value, mask=None, register_hook = False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        self.attn_map = attn # save the attention map
        if register_hook:
            attn.register_hook(self.save_attn_grad)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
  
        return self.output_linear(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):

   #  MultiHeadedAttention  + SublayerConnection .  hidden: hidden size of transformer.    attn_heads: head sizes of multi-head attention.

    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, register_hook = False):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask, register_hook=register_hook))
        return self.dropout(x)


class TransformerLayer(nn.Module):
 
    def __init__(self, feature_size, heads=8, dropout=0.5, num_layers=2):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList([TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None, register_hook = False) -> Tuple[torch.tensor, torch.tensor]:
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask, register_hook)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class Transformer(BaseModel):

    def __init__(self, dataset, feature_keys: List[str], label_key: str, mode: str, embedding_dim: int = 128, output_size=2, dic_group=None, from_categorical_pretrained: bool = True, corpus_tokenizer = False,**kwargs,):
        super(Transformer, self).__init__( dataset=dataset,feature_keys=feature_keys, dic_group=dic_group, label_key=label_key, mode=mode,)
       
        self.corpus_tokenizer = corpus_tokenizer
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = {}   # the key of self.feat_tokenizers only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.label_key = label_key
        self.from_categorical_pretrained = from_categorical_pretrained
        self.dic_group = dic_group
        
       # manera crear el tokenizer
        if self.corpus_tokenizer == 1:  #tipologia tokenizer (categoric numeric)
            print(True,'catnum')
            self.add_feature_transform_dataset(self.dataset.input_info)
        elif self.corpus_tokenizer == 2: # categoria tokenizer (metrica anlitiques condicions)
            print('tokenizer agrupat')
            self.add_feature_transform_grup(self.dataset.input_info)
        
        else: # tokenizer per columna
          for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]  # input_info es un diccionari de diccionaris
            self.add_feature_transform_layer(feature_key, input_info) #dins aquesta funcio se passa es diccionari de feature i se fa es dict feat_tokenizer


        if from_categorical_pretrained:
            #self.pretrained_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
           # self.language_model =  AutoModel.from_pretrained("medicalai/ClinicalBERT")
            self.pretrained_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.language_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.language_model.to(device)
            self.pretrained_proj = nn.Linear(768, self.embedding_dim) ########################COMPROVAR QUE ESTIGUI BE AIXO     
        
        
        self.transformer = nn.ModuleDict()
        self.transformer["merged"] =  TransformerLayer( feature_size=embedding_dim, **kwargs)
        self.fc = nn.Linear(self.embedding_dim, output_size)




    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        pattient_mask = []
      
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]
            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                if self.from_categorical_pretrained:
                    batch_tokens = [x[0] for x in kwargs[feature_key]]
                    x = self.pretrained_tokenizer(batch_tokens, return_tensors="pt", padding=True, truncation=False)
                    x = x.to(device)
                    with torch.no_grad():
                        outputs = self.language_model(**x)
                        x = outputs.last_hidden_state[:,0,:]
                        x = self.pretrained_proj(x) 
                        x = x.unsqueeze(1)
                    mask = torch.any(x !=0, dim=2)
                        # ficar capa lineal fins que 128 #######################
                else:
                    x = self.feat_tokenizers[feature_key].batch_encode_2d(
                        kwargs[feature_key] )
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    x = self.embeddings[feature_key](x)
                    mask = torch.any(x !=0, dim=2)
                
                
            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key] )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                x = torch.sum(x, dim=2)
                mask = torch.any(x !=0, dim=2)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = self.linear_layers[feature_key](x)
                mask = mask.bool().to(self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            patient_emb.append(x)
            pattient_mask.append(mask)


        patient_emb = torch.cat(patient_emb, dim=1)
        patient_mask = torch.cat(pattient_mask, dim=1)
        _, x = self.transformer["merged"](patient_emb, patient_mask, kwargs.get('register_hook'))
        logits = self.fc(x)
        return logits

class TransformerSeparated(BaseModel):

    def __init__(self, dataset, feature_keys: List[str], label_key: str, mode: str, corpus_tokenizer: bool,dic_group: dict = None,
        embedding_dim: int = 128, output_size=2, merged = False,from_categorical_pretrained: bool = True, **kwargs,):
        super(TransformerSeparated, self).__init__(dataset=dataset,feature_keys=feature_keys,label_key=label_key, mode=mode, )
        

        self.embedding_dim = embedding_dim
        self.feat_tokenizers = {}   # the key of self.feat_tokenizers only contains the code based inputs
        self.label_key = label_key
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.from_categorical_pretrained = from_categorical_pretrained
        self.corpus_tokenizer = corpus_tokenizer
        self.dic_group = dic_group
        # variables pretrained

        if from_categorical_pretrained:
            # self.pretrained_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
           # self.language_model =  AutoModel.from_pretrained("medicalai/ClinicalBERT")
            self.pretrained_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.language_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.language_model.to(device)
            self.pretrained_proj = nn.Linear(768, self.embedding_dim) ########################COMPROVAR QUE ESTIGUI BE AIXO

        # manera crear el tokenizer
        if self.corpus_tokenizer == 1:  #tipologia tokenizer (categoric numeric)
            print(True,'catnum')
            self.add_feature_transform_dataset(self.dataset.input_info)
        elif self.corpus_tokenizer == 2: # categoria tokenizer (metrica anlitiques condicions)
            print('tokenizer agrupat')
            self.add_feature_transform_grup(self.dataset.input_info)
        
        else: # tokenizer per columna
          for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]  # input_info es un diccionari de diccionaris
            self.add_feature_transform_layer(feature_key, input_info) #dins aquesta funcio se passa es diccionari de feature i se fa es dict feat_tokenizer


        self.transformer = nn.ModuleDict()


        self.transformer["numeric"] =  TransformerLayer(
            feature_size=embedding_dim, **kwargs
        )
        self.transformer["categoric"] =  TransformerLayer(
            feature_size=embedding_dim, **kwargs
        )
        self.fc = nn.Linear(2*self.embedding_dim, output_size)



    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb_categoric = []
        patient_emb_numeric = []
        patient_mask_categoric = []
        patient_mask_numeric = []
        patient_emb = []
        pattient_mask = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                
                if self.from_categorical_pretrained:
                    batch_tokens = [x[0] for x in kwargs[feature_key]]
                    x = self.pretrained_tokenizer(batch_tokens, return_tensors="pt", padding=True, truncation=False)
                    x = x.to(device)
                    with torch.no_grad():
                        outputs = self.language_model(**x)
                        x = outputs.last_hidden_state[:,0,:]
                        x = self.pretrained_proj(x) 
                        x = x.unsqueeze(1)
                    mask = torch.any(x !=0, dim=2)
                        # ficar capa lineal fins que 128 #######################
                else:
                    x = self.feat_tokenizers[feature_key].batch_encode_2d(
                        kwargs[feature_key] )
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    x = self.embeddings[feature_key](x)
                    mask = torch.any(x !=0, dim=2)
                
                patient_emb_categoric.append(x)
                patient_mask_categoric.append(mask)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key] )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                x = torch.sum(x, dim=2)
                mask = torch.any(x !=0, dim=2)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = self.linear_layers[feature_key](x)
                mask = mask.bool().to(self.device)
                patient_emb_numeric.append(x)
                patient_mask_numeric.append(mask)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            else:
                raise NotImplementedError
           

          
            patient_emb.append(x)
            pattient_mask.append(mask)



        patient_emb_numeric = torch.cat(patient_emb_numeric, dim=1)
        patient_mask_numeric = torch.cat(patient_mask_numeric, dim=1)
        _, x_num = self.transformer["numeric"](patient_emb_numeric, patient_mask_numeric, kwargs.get('register_hook'))
    
        patient_emb_categoric = torch.cat(patient_emb_categoric, dim=1)
        patient_mask_categoric = torch.cat(patient_mask_categoric, dim=1)
        _, x_cat = self.transformer["categoric"](patient_emb_categoric, patient_mask_categoric, kwargs.get('register_hook'))

        x = torch.cat([x_num, x_cat], dim=1)

        logits = self.fc(x)
        return logits



class TransformerGroup(BaseModel):

    def __init__(self, dataset, feature_keys: List[str], label_key: str, mode: str, corpus_tokenizer: bool,
        embedding_dim: int = 128, output_size=2, merged=False, from_categorical_pretrained: bool = True, 
        dic_group: dict = None, **kwargs):
        super(TransformerGroup, self).__init__(dataset=dataset, feature_keys=feature_keys, 
                                               label_key=label_key, mode=mode)
        
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = {}   # the key of self.feat_tokenizers only contains the code based inputs
        self.label_key = label_key
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.from_categorical_pretrained = from_categorical_pretrained
        self.corpus_tokenizer = corpus_tokenizer
        self.dic_group = dic_group if dic_group else {"numeric": [], "categoric": []}
        
        # Check if all feature keys are assigned to a group
        all_features_in_groups = []
        for features in self.dic_group.values():
            all_features_in_groups.extend(features)
        
        missing_features = set(feature_keys) - set(all_features_in_groups)
        if missing_features:
            raise ValueError(f"Some features are not assigned to any group: {missing_features}")

        # variables pretrained
        if from_categorical_pretrained:
            self.pretrained_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.language_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.language_model.to(device)
            self.pretrained_proj = nn.Linear(768, self.embedding_dim)

       # manera crear el tokenizer
        if self.corpus_tokenizer == 1:  #tipologia tokenizer (categoric numeric)
            print(True,'catnum')
            self.add_feature_transform_dataset(self.dataset.input_info)
        elif self.corpus_tokenizer == 2: # categoria tokenizer (metrica anlitiques condicions)
            print('tokenizer agrupat')
            self.add_feature_transform_grup(self.dataset.input_info)
        
        else: # tokenizer per columna
          for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]  # input_info es un diccionari de diccionaris
            self.add_feature_transform_layer(feature_key, input_info) #dins aquesta funcio se passa es diccionari de feature i se fa es dict feat_tokenizer


        # Create a transformer for each group in the dictionary
        self.transformer = nn.ModuleDict()
        for group_name in self.dic_group.keys():
            self.transformer[group_name] = TransformerLayer(
                feature_size=embedding_dim, **kwargs
            )
        
        # Final fully connected layer with adjusted input size based on number of groups
        self.fc = nn.Linear(len(self.dic_group) * self.embedding_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # Dictionary to store embeddings and masks for each group
        group_embeddings = {group: [] for group in self.dic_group.keys()}
        group_masks = {group: [] for group in self.dic_group.keys()}
        
        for feature_key in self.feature_keys:
            
            # no hauria de petar pero per si un cas
            group = None
            for group_name, features in self.dic_group.items():
                if feature_key in features:
                    group = group_name
                    break 
            if group is None:
                raise ValueError(f"Feature {feature_key} not found in any group")
                
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                if self.from_categorical_pretrained:
                    batch_tokens = [x[0] for x in kwargs[feature_key]]
                    x = self.pretrained_tokenizer(batch_tokens, return_tensors="pt", padding=True, truncation=False)
                    x = x.to(device)
                    with torch.no_grad():
                        outputs = self.language_model(**x)
                        x = outputs.last_hidden_state[:,0,:]
                        x = self.pretrained_proj(x) 
                        x = x.unsqueeze(1)
                    mask = torch.any(x != 0, dim=2)
                else:
                    x = self.feat_tokenizers[feature_key].batch_encode_2d(kwargs[feature_key])
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    x = self.embeddings[feature_key](x)
                    mask = torch.any(x != 0, dim=2)
                
            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                x = torch.sum(x, dim=2)
                mask = torch.any(x != 0, dim=2)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = self.linear_layers[feature_key](x)
                mask = mask.bool().to(self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            else:
                raise NotImplementedError
           
            group_embeddings[group].append(x)
            group_masks[group].append(mask)

        group_outputs = []
        for group_name in self.dic_group.keys():
            if not group_embeddings[group_name]:
                raise ValueError(f"Group {group_name} has no features assigned")
                
            group_emb = torch.cat(group_embeddings[group_name], dim=1)
            group_mask = torch.cat(group_masks[group_name], dim=1)
            
            _, group_output = self.transformer[group_name](group_emb, group_mask, kwargs.get('register_hook'))
            group_outputs.append(group_output)

        x = torch.cat(group_outputs, dim=1)

        logits = self.fc(x)
        return logits
