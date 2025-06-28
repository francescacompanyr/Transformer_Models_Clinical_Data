import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
from datetime import datetime

from collections import Counter

from dataset.utils import strptime

# TODO: add other tables

class RadioLungDataset():
    
    def __init__(self, path_df: str, label_key=None, mode = None, contextual = None):
        
        self.df = self.merge_datasets(path_df)
        self.dataset_name = "RadioLung"
        self.map_dataset()
        if contextual == True:
            self.improve_dataset_tokenizer()

        self.input_info = {} # com estan cada feature_key
        self.map_feature_token_space()
        self.mode = None


    def map_dataset(self):
        
        self.df.dropna(inplace=True)

        self.df["birthdate"] = pd.to_datetime(self.df["birthdate"])
        self.df["age"] = datetime.today().year - self.df["birthdate"].dt.year

        self.df.drop(columns=["birthdate"], inplace=True)
        self.df["type"] = self.df["type"].map({"Malignant": 1, "Benign": 0})
        
        #per fer finetuning
        self.df["Education"] = self.df["education"]
        self.df["Age"] = self.df["age"]
        self.df["Smoker"] = self.df["smoking"]   
        self.df["Sex"] = self.df["sex"]    
        print(self.df)
        print(self.df.columns)




    def __len__(self):
        return self.df.shape[0]
    

    def get_df(self):
        return self.df
    
    def map_feature_token_space(self) -> None:
        feature_keys = self.df.columns
        self.map_token_space = {}
        for feature_key in feature_keys:
           
            token_space = self.df[feature_key].unique().tolist() #token_space = self.df[feature_key].unique().flatten().tolist()
            if pd.api.types.is_string_dtype(self.df[feature_key] ) == True:
                self.input_info[feature_key] = {"type": str, "dim": 2} 
            else:
                self.input_info[feature_key] = {"type": int, "dim": 2, 'len':1}# ha petat i tb i he ficat len 
             
                self.df[feature_key]= (self.df[feature_key].apply(lambda x: [x])) # he ficat tots es num dins una llista pq vol aquest format
            


    def merge_datasets(self, list_paths):
            df_list = []  
            selected_columns = ["patient_id", "hospital", "type", "birthdate", "sex", "BMI", "education",
                                "air_pollution", "smoking", "packyear_index", "fh_cancer",
                                "ph_cancer", "COPD", "FVC", "indice", "DLCO"]
       
            for p in list_paths:
                data_read = pd.read_excel(p, header=0)
                data_read = data_read.reindex(columns=selected_columns)
                df_list.append(data_read)

            self.df = pd.concat(df_list, ignore_index=True)
            return self.df

        

         
    def get_all_tokens(self, key:str):
        return self.df[key].unique().flatten().tolist()
    
    def get_distribution_tokens(self, key: str) -> Dict[str, int]:
        tokens = self.get_all_tokens(key)
        counter = Counter(tokens)
        return counter

    def improve_dataset_tokenizer (self):
     
        binary_to_categorical = {
        "ph_cancer": {
            0: "No physical cancer risk", 
            1: "High physical cancer risk"
        },
        "fh_cancer": {
            "No": "No family cancer history", " No": "No family cancer history", 
            "(Yes) Lung cancer": "Family lung cancer history present",
            "(Yes) others": "Family others cancer history present" , " (Yes) others": "Family others cancer history present"
        },
        "air_pollution": {
            0: "Low air pollution exposure", 
            1: "High air pollution exposure"
        }}
        
        for col, mapping in binary_to_categorical.items():
            print(col, mapping)
            self.df[col] = self.df[col].map(mapping)
            print(self.df[col].value_counts())   


    def stat(self) -> str:
        lines = list()
        lines.append(f"Statistics of sample dataset:")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Task: Detect lung cancer") #{self.task_name}
        lines.append(f"\t- Number of patients: {len(self)}")
        num_patients = len(self)
        lines.append(   f"\t- Number of samples per patient: {len(self) / num_patients:.4f}")
        print("\n".join(lines))
        return "\n".join(lines)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        sample_return = {fkey: [value] if fkey != "type" else value for fkey, value in sample.items()}
        sample_return.update({"patient_id": sample_return["patient_id"]})
        return sample_return

class BaseDataset():
    
    def __init__(self, path_df: str, label_key=None, mode = None, contextual = None):
        
        self.df = path_df
        self.dataset_name = "Diabetes"
        
        self.label_key = label_key

        self.map_dataset()
        if contextual == True:
            print('entra')
            self.improve_dataset_tokenizer()
        self.input_info = {} 
        self.map_feature_token_space()
        self.mode = None
        

    def map_dataset(self):
        
        self.df.dropna(inplace=True)
        if ("birthdate"  or "Birthdate") in self.df.columns:
            self.df["birthdate"] = pd.to_datetime(self.df["birthdate"])
            self.df["age"] = datetime.today().year - self.df["birthdate"].dt.year

            self.df.drop(columns=["birthdate"], inplace=True)
        values = self.df[self.label_key ].unique()
        self.df.rename(columns={'ID': 'patient_id'}, inplace=True)

        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()

        print('num',numerical_cols)
        print('cat',categorical_cols)

        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if converted.notna().sum() / len(converted) > 0.8:  # 80% de valores válidos
                self.df[col] = converted
                print(f"Convertida: {col}")
            else:
                print(f"Mantenida como string: {col}")

        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()

        print('num',numerical_cols)
        print('cat',categorical_cols)
        print(self.df.columns)
        if len(values) == 2:
            mapping = {values[0]: 0, values[1]: 1}
            self.df[self.label_key ] = self.df[self.label_key ].map(mapping)
        else:
            raise ValueError("More than 2 elements in label key",values)
        



    def __len__(self):
        return self.df.shape[0]
    

    def get_df(self):
        return self.df
    
    def map_feature_token_space(self) -> None:
        feature_keys = self.df.columns
        self.map_token_space = {}
        for feature_key in feature_keys:
            token_space = self.df[feature_key].unique().tolist() #token_space = self.df[feature_key].unique().flatten().tolist()
            if pd.api.types.is_string_dtype(self.df[feature_key] ) == True:
                self.input_info[feature_key] = {"type": str, "dim": 2} 
            else:
                self.input_info[feature_key] = {"type": int, "dim": 2, 'len':1}
             
                self.df[feature_key]= (self.df[feature_key].apply(lambda x: [x])) 
            
      

         
    def get_all_tokens(self, key:str):
        return self.df[key].unique().flatten().tolist()
    
    def get_distribution_tokens(self, key: str) -> Dict[str, int]:
        tokens = self.get_all_tokens(key)
        counter = Counter(tokens)
        return counter
    
    def improve_dataset_tokenizer (self):
        binary_to_categorical = {
        "HighChol": {"0": "Low cholesterol", "1": "High cholesterol"},  
            "HvyAlcoholConsump": {"0": "Not heavy drinker", "1": "Heavy alcohol use"},
            "Sex": {"0": "Female", "1": "Male"},
            "HeartDiseaseorAttack": {"0": "No heart condition", "1": "Had heart disease"},
            "Stroke": {"0": "No stroke", "1": "Had stroke"},
            "Smoker": {"0": "Non-smoker", "1": "Smoker"},
            "HighBP": {"0": "Normal blood pressure", "1": "High blood pressure"},
            "PhysActivity": {"0": "Physically inactive", "1": "Physically active"},
            "Fruits": {"0": "Does not eat fruits", "1": "Eats fruits regularly"},
            "Veggies": {"0": "Does not eat vegetables", "1": "Eats vegetables regularly"},
        }

        for col, mapping in binary_to_categorical.items():
            self.df[col] = self.df[col].map(mapping)

        
    def stat(self) -> str:
        lines = list()
        lines.append(f"Statistics of sample dataset:")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Task: Detect lung cancer") #{self.task_name}
        lines.append(f"\t- Number of patients: {len(self)}")
        num_patients = len(self)
        lines.append(   f"\t- Number of samples per patient: {len(self) / num_patients:.4f}")
        print("\n".join(lines))
        return "\n".join(lines)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        sample_return = {fkey: [value] if fkey != self.label_key else value for fkey, value in sample.items()}
        sample_return.update({"patient_id": sample_return["patient_id"]})
        return sample_return

