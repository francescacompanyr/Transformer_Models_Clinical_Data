from itertools import chain
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random

def split_v1(
    dataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None ,label = "type"):

    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_ids = [dataset[i]['patient_id'][0] for i in range(len(dataset))]
    num_patients = len(patient_ids)
    
    idx = np.arange(num_patients)  
    train_index, val_index = train_test_split(idx, test_size=ratios[1], random_state=42)
    #val_index, test_index = train_test_split(temp_idx, test_size=0.5, random_state=42)  

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    #test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_labels = [dataset[i][label][0] for i in train_index]
    val_labels = [dataset[i][label][0] for i in val_index]
   # test_labels = [dataset[i]['Diabetes_binary'][0] for i in test_index]

    test_dataset = None
    test_labels = None
    return train_dataset, val_dataset, test_dataset == None, train_labels, val_labels, test_labels == None


def split(
    dataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,):
    """
    Split dataset keeping all positive cases (diabetes=1) and an equal number of negative cases.
    The remaining negative cases are discarded.
    
    Args:
        dataset: The dataset to split
        ratios: Proportions for train/val splits, must sum to 1.0
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels)
    """
 

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    
    # Extract indices of the dataset by class
    diabetes_indices = []    # Positive cases (with diabetes)
    non_diabetes_indices = []  # Negative cases (without diabetes)
    
    for i in range(len(dataset)):
        if dataset[i]['Diabetes_binary'][0] == 1:
            diabetes_indices.append(i)
        else:
            non_diabetes_indices.append(i)
    
    # Count positive cases
    num_positive = len(diabetes_indices)
    print(f"Found {num_positive} positive cases (with diabetes)")
    print(f"Found {len(non_diabetes_indices)} negative cases (without diabetes)")
    
    # Shuffle negative indices to randomly select which ones to keep
    random.shuffle(non_diabetes_indices)
    num_positive = num_positive #int(num_positive * 4)

    non_diabetes_indices = non_diabetes_indices[:num_positive]
    print(f" {len(non_diabetes_indices)} pos ")
    
    # Now we have equal number of positive and negative cases
    all_indices = diabetes_indices + non_diabetes_indices
    random.shuffle(all_indices)  # Shuffle to mix positive and negative cases
    
    # Split according to the ratios
    train_size = int(len(all_indices) * ratios[0])
    
    train_index = all_indices[:train_size]
    val_index = all_indices[train_size:]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    
    # Extract labels for each subset
    train_labels = [dataset[i]['Diabetes_binary'][0] for i in train_index]
    val_labels = [dataset[i]['Diabetes_binary'][0] for i in val_index]
    
    # Verify the balance:
    train_diabetes_count = sum(train_labels)
    val_diabetes_count = sum(val_labels)
    
    print(f"Training set: {train_diabetes_count}/{len(train_labels)} ({train_diabetes_count/len(train_labels)*100:.2f}%) with diabetes")
    print(f"Validation set: {val_diabetes_count}/{len(val_labels)} ({val_diabetes_count/len(val_labels)*100:.2f}%) with diabetes")
    
    # Verify the balance:
    train_diabetes_ratio = sum(train_labels) / len(train_labels)
    val_diabetes_ratio = sum(val_labels) / len(val_labels)
    
    print(f"Training set: {train_diabetes_ratio*100:.2f}% with diabetes")
    print(f"Validation set: {val_diabetes_ratio*100:.2f}% with diabetes")
    
    # Handle test set if needed (currently not used)
    test_dataset = None
    test_labels = None
    return train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels




def split_ratios(dataset, ratios=(0.8, 0.2, 0), seed=42 ):
    pass


def splitsd(dataset, ratios=(0.8, 0.2, 0), seed=42):
    np.random.seed(seed)
    patient_ids = dataset.df["patient_id"]
    num_patients = len(patient_ids)

    idx = np.arange(num_patients)

    # Ahora sacamos los labels correctos para estratificar
    labels = dataset.df.drop_duplicates(subset="patient_id")["Diabetes_binary"].values

    train_index, temp_idx = train_test_split(
        idx,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    # ¡Ojo! Ahora labels también hay que filtrarlos para temp_idx
    labels_temp = labels[temp_idx]

    val_index, test_index = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=labels_temp,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    return train_dataset, val_dataset, test_dataset
