import logging
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Type
import wandb 
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.autonotebook import trange
from sklearn import metrics
from metrics.binary import binary_metrics_fn

from metrics.multiclass import multiclass_metrics_fn
from metrics.regression import regression_metrics_fn
from utils import create_directory
from sklearn.metrics import recall_score, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import *

import time

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")
def set_logger(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    
    if logger.hasHandlers():
        logger.handlers.clear()  # Limpia handlers anteriores

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def set_logger_false(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return

def load_checkpoint(self, checkpoint_path):
    """Carrega un checkpoint al model del trainer"""
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    if 'model_state_dict' in checkpoint:
        self.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        self.model.load_state_dict(checkpoint)

def get_metrics_fn(mode: str) -> Callable:
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "regression":
        return regression_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")



class Trainer:

    def __init__( self, model: nn.Module, checkpoint_path: Optional[str] = None, metrics: Optional[List[str]] = None,
        device: Optional[str] = None, enable_logging: bool = True, output_path: Optional[str] = None,
        exp_name: Optional[str] = None,  info_wandb = None   ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.metrics = metrics
        self.device = device
        self.info_wandb= info_wandb
        # set logger
        if enable_logging:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "output")
            if exp_name is None:
                exp_name = self.info_wandb["nom dataset"]

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            folder_name = f"{exp_name}_{timestamp}"
            self.exp_path = os.path.join(output_path, folder_name)
            set_logger(self.exp_path)
        else:
            self.exp_path = None

        self.exp_name=exp_name
        # set device
        self.model.to(self.device)

        # logging
        logger.info(self.model)
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Device: {self.device}")
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_ckpt(checkpoint_path)
        logger.info("")
        
        return

    def get_metrics ( self, labels, logits ):
        pass


    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[Dict[str, object]] = None,
        steps_per_epoch: int = None,
        evaluation_steps: int = 1,
        weight_decay: float = 0.0,
        max_grad_norm: float = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        labels = None,
        label_key = None,  
    ):

        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        dic_loss = {}
        dic_recall = {}
        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay, },
            { "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,},]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch == None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0

        run = wandb.init(
            name = self.exp_name,
            entity="tfg_pyh", project="tfg",
            config={
                "learning_rate": optimizer_params["lr"],
                "dataset": self.info_wandb['nom dataset'],
                "epochs": epochs, "Merged":self.info_wandb["separat"], 
                "pretrained": self.info_wandb['pretrained'], "corpus tokenizer": self.info_wandb['corpus tokenizer'],
                "notes":self.info_wandb["notes"]},)

   
        weights, _ = self.getNormalizedClassWeights(labels)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_func = torch.nn.CrossEntropyLoss( weight=weights )

        # epoch training loop
        for epoch in range(epochs):
            training_loss = []
            recall_0_list = []
            recall_1_list = []
            list_auc0 =[]
            list_auc1 =[]

            list_f1 = []
            pr_0_epochs = []
            tpr_0_epochs = []
            fpr_0_epochs = []
            fpr_1_epochs = []
            tpr_1_epochs = []






            self.model.zero_grad()
            self.model.train()
            _evaluar_final = {}
            # batch training loop
            logger.info("")
           # for _ in trange(steps_per_epoch, desc=f"Epoch {epoch} / {epochs}", smoothing=0.05, ):
            for _ in trange(steps_per_epoch, desc=f"Epoch {epoch} / {epochs}", smoothing=0.05, mininterval=6.0):

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward

                logits = self.model(**data)
                labels_train = data[label_key]


                y_true = torch.LongTensor(labels_train).squeeze(1).to(logits.device)
                y_prob = torch.softmax(logits, dim=-1)
              
                loss = loss_func(logits.to(device), y_true.to(device)) # bin--> cross_entropy_logits                
                report = classification_report(y_true.cpu(), torch.argmax(y_prob, dim = 1).cpu() ,output_dict=True, zero_division=0)
                recall_0 = report['0']['recall'] if '0' in report else 0.0
                recall_1 = report['1']['recall'] if '1' in report else 0.0
                f1_macro = report['macro avg']['f1-score']
              #  fpr, tpr, thresholds = metrics.roc_curve(y_true.cpu().numpy(), y_prob[:, 1].cpu().detach().numpy(), pos_label=1)
              #  l_auc = metrics.auc(fpr, tpr)
                f1 = f1_score(y_true.cpu().numpy(), torch.argmax(y_prob, dim = 1).cpu(), average='weighted')


                fpr_0, tpr_0, _ = metrics.roc_curve(y_true.cpu().numpy(), y_prob[:, 0].cpu().detach().numpy(), pos_label=0)
                auc_0 = metrics.auc(fpr_0, tpr_0)

                # AUC per classe 1 
                fpr_1, tpr_1, _ = metrics.roc_curve(y_true.cpu().numpy(), y_prob[:, 1].cpu().detach().numpy(), pos_label=1)
                auc_1 = metrics.auc(fpr_1, tpr_1)



             #   results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits, "classification_report":report, "auc":l_auc, "f1":f1}


                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()

                training_loss.append(loss.item())
                recall_0_list.append(recall_0)
                recall_1_list.append(recall_1)
                list_auc0.append(auc_0)
                list_auc1.append(auc_1)
                list_f1.append(f1)
                fpr_0_epochs.append(fpr_0.tolist())
                tpr_0_epochs.append(tpr_0.tolist())
                fpr_1_epochs.append(fpr_1.tolist())
                tpr_1_epochs.append(tpr_1.tolist())
                global_step += 1
           
            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            logger.info(f"auc_0: {sum(list_auc0) / len(list_auc0):.4f}")
            logger.info(f"auc_1: {sum(list_auc1) / len(list_auc1):.4f}")
            logger.info(f"f1: {sum(list_f1) / len(list_f1):.4f}")        
                    
          #  run.log({"epoch": epoch,"loss_train": sum(training_loss) / len(training_loss), 'recall0':sum(recall_0_list) / len(recall_0_list), "recall1":sum(recall_1_list) / len(recall_1_list), "f1":sum(list_f1) / len(list_f1)
           # , "auc0":sum(loss_auc) / len(loss_auc)})
        
            run.log({
                    "epoch": epoch,
                    "loss_train": sum(training_loss) / len(training_loss), 
                    'recall0': sum(recall_0_list) / len(recall_0_list), 
                    "recall1": sum(recall_1_list) / len(recall_1_list), 
                    "f1": sum(list_f1) / len(list_f1),
                    "auc_0": sum(list_auc0) / len(list_auc0),
                    "auc_1": sum(list_auc1) / len(list_auc1),
                    # ROC  wandb
                  #  "fpr_class_0": fpr_0.tolist(),
                  #  "tpr_class_0": tpr_0.tolist(),
                  #  "fpr_class_1": fpr_1.tolist(),
                  #  "tpr_class_1": tpr_1.tolist()
                })
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))
            
            # validation
            if val_dataloader is not None:
                
                scores = self.evaluate(val_dataloader, labels=labels, label_key = label_key)
             #   run.log({"epoch": epoch,"loss_val": scores["loss"], "recall0_val":scores["recall0"], "recall1_val":scores["recall1"]
             #   , "auc_val":scores['auc'], "f1_val":scores['f1']})
                run.log({
                    "epoch": epoch,
                    "loss_val": scores["loss"], 
                    'recall0_val': scores["recall0"], 
                    "recall1_val": scores["recall1"], 
                    "f1_val": scores["f1"],
                    "auc_0_val": scores["auc0"],
                    "auc_1_val": scores["auc1"],
                    # ROC  wandb
                  #  "fpr_class_0": fpr_0.tolist(),
                 #   "tpr_class_0": tpr_0.tolist(),
                  #  "fpr_class_1": fpr_1.tolist(),
                   # "tpr_class_1": tpr_1.tolist()
                })
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
               
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}" )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))
                


        # test
        
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader, labels=labels, label_key = label_key)
            run.log({
                    "epoch": epoch,
                    "loss_test": scores["loss"], 
                    'recall0_test': scores["recall0"], 
                    "recall1_test": scores["recall1"], 
                    "f1_test": scores["f1"],
                    "auc_0_test": scores["auc0"],
                    "auc_1_test": scores["auc1"],
                    # ROC  wandb
                  #  "fpr_class_0": fpr_0.tolist(),
                 #   "tpr_class_0": tpr_0.tolist(),
                  #  "fpr_class_1": fpr_1.tolist(),
                   # "tpr_class_1": tpr_1.tolist()
                })
            print('scoresr test',scores)

            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))
    
        run.finish()
        return None # results


    def evaluate(self, dataloader, labels, label_key) -> Dict[str, float]:
        loss_all = []
        recall_all_0 = []
        recall_all_1 = []
        list_auc0 =[]
        list_auc1 =[]

        list_f1 = []
        
        weights, _ = self.getNormalizedClassWeights(labels)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_func = torch.nn.CrossEntropyLoss( weight=weights )
        self.model.eval()

        for data in tqdm(dataloader, desc="Evaluation"):
            with torch.no_grad():

                logits = self.model(**data)
                labels = data[label_key]
                y_true = torch.LongTensor(labels).squeeze(1).to(logits.device)
                y_prob = torch.softmax(logits, dim=-1)
                loss = loss_func(logits.to(device), y_true.to(device)) # bin--> cross_entropy_logits
                acc = (y_true == torch.argmax(y_prob, dim = 1)).sum()/len(y_true)
                report = classification_report(y_true.cpu(), torch.argmax(y_prob, dim = 1).cpu() ,output_dict=True, zero_division=0)
                recall_0 = report['0']['recall']
                recall_1 = report['1']['recall']
                f1_macro = report['macro avg']['f1-score']
              
                
                fpr_0, tpr_0, _ = metrics.roc_curve(y_true.cpu().numpy(), y_prob[:, 0].cpu().detach().numpy(), pos_label=0)
                auc_0 = metrics.auc(fpr_0, tpr_0)

                # AUC per classe 1 
                fpr_1, tpr_1, _ = metrics.roc_curve(y_true.cpu().numpy(), y_prob[:, 1].cpu().detach().numpy(), pos_label=1)
                auc_1 = metrics.auc(fpr_1, tpr_1)


                f1 = f1_score(y_true.cpu().numpy(), torch.argmax(y_prob, dim = 1).cpu(), average='weighted')

                loss_all.append(loss.item())
                recall_all_0.append(recall_0)
                recall_all_1.append(recall_1)
                list_auc0.append(auc_0)
                list_auc1.append(auc_1)

                list_f1.append(f1)

        loss_mean = sum(loss_all) / len(loss_all)
        recall1_mean = sum(recall_all_1) / len(recall_all_1)
        recall0_mean = sum(recall_all_0) / len(recall_all_0)
        auc_mean0 = sum(list_auc0) / len(list_auc0) 
        auc_mean1 = sum(list_auc1) / len(list_auc1) 

        f1_mean = sum(list_f1) / len(list_f1)
        scores = {"loss": loss_mean, "recall0":recall0_mean, "recall1":recall1_mean, "auc0":auc_mean0,  "auc1":auc_mean1, "f1":f1_mean}
        return scores

        

    @torch.no_grad()
    def infer_one_sample(self, sample:dict) -> Dict[str, float]:
        
        logits = self.model(**sample)
        labels = sample[label_key]
        y_true = torch.LongTensor(labels).squeeze(1).to(logits.device)
        y_prob = torch.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, y_true)

        y_true = y_true.cpu().numpy()
        y_prob = y_prob.cpu().numpy()

        return [loss, y_true, y_prob]



        return outputs
    
    def inference(self, dataloader, additional_outputs=None, return_patient_ids=False) -> Dict[str, float]:
        loss_all = []
        y_true_all = []
        y_prob_all = []
        patient_ids = []
        if additional_outputs is not None:
            additional_outputs = {k: [] for k in additional_outputs}
        for data in tqdm(dataloader, desc="Evaluation"):
            self.model.eval()
            loss, y_true, y_prob = self.infer_one_sample(data)
           # run.log({"epoch": epoch,"loss_test?": loss})
            loss_all.append(loss.item())
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

            if return_patient_ids:
                patient_ids.extend(data["patient_id"])

        loss_mean = sum(loss_all) / len(loss_all)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        outputs = [y_true_all, y_prob_all, loss_mean]

        if additional_outputs is not None:
            additional_outputs = {key: np.concatenate(val)
                                  for key, val in additional_outputs.items()}
            outputs.append(additional_outputs)
        if return_patient_ids:
            outputs.append(patient_ids)

    def save_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = self.model.state_dict()
        torch.save(state_dict, ckpt_path)
        return

    def load_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return
    
    def getNormalizedClassWeights(self, y_train):   

        unique, counts = np.unique(y_train, return_counts=True)
        weights = 1./counts

        weights = weights / np.sum(weights)

        print("Classes weights: {}".format(weights))

        return weights, dict(zip(unique, counts))


