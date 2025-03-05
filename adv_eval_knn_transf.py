"""Evaluation modules and script.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import hydra
import sys

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
from torchvision.transforms import Compose
from helpers import setup_seed

from torchmetrics import AveragePrecision, Accuracy

from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug
# from helpers import (parse_args, get_exp_name, config_loggers,
#                                  get_num_worker)
from datasets.loaders import get_num_worker
from main import HiDiscModel
import argparse

import logging

from attacks.pgd import PGD_KNN



# code for kNN prediction is from the github repo IgorSusmelj/barlowtwins
# https://github.com/IgorSusmelj/barlowtwins/blob/main/utils.py
def knn_predict(feature, feature_bank, feature_labels, classes: int,
                knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features from a feature bank.

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: Temperature
    """
    # cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1,
                              index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) *
                            sim_weight.unsqueeze(dim=-1),
                            dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels, pred_scores


def get_embeddings(cf, experiments, log) :
    """Run forward pass on the dataset, and generate embeddings and logits"""


    # above code if argparser is used
    if cf.source_model_backbone.startswith("vit"):
        aug_func = get_srh_vit_base_aug
    else:
        aug_func = get_srh_base_aug



    train_dset = OpenSRHDataset(data_root=cf.data_db_root,
                                                studies="train",
                                                transform=Compose(aug_func()),
                                                balance_patch_per_class=False)


    train_loader = torch.utils.data.DataLoader( train_dset, batch_size=cf.eval_predict_batch_size,
                                                drop_last=False, pin_memory=True, persistent_workers=False)


    val_dset = OpenSRHDataset(data_root=cf.data_db_root, studies="val",
                              transform=Compose(aug_func()), balance_patch_per_class=False)


    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=cf.eval_predict_batch_size,
                                             drop_last=False, pin_memory=True, persistent_workers=False)


    ### Loading Source Model
    source_ckpt_path = cf.source_ckpt_path


    source_model_dict = {"model": {"backbone": cf.source_model_backbone, "mlp_hidden": experiments[cf.source_exp_no]["model.mlp_hidden"],
                            "num_embedding_out": experiments[cf.source_exp_no]["model.num_embedding_out"], "proj_head": experiments[cf.source_exp_no]["model.proj_head"],
                            }
                  }
    source_model = HiDiscModel(source_model_dict)
    dual_bn = True if cf.source_model_backbone == "resnet50_multi_bn" else False

    if cf.load_source_from_ssl:
        source_ckpt = torch.load(source_ckpt_path)

        if "state_dict" in source_ckpt.keys():
            msg = source_model.load_state_dict(source_ckpt["state_dict"])
            source_epoch = source_ckpt["epoch"]
        else:
            source_ckpt_weights = source_ckpt["model"]
            source_epoch = source_ckpt["epoch"]
            source_ckpt_weights = {k.replace("module.model", "model"): v for k, v in source_ckpt_weights.items()}
            msg = source_model.load_state_dict(source_ckpt_weights)

        log.info(f"Loaded Source model {cf.source_model_backbone} from {source_ckpt_path} Epoch {source_epoch} with message {msg}")
    else:
        log.info(f"Loaded Source model {cf.source_model_backbone} trained on ImageNet")

    source_model.to("cuda")
    source_model.eval()

    if cf.attack_name == "pgd_knn":
        attack = PGD_KNN(model=source_model, eps=cf.eps/255.0, alpha=2/255, steps=cf.steps)
    else:
        raise ValueError(f"Attack {cf.attack_name} not implemented")


    ### Loading Target Model
    target_ckpt_path = cf.target_ckpt_path
    target_model_dict = {"model": {"backbone": cf.target_model_backbone, "mlp_hidden": experiments[cf.target_exp_no]["model.mlp_hidden"],
                            "num_embedding_out": experiments[cf.target_exp_no]["model.num_embedding_out"], "proj_head": experiments[cf.target_exp_no]["model.proj_head"],
                            }
                  }
    target_model = HiDiscModel(target_model_dict)
    dual_bn = True if cf.target_model_backbone == "resnet50_multi_bn" else False

    target_ckpt = torch.load(target_ckpt_path)

    if "state_dict" in target_ckpt.keys():
        msg = target_model.load_state_dict(target_ckpt["state_dict"])
        target_epoch = target_ckpt["epoch"]
    else:
        target_ckpt_weights = target_ckpt["model"]
        target_epoch = target_ckpt["epoch"]
        target_ckpt_weights = {k.replace("module.model", "model"): v for k, v in target_ckpt_weights.items()}
        msg = target_model.load_state_dict(target_ckpt_weights)

    log.info(f"Loaded target model {cf.target_model_backbone} from {target_ckpt_path} Epoch {target_epoch} with message {msg}")
    target_model.to("cuda")
    target_model.eval()



    train_predictions = []
    for i, batch in enumerate(train_loader):

        images = batch["image"].to("cuda")
        labels = batch["label"].to("cuda")


        with torch.no_grad():
            outputs = target_model.get_features(images, bn_name="normal") if dual_bn else target_model.get_features(images)

        d = {"embeddings": outputs, "label": labels, "path": batch["path"]}
        log.info(f"BATCH {i} {d['embeddings'].shape} {d['label'].shape}")

        train_predictions.append(d)

    val_predictions = []

    for i, batch in enumerate(val_loader):

        images = batch["image"].to("cuda")
        labels = batch["label"].to("cuda")

        adv_images = attack(images, labels, dual_bn)

        with torch.no_grad():
            outputs = target_model.get_features(adv_images, bn_name="normal") if dual_bn else target_model.get_features(adv_images)

        d = {"embeddings": outputs, "label": labels, "path": batch["path"]}

        log.info(f"BATCH {i} {d['embeddings'].shape} {d['label'].shape}")
        val_predictions.append(d)

    def process_predictions(predictions):
        pred = {}
        for k in predictions[0].keys():
            if k == "path":
                pred[k] = [pk for p in predictions for pk in p[k][0]]
            else:
                pred[k] = torch.cat([p[k] for p in predictions])
        return pred

    train_predictions = process_predictions(train_predictions)
    val_predictions = process_predictions(val_predictions)

    train_embs = torch.nn.functional.normalize(train_predictions["embeddings"],
                                               p=2,
                                               dim=1).T
    val_embs = torch.nn.functional.normalize(val_predictions["embeddings"],
                                             p=2,
                                             dim=1)

    log.info(train_predictions["embeddings"].shape)
    log.info(train_predictions["label"].shape)

    # knn evaluation
    # batch_size = cf["eval"]["knn_batch_size"]
    batch_size = cf.eval_knn_batch_size
    all_scores = []
    for k in tqdm(range(val_embs.shape[0] // batch_size + 1)):
        start_coeff = batch_size * k
        end_coeff = min(batch_size * (k + 1), val_embs.shape[0])
        val_embs_k = val_embs[start_coeff:end_coeff]  # 1536 x 2048

        pred_labels, pred_scores = knn_predict(
            val_embs_k,
            train_embs,
            train_predictions["label"],
            len(train_loader.dataset.classes_),
            knn_k=200,
            knn_t=0.07)

        all_scores.append(
            torch.nn.functional.normalize(pred_scores, p=1, dim=1))
        torch.cuda.empty_cache()

    val_predictions["logits"] = torch.vstack(all_scores)
    return val_predictions


def make_specs(predictions: Dict[str, Union[torch.Tensor, List[str]]], log) -> None:
    """Compute all specs for an experiment"""

    # aggregate prediction into a dataframe
    pred = pd.DataFrame.from_dict({
        "path":
        predictions["path"],
        "labels": [l.item() for l in list(predictions["label"])],
        "logits": [l.tolist() for l in list(predictions["logits"])]
    })
    pred["logits"] = pred["logits"].apply(
        lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0))

    # add patient and slide info from patch paths
    pred["patient"] = pred["path"].apply(lambda x: x.split("/")[-4])
    pred["slide"] = pred["path"].apply(lambda x: "/".join(
        [x.split("/")[-4], x.split("/")[-3]]))

    # aggregate logits
    get_agged_logits = lambda pred, mode: pd.DataFrame(
        pred.groupby(by=[mode, "labels"])["logits"].apply(
            lambda x: [sum(y) for y in zip(*x)])).reset_index()

    slides = get_agged_logits(pred, "slide")
    patients = get_agged_logits(pred, "patient")

    normalize_f = lambda x: torch.nn.functional.normalize(x, dim=1, p=1)
    patch_logits = normalize_f(torch.tensor(np.vstack(pred["logits"])))
    slides_logits = normalize_f(torch.tensor(np.vstack(slides["logits"])))
    patient_logits = normalize_f(torch.tensor(np.vstack(patients["logits"])))

    patch_label = torch.tensor(pred["labels"])
    slides_label = torch.tensor(slides["labels"])
    patient_label = torch.tensor(patients["labels"])

    # generate metrics
    def get_all_metrics(logits, label):
        map = AveragePrecision(num_classes=7, task="multiclass")
        acc = Accuracy(num_classes=7, task="multiclass")
        t2 = Accuracy(num_classes=7, top_k=2, task="multiclass")
        t3 = Accuracy(num_classes=7, top_k=3, task="multiclass")
        mca = Accuracy(num_classes=7, average="macro", task="multiclass")

        logits = logits.squeeze()  # Remove extra dimensions from logits
        acc_val = acc(logits, label)
        t2_val = t2(logits, label)
        t3_val = t3(logits, label)
        mca_val = mca(logits, label)
        map_val = map(logits, label)

        return torch.stack((acc_val, t2_val, t3_val, mca_val, map_val))

    all_metrics = torch.vstack((get_all_metrics(patch_logits, patch_label),
                                get_all_metrics(slides_logits, slides_label),
                                get_all_metrics(patient_logits,
                                                patient_label)))
    all_metrics = pd.DataFrame(all_metrics,
                               columns=["acc", "t2", "t3", "mca", "map"],
                               index=["patch", "slide", "patient"])

    # generate confusion matrices
    patch_conf = confusion_matrix(y_true=patch_label,
                                  y_pred=patch_logits.argmax(dim=1))

    slide_conf = confusion_matrix(y_true=slides_label,
                                  y_pred=slides_logits.argmax(dim=1))

    patient_conf = confusion_matrix(y_true=patient_label,
                                    y_pred=patient_logits.argmax(dim=1))

    log.info("\nmetrics")
    log.info(all_metrics)
    log.info("\npatch confusion matrix")
    log.info(patch_conf)
    log.info("\nslide confusion matrix")
    log.info(slide_conf)
    log.info("\npatient confusion matrix")
    log.info(patient_conf)

    return

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1000)

    parser.add_argument('--data_db_root', type=str, default=r'F:\Code\datasets\hidisc_data_small')
    parser.add_argument('--data_meta_json', type=str, default='opensrh.json')
    parser.add_argument('--data_meta_split_json', type=str, default='train_val_split.json')

    parser.add_argument('--source_model_backbone', type=str, default='resnet50_at')
    parser.add_argument('--source_exp_no', type=int, default=18)
    parser.add_argument('--source_ckpt_path', type=str, default=r'Results/Baseline/resnet50_at_exp18/checkpoint_40000.pth')
    parser.add_argument('--load_source_from_ssl', default=True, type=lambda x: (str(x).lower() == 'true'))




    parser.add_argument('--target_model_backbone', type=str, default='resnet50')
    parser.add_argument('--target_exp_no', type=int, default=18)
    parser.add_argument('--target_ckpt_path', type=str, default=r'Results/Baseline/resnet50_exp18/checkpoint_40000.pth')


    parser.add_argument('--eval_predict_batch_size', type=int, default=32)
    parser.add_argument('--eval_knn_batch_size', type=int, default=1024)

    parser.add_argument('--save_results_path', type=str, default='eval_knn_transf_results')

    parser.add_argument('--attack_name', type=str, default='pgd_knn')
    parser.add_argument('--eps', type=int, default=8)
    parser.add_argument('--steps', type=int, default=7)

    args = parser.parse_args()

    return args

def main():


    experiments = {
        18: {"model.num_embedding_out": 128, "model.proj_head": False, "model.mlp_hidden": []},
        19: {"model.num_embedding_out": 128, "model.proj_head": False, "model.mlp_hidden": []},
        20: {"model.num_embedding_out": 128, "model.proj_head": False, "model.mlp_hidden": []},
        28: {"model.num_embedding_out": 128, "model.proj_head": False, "model.mlp_hidden": []},
        244: {"model.num_embedding_out": 128, "model.proj_head": False, "model.mlp_hidden": []},
        24: {"model.num_embedding_out": 256, "model.proj_head": False, "model.mlp_hidden": []},
        25: {"model.num_embedding_out": 512, "model.proj_head": False, "model.mlp_hidden": []},
        26: {"model.num_embedding_out": 768, "model.proj_head": False, "model.mlp_hidden": []},
        27: {"model.num_embedding_out": 1024, "model.proj_head": False, "model.mlp_hidden": []},
        29: {"model.num_embedding_out": 2048, "model.proj_head": True, "model.mlp_hidden": []},
        30: {"model.num_embedding_out": 2048, "model.proj_head": False, "model.mlp_hidden": [2048, 2048]}
    }

    cf = get_args()

    results_path = cf.save_results_path
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    if cf.load_source_from_ssl:

        source_exp_no = cf.source_exp_no
        source_ckpt_path = cf.source_ckpt_path
        if not os.path.exists(source_ckpt_path):
            print(f"Checkpoint file {source_ckpt_path} does not exist. Exiting.")
            sys.exit(1)

        source_ckpt = torch.load(source_ckpt_path)
        source_epoch = source_ckpt["epoch"]
        del source_ckpt

    target_exp_no = cf.target_exp_no
    target_ckpt_path = cf.target_ckpt_path
    target_ckpt = torch.load(target_ckpt_path)
    target_epoch = target_ckpt["epoch"]
    del target_ckpt


    if cf.load_source_from_ssl:
        log_dir = os.path.join(results_path, f"S_{cf.source_model_backbone}_epoch{source_epoch}_exp_{source_exp_no}_T_{cf.target_model_backbone}_epoch{target_epoch}_exp_{target_exp_no}_adv_eval_{cf.attack_name}_{cf.steps}_eps{cf.eps}.log")
    else:
        log_dir = os.path.join(results_path, f"S_{cf.source_model_backbone}_T_{cf.target_model_backbone}_epoch{target_epoch}_exp_{target_exp_no}_adv_eval_{cf.attack_name}_{cf.steps}_eps{cf.eps}.log")

    logging.basicConfig(filename=log_dir, filemode="a",
                        format="%(name)s → %(levelname)s: %(message)s")
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # add console handler to logger
    log.addHandler(ch)
    log.info("Info level message")
    log.info("Logs will be saved in %s", log_dir)
    # log the source model name, target model name, attack name, eps, steps, load_source_from_ssl, target_exp_no, target_epoch, target_ckp_path
    log.info(f"Source Model: {cf.source_model_backbone} Target Model: {cf.target_model_backbone} Attack: {cf.attack_name} Eps: {cf.eps} Steps: {cf.steps}"
             f" Load Source from SSL: {cf.load_source_from_ssl} Target Exp No: {cf.target_exp_no} Target Epoch: {target_epoch} Target CKP Path: {cf.target_ckpt_path}")


    setup_seed(cf.seed)
    if cf.load_source_from_ssl:
        prediction_path = os.path.join(cf.save_results_path, f"predictions_S_{cf.source_model_backbone}_epoch{source_epoch}_exp_{source_exp_no}_T_{cf.target_model_backbone}_epoch{target_epoch}_exp_{target_exp_no}_adv_eval_{cf.attack_name}_{cf.steps}_eps{cf.eps}.pt")
    else:
        prediction_path = os.path.join(cf.save_results_path, f"predictions_S_{cf.source_model_backbone}_T_{cf.target_model_backbone}_epoch{target_epoch}_exp_{target_exp_no}_adv_eval_{cf.attack_name}_{cf.steps}_eps{cf.eps}.pt")


    # if os.path.exists(prediction_path):
    #     log.info("loading predictions")
    #     predictions = torch.load(prediction_path)

    log.info("generating predictions")
    predictions = get_embeddings(cf, experiments, log=log)
    torch.save(predictions, prediction_path)

    # generate specs
    make_specs(predictions, log=log)


if __name__ == "__main__":
    main()