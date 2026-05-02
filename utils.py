from collections import defaultdict
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
import wandb

## wandb
def get_wandb_res(project_name, sweep_name = [], sweep_id = "", entity="Your-Entity", filter_accuracy=1):
    api = wandb.Api()
    sweeps = api.project(project_name, entity=entity).sweeps()

    best_run_dict = {}
    for sweep in sweeps:        
        include_sweep = True
        for part in sweep_name:
            if part.startswith("-"): # Exclude sweeps containing this part
                if part[1:] in sweep.name:
                    include_sweep = False
                    break
            else: # Include only sweeps containing this part
                if part not in sweep.name:
                    include_sweep = False
                    break
        if sweep_id: # If a specific sweep ID is provided, only include that sweep
            include_sweep = True if sweep_id == sweep.id else False
        if not include_sweep:
            continue
        
        print(f"Processing sweep: {sweep.name} (ID: {sweep.id})")
        try:
            runs = sweep.runs
        except Exception as e:
            print(f"Warning: Could not fetch runs for sweep {sweep.name} - {e}")
            continue

        sweep_results = []
        for run in runs:
            sweep_results.append({
                "run_id": run.id,
                "sweep_id": sweep.id,
                "train_acc": run.summary.get("train_acc", None),
                "val_acc": run.summary.get("val_acc", None),
                "test_acc": run.summary.get("test_acc", None),
                "avg_train_acc": run.summary.get("avg_train_acc", None),
                "avg_val_acc": run.summary.get("avg_val_acc", None),
                "avg_test_acc": run.summary.get("avg_test_acc", None),
            })

        df = pd.DataFrame(sweep_results)
        if df.empty:
            print(f"No valid runs for sweep {sweep.name}. Skipping.")
            continue
        df = df.dropna(subset=["avg_train_acc", "avg_test_acc"])
        if df.empty:
            print(f"All runs in sweep {sweep.name} missing avg_train_acc or avg_test_acc. Skipping.")
            continue
        
        # Filter runs based on avg_train_acc, then select the one with the highest avg_test_acc
        filtered_df = df[df["avg_train_acc"] == filter_accuracy]
        if filtered_df.empty:
            abs_diff = (df["avg_train_acc"] - filter_accuracy).abs()
            min_diff = abs_diff.min()
            filtered_df = df[abs_diff == min_diff]
            print(f"No matching training accuracy found. The closest is {min_diff} away.")

        best_run = filtered_df.loc[filtered_df["avg_test_acc"].idxmax()]
        best_run_dict[sweep.name] = {
            "sweep_id": best_run["sweep_id"],
            "run_id": best_run["run_id"],
            "train_acc": best_run["train_acc"],
            "val_acc": best_run["val_acc"],
            "test_acc": best_run["test_acc"],
        }

    return best_run_dict

def get_wandb_model(project_name, best_run_dict, scale_factors):
    # Retrieve the best model for each sweep and scale factor
    best_model_dict = defaultdict(dict)
    for sweep_name, sweep_res in best_run_dict.items():
        sweep_id, run_id = sweep_res["sweep_id"], sweep_res["run_id"]
        for scale in scale_factors:
            model_path = f"/path/to/model/{project_name}/sweep{sweep_id}_run{run_id}_scale{scale}.pth"
            try:
                model = torch.load(model_path, map_location="cpu", weights_only=False)
                best_model_dict[sweep_name][scale] = model
            except Exception as e:
                print(f"Skipping sweep '{sweep_name}' (file {model_path}) due to error: {e}")
    return best_model_dict

## KTA
def kernel_target_alignment(
    K: torch.Tensor,
    L: torch.Tensor,
    centered: bool = True,
    eps: float = 1e-8
):
    def _center_gram(K: torch.Tensor) -> torch.Tensor:
        row_mean = K.mean(dim=1, keepdim=True)
        col_mean = K.mean(dim=0, keepdim=True)
        total_mean = K.mean()
        return K - row_mean - col_mean + total_mean
    if centered:
        K = _center_gram(K)
        L = _center_gram(L)
    
    inner = torch.sum(K * L)
    norm_K = torch.norm(K, p='fro')
    norm_L = torch.norm(L, p='fro')
    return inner / (norm_K * norm_L + eps)

## seed
def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

## data
class AddPosToFeatures(BaseTransform):
    """Graph MNIST operation: Appends 2D coordinates (`pos`) to node features (`x`)."""
    def forward(self, data):
        data.x = torch.cat([data.x, data.pos], dim=1) 
        return data

## train/test for superpixels
def train_sp(model, optimizer, train_loader, device):
    """ Train model using batched graph data. """
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test_sp(model, loader, device):
    """ Evaluate model performance. """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = out.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

## train/test for superpixels + PE
def train_pe(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test_pe(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
            pred = out.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

## train/test for images
def train_image(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_image(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy