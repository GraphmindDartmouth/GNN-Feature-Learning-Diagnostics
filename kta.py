import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as ImDataloader
from torchvision import transforms
from torchvision import datasets as Imdatasets
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
import matplotlib.pyplot as plt
from utils import get_wandb_res, get_wandb_model, kernel_target_alignment, AddPosToFeatures
from models import SimpleNN, GPS, GraphTransformer, SGC, GCN, TokenGT, spMLP, spGCN, spGCN2, spGAT, spSAGE, peMLP, spADGN

DEFAULT_SCALES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_BS = 1000
DEFAULT_N_CLASSES = 10
DEFAULT_DATA_ROOT = "/path/to/data"
DEFAULT_DATA_ROOT_LAPPE = "/path/to/data_lappe"
DEFAULT_DATA_ROOT_RWPE = "/path/to/data_rwpe"
DEFAULT_OUTPUT_DIR = "./logs/kta"

ATTN_BASES = {"tokengt", "gt", "gps", "gcn+"}
SP_BASES = {"mlp", "gcn", "gcn2", "gat", "sage", "sgc", "adgn"}
PE_MLP_BASES = {"mlp_rwpe", "mlp_lappe"}
IMAGE_BASES = {"mlp_image", "mlp_image_sp"}
ALL_BASES = sorted(ATTN_BASES | SP_BASES | PE_MLP_BASES | IMAGE_BASES)

WANDB_MODEL_KEY = {
    "tokengt": "TokenGT-PE_True",
    "gt": "GT-PE_True-Edge_True",
    "gps": "GPS-PE_True-Conv_False-Attn_False",
    "gcn+": "GCN-PE_True",
    "mlp": "MLP",
    "gcn": "GCN",
    "gcn2": "GCN2",
    "gat": "GAT",
    "sage": "SAGE",
    "sgc": "SGC",
    "adgn": "ADGN",
    "mlp_lappe": "MLP_LAPPE",
    "mlp_rwpe": "MLP_RWPE",
    "mlp_image": "mlp-sp_False",
    "mlp_image_sp": "mlp-sp_True",
}
# ------------------------------
# Helper functions
# ------------------------------
def kernel_change_metrics(k_init, k_final, label):
    init_kta = kernel_target_alignment(k_init, label)
    kta = kernel_target_alignment(k_final, label)
    return kta - init_kta


def build_models(base, best_model_dict, scale, device, num_classes):
    ## NOTE: ensure init_model configurations match those used in the original training runs for the best models in trainer.py, otherwise the KTA comparison may not be meaningful
    if base == "tokengt":
        init_model = TokenGT(channels=128, num_layers=2, num_classes=num_classes).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "gt":
        init_model = GraphTransformer(channels=128, pe_dim=16, num_layers=2, num_classes=num_classes, use_edge=False).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "gps":
        init_model = GPS(channels=128, pe_dim=16, num_layers=2, num_classes=num_classes, attn_type="multihead").to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "mlp":
        init_model = spMLP(d=3, h=512, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "gcn":
        init_model = spGCN(d=3, h=512, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "gcn2":
        init_model = spGCN2(d=3, h=512, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "gat":
        init_model = spGAT(d=3, h=128, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "sage":
        init_model = spSAGE(d=3, h=256, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "sgc":
        init_model = SGC(d=3, h=8192, c=num_classes, l=5).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "mlp_lappe":
        init_model = peMLP(d=3, h=256, p=32, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "mlp_rwpe":
        init_model = peMLP(d=3, h=256, p=32, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "mlp_image":
        init_model = SimpleNN().to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "mlp_image_sp":
        init_model = SimpleNN().to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "gcn+":
        init_model = GCN(channels=128, pe_dim=16, num_layers=2, num_classes=num_classes).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    elif base == "adgn":
        init_model = spADGN(d=3, h=256, c=num_classes, l=4).to(device)
        model = best_model_dict[WANDB_MODEL_KEY[base]][scale].to(device)
    else:
        raise ValueError(f"Unknown base: {base}")

    with torch.no_grad():
        if base == "tokengt":
            for layer in init_model.layers:
                layer.attn.in_proj_weight *= scale
        elif base == "gt":
            for layer in init_model.layers:
                layer.lin_q.weight *= scale
                layer.lin_k.weight *= scale
                layer.lin_v.weight *= scale
                layer.lin_out.weight *= scale
        elif base == "gps":
            for conv in init_model.convs:
                conv.conv.lin.weight *= scale
                conv.attn.in_proj_weight *= scale
        elif base == "mlp":
            for layer in init_model.hidden_layers:
                layer.weight *= scale
        elif base == "gcn":
            for layer in init_model.hidden_layers:
                layer.lin.weight *= scale
        elif base == "gcn2":
            for layer in init_model.hidden_layers:
                layer.weight1 *= scale
        elif base == "gat":
            for layer in init_model.hidden_layers:
                layer.lin.weight *= scale
        elif base == "sage":
            for layer in init_model.hidden_layers:
                layer.lin_l.weight *= scale
                layer.lin_r.weight *= scale
        elif base == "sgc":
            init_model.conv.lin.weight *= scale
        elif base in {"mlp_lappe", "mlp_rwpe"}:
            for layer in init_model.hidden_layers:
                layer.weight *= scale
        elif base in {"mlp_image", "mlp_image_sp"}:
            init_model.layer1.weight *= scale
        elif base == "gcn+":
            for conv in init_model.convs:
                conv.conv.lin.weight *= scale
        elif base == "adgn":
            for layer in init_model.hidden_layers:
                layer.W *= scale
                layer.bias *= scale

    if base in ATTN_BASES:
        init_model.out = nn.Identity()
        model.out = nn.Identity()
    elif base in SP_BASES or base in PE_MLP_BASES:
        init_model.final_layer = nn.Identity()
        model.final_layer = nn.Identity()
    elif base in IMAGE_BASES:
        init_model.layer2 = nn.Identity()
        model.layer2 = nn.Identity()

    return init_model, model


def load_datasets(base, data_root, data_root_lappe, data_root_rwpe):
    if base == "mlp_image":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = Imdatasets.MNIST(root=data_root, train=True, transform=transform, download=True)
        test_dataset = Imdatasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    elif base == "mlp_image_sp":
        train_data, train_labels = torch.load(os.path.join(data_root, "mnist_train_superpixel.pt"), map_location="cpu", weights_only=False)
        test_data, test_labels = torch.load(os.path.join(data_root, "mnist_test_superpixel.pt"), map_location="cpu", weights_only=False)
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
    elif base in SP_BASES:
        train_dataset = datasets.GNNBenchmarkDataset(root=data_root, name="MNIST", split="train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root=data_root, name="MNIST", split="test", transform=AddPosToFeatures())
    elif base == "mlp_lappe":
        train_dataset = datasets.GNNBenchmarkDataset(root=data_root_lappe, name="MNIST", split="train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root=data_root_lappe, name="MNIST", split="test", transform=AddPosToFeatures())
    elif base == "mlp_rwpe":
        train_dataset = datasets.GNNBenchmarkDataset(root=data_root_rwpe, name="MNIST", split="train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root=data_root_rwpe, name="MNIST", split="test", transform=AddPosToFeatures())
    else:
        train_dataset = datasets.GNNBenchmarkDataset(root=data_root_lappe, name="MNIST", split="train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root=data_root_lappe, name="MNIST", split="test", transform=AddPosToFeatures())

    return train_dataset, test_dataset


def extract_representations(model, dataset, base, batch_size, device, num_classes):
    model.eval()
    rep_list, label_list = [], []
    with torch.no_grad():
        if base in IMAGE_BASES:
            loader = ImDataloader(dataset=dataset, batch_size=batch_size, shuffle=False)
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                rep = model(images)
                rep_list.append(rep)
                label_oh = F.one_hot(labels, num_classes=num_classes)
                label_list.append(label_oh)
        else:
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            for batch in loader:
                batch = batch.to(device)
                if base in SP_BASES:
                    rep = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    rep = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                rep_list.append(rep)
                labels = batch.y.to(device)
                label_oh = F.one_hot(labels, num_classes=num_classes)
                label_list.append(label_oh)
    return torch.cat(rep_list, dim=0).float(), torch.cat(label_list, dim=0).float()


def main(best_model_dict, scale, base, device, batch_size, num_classes, data_root, data_root_lappe, data_root_rwpe):
    init_model, model = build_models(base, best_model_dict, scale, device, num_classes)
    _, test_dataset = load_datasets(base, data_root, data_root_lappe, data_root_rwpe)

    rep_tensor, label_tensor = extract_representations(init_model, test_dataset, base, batch_size, device, num_classes)
    K0, L = rep_tensor @ rep_tensor.t(), label_tensor @ label_tensor.t()
    assert K0.size(0) == K0.size(1) == L.size(0) == L.size(1)

    rep_tensor, label_tensor = extract_representations(model, test_dataset, base, batch_size, device, num_classes)
    K, L = rep_tensor @ rep_tensor.t(), label_tensor @ label_tensor.t()
    assert K.size(0) == K.size(1) == L.size(0) == L.size(1)

    kta_diff = kernel_change_metrics(K0, K, L).item()
    return kta_diff


def plot_series(scale_list, values, ylabel, path, ylim=None):
    plt.figure()
    plt.plot(scale_list, values, linestyle="-", linewidth=4)
    plt.xscale("log")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, axis="y")
    plt.xlabel(r"$\alpha$", fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.tick_params(axis="both", labelsize=25)
    plt.savefig(path, bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, choices=ALL_BASES)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    base = args.base
    scale_list = DEFAULT_SCALES

    print(f"Running KTA analysis for base {base}...")

    if base in ATTN_BASES:
        project_name = "attn"
    elif base in SP_BASES:
        project_name = "mpnn"
    elif base in PE_MLP_BASES:
        project_name = "mpnn_pe"
    elif base in IMAGE_BASES:
        project_name = "mlp"
    else:
        raise ValueError(f"Unsupported base: {base}")

    best_run_dict = get_wandb_res(project_name) # Includes the acc of best runs for all sweeps in the project, which will be used to identify the corresponding model files for KTA analysis
    best_model_dict = get_wandb_model(project_name, best_run_dict, scale_list) # Obtain the best models for all sweeps and scales, which will be used for KTA analysis

    kta_diff_list = []
    for scale in scale_list:
        kta_diff = main(
            best_model_dict,
            scale,
            base,
            device,
            DEFAULT_BS,
            DEFAULT_N_CLASSES,
            DEFAULT_DATA_ROOT,
            DEFAULT_DATA_ROOT_LAPPE,
            DEFAULT_DATA_ROOT_RWPE,
        )
        kta_diff_list.append(kta_diff)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    plot_series(scale_list, kta_diff_list, r"$\Delta$ KTA", os.path.join(DEFAULT_OUTPUT_DIR, f"kta_diff_{base}.png"), ylim=(0, 0.55))
