import pickle
import argparse
import copy
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as ImDataloader
from torchvision import transforms
from torchvision import datasets as Imdatasets
from models import SimpleNN, GPS, GraphTransformer, TokenGT, SGC, GCN, spMLP, spGCN, spGCN2, spGAT, spSAGE, peMLP, spADGN
from utils import train_sp, test_sp, train_pe, test_pe, train_image, test_image, set_seed, kernel_target_alignment, AddPosToFeatures
from torch_geometric.loader import DataLoader
from torch_geometric import datasets

set_seed(42)
scale_factors = [0.1, 1, 10]
bs = 64
epochs = 500
n_classes = 10
train_dict, test_dict, kta_dict = defaultdict(list), defaultdict(list), defaultdict(list)

SP_BASES = {"mlp", "gcn", "gcn2", "gat", "sage", "sgc", "adgn"}
PE_MLP_BASES = {"mlp_rwpe", "mlp_lappe"}
IMAGE_BASES = {"mlp_image", "mlp_image_sp"}

def main(scale, base, train_loader, test_loader, lr, device, width=256, pe_dim=32, layers=3):
    # NOTE: model configuration may need to be adjusted
    if base == "gt":
        model = GraphTransformer(channels=width, pe_dim=pe_dim, num_layers=layers, num_classes=n_classes, use_edge=False).to(device)
    elif base == "tokengt":
        model = TokenGT(channels=width, pe_dim=pe_dim, num_layers=layers, num_classes=n_classes).to(device)
    elif base == "gps":
        model = GPS(channels=width, pe_dim=pe_dim, num_layers=layers, num_classes=n_classes, attn_type='multihead').to(device)
    elif base == "mlp":
        model = spMLP(d = 3, h = width, c = n_classes, l = layers).to(device)
    elif base == "gcn":
        model = spGCN(d = 3, h = width, c = n_classes, l = layers).to(device)
    elif base == "gcn2":
        model = spGCN2(d = 3, h = width, c = n_classes, l = layers).to(device)
    elif base == "gat":
        model = spGAT(d = 3, h = width, c = n_classes, l = layers).to(device)
    elif base == "sage":
        model = spSAGE(d = 3, h = width, c = n_classes, l = layers).to(device)
    elif base == "sgc":
        model = SGC(d = 3, h = 8192, c = n_classes, l = 5).to(device)
    elif base == "mlp_rwpe":
        model = peMLP(d = 3, h = width, p = pe_dim, c = n_classes, l = layers).to(device)
    elif base == "mlp_lappe":
        model = peMLP(d = 3, h = width, p = pe_dim, c = n_classes, l = layers).to(device)
    elif base == "mlp_image":
        model = SimpleNN().to(device)
    elif base == "mlp_image_sp":
        model = SimpleNN().to(device)
    elif base == "gcn+":
        model = GCN(channels=width, pe_dim=pe_dim, num_layers=layers, num_classes=n_classes).to(device)
    elif base == "adgn":
        model = spADGN(d = 3, h = width, c = n_classes, l = layers).to(device)
    else:
        raise ValueError(f"Unsupported base: {base}")

    with torch.no_grad():
        if base == "gt":
            for layer in model.layers:
                layer.lin_q.weight *= scale
                layer.lin_k.weight *= scale
                layer.lin_v.weight *= scale
                layer.lin_out.weight *= scale
        elif base == "tokengt":
            for layer in model.layers:
                layer.attn.in_proj_weight *= scale
        elif base == "gps":
            for conv in model.convs:
                conv.conv.lin.weight *= scale
                conv.attn.in_proj_weight *= scale
        elif base == "mlp":
            for layer in model.hidden_layers:
                layer.weight *= scale
        elif base == "gcn":
            for layer in model.hidden_layers:
                layer.lin.weight *= scale
        elif base == "gcn2":
            for layer in model.hidden_layers:
                layer.weight1 *= scale
        elif base == "gat":
            for layer in model.hidden_layers:
                layer.lin.weight *= scale
        elif base == "sage":
            for layer in model.hidden_layers:
                layer.lin_l.weight *= scale
                layer.lin_r.weight *= scale
        elif base == "sgc":
            model.conv.lin.weight *= scale
        elif base == "mlp_rwpe":
            for layer in model.hidden_layers:
                layer.weight *= scale
        elif base == "mlp_lappe":
            for layer in model.hidden_layers:
                layer.weight *= scale
        elif base == "mlp_image":
            model.layer1.weight *= scale
        elif base == "mlp_image_sp":
            model.layer1.weight *= scale
        elif base == "gcn+":
            for conv in model.convs:
                conv.conv.lin.weight *= scale
        elif base == "adgn":
            for layer in model.hidden_layers:
                layer.W *= scale
                layer.bias *= scale
    
    # NOTE: Adjust learning rate based on scale and model type
    lr_scale = lr * (0.5 + 0.5 * scale**(-1))
    if base in IMAGE_BASES:
        criterion = nn.CrossEntropyLoss()
    if base in IMAGE_BASES:
        optimizer = optim.SGD(model.parameters(), lr=lr_scale)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_scale)
    
    ## Training dynamics
    for _ in range(epochs):
        rep_list, label_list = [], []
        if base in IMAGE_BASES:
            ## Training
            _ = train_image(model, train_loader, criterion, optimizer, device)
            ## Acc
            train_acc = test_image(model, train_loader, device)
            test_acc = test_image(model, test_loader, device)
            ## KTA
            model_copy = copy.deepcopy(model)
            model_copy.layer2 = nn.Identity()
            model_copy.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    rep = model_copy(images)
                    rep_list.append(rep)
                    label_oh = F.one_hot(labels, num_classes=n_classes)
                    label_list.append(label_oh)
        elif base in SP_BASES:
            ## Training
            _ = train_sp(model, optimizer, train_loader, device)
            ## Acc
            train_acc = test_sp(model, train_loader, device)
            test_acc = test_sp(model, test_loader, device)
            ## KTA
            model_copy = copy.deepcopy(model)
            model_copy.final_layer = nn.Identity()
            model_copy.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    rep = model_copy(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    rep_list.append(rep)
                    labels = batch.y.to(device)
                    label_oh = F.one_hot(labels, num_classes=n_classes)
                    label_list.append(label_oh)
        else:
            ## Training
            _ = train_pe(model, optimizer, train_loader, device)
            ## Acc
            train_acc = test_pe(model, train_loader, device)
            test_acc = test_pe(model, test_loader, device)
            ## KTA
            model_copy = copy.deepcopy(model)
            if base in PE_MLP_BASES:
                model_copy.final_layer = nn.Identity()
            else:
                model_copy.out = nn.Identity()
            model_copy.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    rep = model_copy(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                    rep_list.append(rep)
                    labels = batch.y.to(device)
                    label_oh = F.one_hot(labels, num_classes=n_classes)
                    label_list.append(label_oh)

        train_dict[scale].append(train_acc)
        test_dict[scale].append(test_acc)
        del model_copy
        
        rep_tensor, label_tensor = torch.cat(rep_list, dim=0).float(), torch.cat(label_list, dim=0).float()
        K, L = rep_tensor @ rep_tensor.t(), label_tensor @ label_tensor.t()
        assert K.size(0) == K.size(1) == L.size(0) == L.size(1)
        # kta
        kta = kernel_target_alignment(K, L)
        kta_dict[scale].append(kta.item())
    
    del K, L, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, choices=[
        "gt",
        "tokengt",
        "gps",
        "mlp",
        "gcn",
        "gcn2",
        "gat",
        "sage",
        "sgc",
        "mlp_rwpe",
        "mlp_lappe",
        "mlp_image",
        "mlp_image_sp",
        "gcn+",
        "adgn",
    ])
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--pe_dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()
    base = args.base
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Running with base {base} on device {args.device}")

    # NOTE: learning rate may need to be adjusted based on model type
    lr = 3e-4
    if base == "mlp_image": # original images
        transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
        train_dataset = Imdatasets.MNIST(root='/path/to/data', train=True,
                                    transform=transform, download=True)
        test_dataset = Imdatasets.MNIST(root='/path/to/data', train=False,
                                    transform=transform, download=True)
    elif base == "mlp_image_sp": # superpixel images (processed in data.ipynb)
        train_data, train_labels = torch.load("/path/to/data/mnist_train_superpixel.pt", map_location="cpu", weights_only=False)
        test_data, test_labels = torch.load("/path/to/data/mnist_test_superpixel.pt", map_location="cpu", weights_only=False)
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
    elif base in SP_BASES:
        train_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data", name = "MNIST", split = "train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data", name = "MNIST", split = "test", transform=AddPosToFeatures())
    elif base == "mlp_rwpe":
        train_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_rwpe", name = "MNIST", split = "train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_rwpe", name = "MNIST", split = "test", transform=AddPosToFeatures())
    else:
        train_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_lappe", name = "MNIST", split = "train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_lappe", name = "MNIST", split = "test", transform=AddPosToFeatures())
    
    if base in IMAGE_BASES:
        train_loader = ImDataloader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = ImDataloader(dataset=test_dataset, batch_size=bs, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    save_dict_train = "/path/to/eval/train_" + base + ".pkl"
    save_dict_test = "/path/to/eval/test_" + base + ".pkl"
    save_dict_kta = "/path/to/eval/kta_" + base + ".pkl"
    os.makedirs("/path/to/eval", exist_ok=True)

    for scale in scale_factors:
        main(scale, base, train_loader, test_loader, lr, device, args.width, args.pe_dim, args.layers)
    with open(save_dict_train, "wb") as f:
        pickle.dump(train_dict, f)
    with open(save_dict_test, "wb") as f:
        pickle.dump(test_dict, f)
    with open(save_dict_kta, "wb") as f:
        pickle.dump(kta_dict, f)

    with open(save_dict_train, "rb") as f:
        train_dict = pickle.load(f)
    with open(save_dict_test, "rb") as f:
        test_dict = pickle.load(f)
    with open(save_dict_kta, "rb") as f:
        kta_dict = pickle.load(f)

    save_fig_train = "./logs/eval/train_" + base + ".png"
    save_fig_test = "./logs/eval/test_" + base + ".png"
    save_fig_kta = "./logs/eval/kta_" + base + ".png"
    os.makedirs("./logs/eval", exist_ok=True)

    def plot_series(save_path, series_dict, ylabel, ylim, legend_loc):
        plt.figure()
        plt.gca().set_prop_cycle(color=["#1f77b4", "#ff7f0e", "#9467bd"])
        for key, values in series_dict.items():
            plt.plot(range(epochs), values, label=fr"$\alpha={key}$", linewidth=3.5)
        plt.legend(loc=legend_loc, fontsize=25)
        plt.grid(True, axis="y")
        plt.ylim(*ylim)
        plt.xticks([])
        plt.xlabel(r"$t$", fontsize=28)
        plt.ylabel(ylabel, fontsize=28)
        plt.tick_params(axis="both", labelsize=28)
        plt.savefig(save_path, bbox_inches="tight")

    legend_loc = "upper right" if base in ["mlp", "sgc"] else "lower right"
    plot_series(save_fig_train, train_dict, "Training Accuracy", (0.3, 1), legend_loc)
    plot_series(save_fig_test, test_dict, "Test Accuracy", (0.3, 1), legend_loc)
    plot_series(save_fig_kta, kta_dict, "KTA", (.1, .65), "upper right")
