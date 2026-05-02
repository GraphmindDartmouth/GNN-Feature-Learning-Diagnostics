import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import GPS, GraphTransformer, TokenGT, GCN, spMLP, spGCN, spGCN2, spGAT, spSAGE, spADGN, peMLP
from utils import train_sp, test_sp, train_pe, test_pe, set_seed, kernel_target_alignment, AddPosToFeatures
from torch_geometric.loader import DataLoader
from torch_geometric import datasets

hyper = {
    "hd": 256,
    "lr_probe": 1e-3,
    "probe_epochs": 20,
    "train_epochs": 500,
}
set_seed(42)
hd = hyper["hd"]
train_epochs, probe_epochs = hyper["train_epochs"], hyper["probe_epochs"]
bs = 64
n_classes = 5
mpnn_bases = {"mlp", "gcn", "gcn2", "gat", "sage", "adgn"}
mpnn_pe_bases = {"mlp_rwpe", "mlp_lappe"}
class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    def forward(self, x):
        return self.classifier(x)

def split_mnist_by_class(dataset):
    data_0_4 = [data for data in dataset if data.y.item() <= 4]
    data_5_9 = [data for data in dataset if data.y.item() >= 5]
    return data_0_4, data_5_9

def extract_features_kta(model, loader, base, device):
    if (base in mpnn_bases) or (base in mpnn_pe_bases):
        model.final_layer = nn.Identity()
    else:
        model.out = nn.Identity()
    model.eval()
    rep_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if base in mpnn_bases:
                rep = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                rep = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
            rep_list.append(rep)
            label_oh = F.one_hot(batch.y.to(device), num_classes=n_classes)
            label_list.append(label_oh)
    return torch.cat(rep_list, dim=0).float(), torch.cat(label_list, dim=0).float()

def train(model, train_loader, test_loader, lr, base, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ## Training
    for _ in range(epochs):
        if base in mpnn_bases:
            _ = train_sp(model, optimizer, train_loader, device)
        else:
            _ = train_pe(model, optimizer, train_loader, device)
    ## KTA
    model_copy = copy.deepcopy(model)
    rep_tensor, label_tensor = extract_features_kta(model_copy, test_loader, base, device)
    del model_copy
    K, L = rep_tensor @ rep_tensor.t(), label_tensor @ label_tensor.t()
    assert K.size(0) == K.size(1) == L.size(0) == L.size(1)
    kta = kernel_target_alignment(K, L).item()
    return kta

def extract_features_probe(model, loader, base, device):
    if (base in mpnn_bases) or (base in mpnn_pe_bases):
        model.final_layer = nn.Identity()
    else:
        model.out = nn.Identity()
    model.eval()
    all_x, all_y = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if base in mpnn_bases:
                x = model(data.x, data.edge_index, data.edge_attr, data.batch)
            else:
                x = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
            all_x.append(x)
            all_y.append(data.y - 5)  # reindex 5–9 to 0–4
    return torch.cat(all_x), torch.cat(all_y)

def probe(prober, model, train_loader, test_loader, lr, base, epochs, device):
    X_train, y_train = extract_features_probe(model, train_loader, base, device)
    X_test, y_test = extract_features_probe(model, test_loader, base, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(prober.parameters(), lr=lr)
    best_acc_test = 0
    for _ in range(epochs):
        # Train
        prober.train()
        optimizer.zero_grad()
        out = prober(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        # Evaluate
        prober.eval()
        with torch.no_grad():
            pred_test = prober(X_test).argmax(dim=-1)
            acc_test = (pred_test == y_test).float().mean().item()
        if acc_test > best_acc_test:
            best_acc_test = acc_test
    return best_acc_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, choices=[
        # list is extended as new models are added
        "gt",
        "tokengt",
        "gps",
        "mlp",
        "gcn",
        "gcn2",
        "gat",
        "sage",
        "mlp_rwpe",
        "mlp_lappe",
        "gcn+",
        "adgn",
    ])
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    base = args.base
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if base == "gt":
        model = GraphTransformer(channels=hd, pe_dim=hd//8, num_layers=2, num_classes=n_classes, use_edge=False).to(device)
    elif base == "tokengt":
        model = TokenGT(channels=hd, num_layers=2, num_classes=n_classes).to(device)
    elif base == "gps":
        model = GPS(channels=hd, pe_dim=hd//8, num_layers=2, num_classes=n_classes, attn_type='multihead').to(device)
    elif base == "mlp":
        model = spMLP(d = 3, h = hd, c = n_classes, l = 4).to(device)
    elif base == "gcn":
        model = spGCN(d = 3, h = hd, c = n_classes, l = 4).to(device)
    elif base == "gcn2":
        model = spGCN2(d = 3, h = hd, c = n_classes, l = 4).to(device)
    elif base == "gat":
        model = spGAT(d = 3, h = hd, c = n_classes, l = 4).to(device)
    elif base == "sage":
        model = spSAGE(d = 3, h = hd, c = n_classes, l = 4).to(device)
    elif base == "gcn+":
        model = GCN(channels=hd, pe_dim=hd//8, num_layers=2, num_classes=n_classes).to(device)
    elif base == "adgn":
        model = spADGN(d = 3, h = hd, c = n_classes, l = 4).to(device)
    elif base == "mlp_rwpe":
        model = peMLP(d = 3, h = hd, p = hd//8, c = n_classes, l = 4).to(device)
    elif base == "mlp_lappe":
        model = peMLP(d = 3, h = hd, p = hd//8, c = n_classes, l = 4).to(device)
    else:
        raise ValueError(f"Unsupported base: {base}")

    if base in mpnn_bases:
        train_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data", name = "MNIST", split = "train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data", name = "MNIST", split = "test", transform=AddPosToFeatures())
    elif base == "mlp_rwpe":
        train_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_rwpe", name = "MNIST", split = "train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_rwpe", name = "MNIST", split = "test", transform=AddPosToFeatures())
    else:
        train_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_lappe", name = "MNIST", split = "train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root = "/path/to/data_lappe", name = "MNIST", split = "test", transform=AddPosToFeatures())
    train_0_4, train_5_9 = split_mnist_by_class(train_dataset)
    test_0_4, test_5_9 = split_mnist_by_class(test_dataset)

    # NOTE: learning rate may need to be tuned for different models
    lr = 3e-4
    ## Train
    train_loader = DataLoader(dataset=train_0_4, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset=test_0_4, batch_size=bs, shuffle=False)
    kta = train(model, train_loader, test_loader, lr, base, train_epochs, device)
    if base in mpnn_bases:
        source_test_acc = test_sp(model, test_loader, device)
    else:
        source_test_acc = test_pe(model, test_loader, device)

    ## Probe
    train_loader = DataLoader(dataset=train_5_9, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset=test_5_9, batch_size=bs, shuffle=False)
    prober = LinearProbe(feature_dim = hd, num_classes = n_classes).to(device)
    target_test_acc = probe(prober, model, train_loader, test_loader, hyper["lr_probe"], base, probe_epochs, device)
    print(f"Base {base}: Source Test Acc = {source_test_acc:.4f} | Target Test Acc = {target_test_acc:.4f} | KTA = {kta:.4f}")
