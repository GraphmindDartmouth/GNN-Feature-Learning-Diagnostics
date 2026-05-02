import os
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from torchvision import datasets as Imdatasets, transforms
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric import datasets

from models import (
    SimpleNN,
    SGC,
    TokenGT,
    GPS,
    GCN,
    GraphTransformer,
    spMLP,
    spGCN,
    spGAT,
    spSAGE,
    spGCN2,
    spADGN,
    peMLP,
)
from utils import train_sp, test_sp, train_pe, test_pe, set_seed, AddPosToFeatures

PROJECT_MPNN = "mpnn"
PROJECT_ATTN = "attn"
PROJECT_PE = "mpnn_pe"
PROJECT_IMAGE = "mlp"
SCALE_FACTORS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

set_seed(42)


def run_mpnn(args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    model_dict = {
        'SGC': SGC,
        'MLP': spMLP,
        'GAT': spGAT,
        'SAGE': spSAGE,
        'GCN': spGCN,
        'GCN2': spGCN2,
        'ADGN': spADGN,
    }
    train_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data", name="MNIST", split="train", transform=AddPosToFeatures())
    test_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data", name="MNIST", split="test", transform=AddPosToFeatures())

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        # NOTE: hyperparameters need to be searched per model and scale
        lr_val = 3e-4
        hd_val = 256
        sweep_configuration = {
            "method": "grid",
            "name": f"{args.model_name}",
            "metric": {"goal": "maximize", "name": "avg_train_acc"},
            "parameters": {
                "lr": {"value": lr_val},
                "hd": {"value": hd_val},
                "bs": {"value": 64},
                "L": {"value": args.layers},
                "epochs": {"value": args.epochs},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_MPNN)

    def main():
        train_dict, test_dict = {}, {}
        run = wandb.init(config={"model": args.model_name}, reinit=True)
        run_id = wandb.run.id
        config = wandb.config
        lr, hd, bs, l, epochs = config.lr, config.hd, config.bs, config.L, config.epochs
        train_loader = GeoDataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = GeoDataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

        for scale in SCALE_FACTORS:
            model = model_dict[args.model_name](d=3, h=hd, c=10, l=l).to(device)
            with torch.no_grad():
                if args.model_name == "SGC":
                    model.conv.lin.weight *= scale
                elif args.model_name == "MLP":
                    for layer in model.hidden_layers:
                        layer.weight *= scale
                elif args.model_name in ["GCN", "GAT"]:
                    for layer in model.hidden_layers:
                        layer.lin.weight *= scale
                elif args.model_name == "GCN2":
                    for layer in model.hidden_layers:
                        layer.weight1 *= scale
                elif args.model_name == "SAGE":
                    for layer in model.hidden_layers:
                        layer.lin_l.weight *= scale
                        layer.lin_r.weight *= scale
                elif args.model_name == "ADGN":
                    for layer in model.hidden_layers:
                        layer.W *= scale
                        layer.bias *= scale

            # NOTE: learning rate scaling rule may need to be adjusted per model
            lr_scale = lr * (0.5 + 0.5 * scale**(-0.8))
            optimizer = optim.Adam(model.parameters(), lr=lr_scale)
            for _ in range(1, epochs + 1):
                _ = train_sp(model, optimizer, train_loader, device)
            train_acc = test_sp(model, train_loader, device)
            test_acc = test_sp(model, test_loader, device)
            train_dict[scale] = train_acc
            test_dict[scale] = test_acc

            model_dir = f"/path/to/model/{PROJECT_MPNN}"
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"sweep{sweep_id}_run{run_id}_scale{scale}.pth"
            model_path = os.path.join(model_dir, model_filename)
            torch.save(model, model_path)
            artifact = wandb.Artifact(
                name=f"sweep{sweep_id}_run{run_id}_scale{scale}",
                type="model",
                metadata={
                    "sweep_id": sweep_id,
                    "run_id": run_id,
                    "scale": scale,
                    "model_name": args.model_name,
                },
            )
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)
            del model

        wandb.log(
            {
                "train_acc": train_dict,
                "test_acc": test_dict,
                "avg_train_acc": np.mean(list(train_dict.values())),
                "avg_test_acc": np.mean(list(test_dict.values())),
            }
        )
        run.finish()

    wandb.agent(sweep_id, function=main, count=1, project=PROJECT_MPNN)


def run_attn(args):
    if args.conv_only and args.attn_only:
        raise ValueError("Can only keep either conv or attn.")

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    train_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data_lappe", name="MNIST", split="train", transform=AddPosToFeatures())
    test_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data_lappe", name="MNIST", split="test", transform=AddPosToFeatures())
    if args.no_pe:
        train_pe_data = train_dataset.data.pe
        train_dataset.data.pe = torch.randn_like(train_pe_data)
        test_pe_data = test_dataset.data.pe
        test_dataset.data.pe = torch.randn_like(test_pe_data)

    sweep_config = {"model": args.model_name, "PE": not args.no_pe}
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        if args.model_name == "GPS":
            sweep_name = f"GPS-PE_{not args.no_pe}-Conv_{args.conv_only}-Attn_{args.attn_only}"
        elif args.model_name == "GT":
            sweep_name = f"GT-PE_{not args.no_pe}-Edge_{not args.no_edge}"
        elif args.model_name == "GCN":
            sweep_name = f"GCN-PE_{not args.no_pe}"
        elif args.model_name == "TokenGT":
            sweep_name = f"TokenGT-PE_{not args.no_pe}"
        # NOTE: hyperparameters need to be searched per model and scale
        lr_val = 2e-4
        hd_val = 128
        sweep_configuration = {
            "method": "grid",
            "name": sweep_name,
            "metric": {"goal": "maximize", "name": "avg_train_acc"},
            "parameters": {
                "lr": {"value": lr_val},
                "hd": {"value": hd_val},
                "bs": {"value": 64},
                "L": {"value": args.layers},
                "epochs": {"value": args.epochs},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_ATTN)
        if args.model_name == "GPS":
            sweep_config.update({"Conv": args.conv_only, "Attn": args.attn_only})
        elif args.model_name == "GT":
            sweep_config.update({"Edge": not args.no_edge})

    def main():
        train_dict, test_dict = {}, {}
        run = wandb.init(config=sweep_config, reinit=True)
        run_id = wandb.run.id
        config = wandb.config
        lr, hd, bs, l, epochs = config.lr, config.hd, config.bs, config.L, config.epochs
        train_loader = GeoDataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = GeoDataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

        for scale in SCALE_FACTORS:
            if args.model_name == "GPS":
                model = GPS(channels=hd, pe_dim=hd // 8, num_layers=l, num_classes=10, attn_type='multihead', conv_only=args.conv_only, attn_only=args.attn_only).to(device)
                with torch.no_grad():
                    for conv in model.convs:
                        if args.conv_only:
                            conv.conv.lin.weight *= scale
                        elif args.attn_only:
                            conv.attn.in_proj_weight *= scale
                        else:
                            conv.conv.lin.weight *= scale
                            conv.attn.in_proj_weight *= scale
            elif args.model_name == "GT":
                model = GraphTransformer(channels=hd, pe_dim=hd // 8, num_layers=l, num_classes=10, use_edge=not args.no_edge).to(device)
                with torch.no_grad():
                    for layer in model.layers:
                        layer.lin_q.weight *= scale
                        layer.lin_k.weight *= scale
                        layer.lin_v.weight *= scale
                        layer.lin_out.weight *= scale
            elif args.model_name == "GCN":
                model = GCN(channels=hd, pe_dim=hd // 8, num_layers=l, num_classes=10).to(device)
                with torch.no_grad():
                    for conv in model.convs:
                        conv.conv.lin.weight *= scale
            elif args.model_name == "TokenGT":
                model = TokenGT(channels=hd, num_layers=l, num_classes=10).to(device)
                with torch.no_grad():
                    for layer in model.layers:
                        layer.attn.in_proj_weight *= scale

            # NOTE: learning rate scaling rule may need to be adjusted per model
            lr_scale = lr * (0.5 + 0.5 * scale**(-0.8))
            optimizer = optim.Adam(model.parameters(), lr=lr_scale)
            for _ in range(epochs):
                _ = train_pe(model, optimizer, train_loader, device)
            train_acc = test_pe(model, train_loader, device)
            test_acc = test_pe(model, test_loader, device)
            train_dict[scale] = train_acc
            test_dict[scale] = test_acc

            model_dir = f"/path/to/model/{PROJECT_ATTN}"
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"sweep{sweep_id}_run{run_id}_scale{scale}.pth"
            model_path = os.path.join(model_dir, model_filename)
            torch.save(model, model_path)
            artifact = wandb.Artifact(
                name=f"sweep{sweep_id}_run{run_id}_scale{scale}",
                type="model",
                metadata={
                    "sweep_id": sweep_id,
                    "run_id": run_id,
                    "scale": scale,
                    "model_name": args.model_name,
                },
            )
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)
            del model

        wandb.log(
            {
                "train_acc": train_dict,
                "test_acc": test_dict,
                "avg_train_acc": np.mean(list(train_dict.values())),
                "avg_test_acc": np.mean(list(test_dict.values())),
            }
        )
        run.finish()

    wandb.agent(sweep_id, function=main, count=1, project=PROJECT_ATTN)


def run_pe(args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if args.model_name == 'MLP_LAPPE':
        train_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data_lappe", name="MNIST", split="train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data_lappe", name="MNIST", split="test", transform=AddPosToFeatures())
    else:
        train_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data_rwpe", name="MNIST", split="train", transform=AddPosToFeatures())
        test_dataset = datasets.GNNBenchmarkDataset(root="/path/to/data_rwpe", name="MNIST", split="test", transform=AddPosToFeatures())

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        # NOTE: hyperparameters need to be searched per model and scale
        lr_val = 3e-4
        hd_val = 256
        sweep_configuration = {
            "method": "grid",
            "name": f"{args.model_name}",
            "metric": {"goal": "maximize", "name": "avg_train_acc"},
            "parameters": {
                "lr": {"value": lr_val},
                "hd": {"value": hd_val},
                "bs": {"value": 64},
                "L": {"value": args.layers},
                "epochs": {"value": args.epochs},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_PE)

    def main():
        train_dict, test_dict = {}, {}
        run = wandb.init(config={"model": args.model_name}, reinit=True)
        run_id = wandb.run.id
        config = wandb.config
        lr, hd, bs, l, epochs = config.lr, config.hd, config.bs, config.L, config.epochs
        train_loader = GeoDataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
        test_loader = GeoDataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

        for scale in SCALE_FACTORS:
            model = peMLP(d=3, h=hd, p=hd // 8, c=10, l=l).to(device)
            with torch.no_grad():
                for layer in model.hidden_layers:
                    layer.weight *= scale
            # NOTE: learning rate scaling rule may need to be adjusted per model
            lr_scale = lr * (0.64 + 0.36 * scale**(-1))
            optimizer = optim.Adam(model.parameters(), lr=lr_scale)
            for _ in range(1, epochs + 1):
                _ = train_pe(model, optimizer, train_loader, device)
            train_acc = test_pe(model, train_loader, device)
            test_acc = test_pe(model, test_loader, device)
            train_dict[scale] = train_acc
            test_dict[scale] = test_acc

            model_dir = f"/path/to/model/{PROJECT_PE}"
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"sweep{sweep_id}_run{run_id}_scale{scale}.pth"
            model_path = os.path.join(model_dir, model_filename)
            torch.save(model, model_path)
            artifact = wandb.Artifact(
                name=f"sweep{sweep_id}_run{run_id}_scale{scale}",
                type="model",
                metadata={
                    "sweep_id": sweep_id,
                    "run_id": run_id,
                    "scale": scale,
                    "model_name": args.model_name,
                },
            )
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)
            del model

        wandb.log(
            {
                "train_acc": train_dict,
                "test_acc": test_dict,
                "avg_train_acc": np.mean(list(train_dict.values())),
                "avg_test_acc": np.mean(list(test_dict.values())),
            }
        )
        run.finish()

    wandb.agent(sweep_id, function=main, count=1, project=PROJECT_PE)


def run_image(args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        lr_val = 1e-3
        sweep_configuration = {
            "method": "grid",
            "name": f"mlp-sp_{args.sp}",
            "metric": {"goal": "maximize", "name": "avg_train_acc"},
            "parameters": {
                "lr": {"value": lr_val},
                "wd": {"value": 0.0},
                "batch_size": {"value": 64},
                "epochs": {"value": args.epochs},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_IMAGE)

    def get_data(batch_size):
        if args.sp: # superpixel data
            train_data, train_labels = torch.load("/path/to/data/mnist_train_superpixel.pt", map_location="cpu", weights_only=False)
            test_data, test_labels = torch.load("/path/to/data/mnist_test_superpixel.pt", map_location="cpu", weights_only=False)
            train_dataset = TensorDataset(train_data, train_labels)
            test_dataset = TensorDataset(test_data, test_labels)
        else: # raw image data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            train_dataset = Imdatasets.MNIST(root='/path/to/data', train=True, transform=transform, download=True)
            test_dataset = Imdatasets.MNIST(root='/path/to/data', train=False, transform=transform, download=True)

        train_loader = TorchDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = TorchDataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def evaluate_image(model, loader):
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
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def main():
        train_dict, test_dict = {}, {}
        run = wandb.init(config={"sp": args.sp}, reinit=True)
        run_id = wandb.run.id
        config = wandb.config
        batch_size, epochs, lr, wd = config.batch_size, config.epochs, config.lr, config.wd
        train_loader, test_loader = get_data(batch_size)

        for scale in SCALE_FACTORS:
            criterion = nn.CrossEntropyLoss()
            model = SimpleNN().to(device)
            with torch.no_grad():
                model.layer1.weight *= scale
            lr_scale = lr * (0.5 + 0.5 * scale**(-1))
            optimizer = optim.SGD(model.parameters(), lr=lr_scale, weight_decay=wd, momentum=0)
            for _ in range(epochs):
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            train_acc = evaluate_image(model, train_loader)
            test_acc = evaluate_image(model, test_loader)
            train_dict[scale] = train_acc
            test_dict[scale] = test_acc

            model_dir = f"/path/to/model/{PROJECT_IMAGE}"
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"sweep{sweep_id}_run{run_id}_scale{scale}.pth"
            model_path = os.path.join(model_dir, model_filename)
            torch.save(model, model_path)
            artifact = wandb.Artifact(
                name=f"sweep{sweep_id}_run{run_id}_scale{scale}",
                type="model",
                metadata={
                    "sweep_id": sweep_id,
                    "run_id": run_id,
                    "scale": scale,
                },
            )
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)

        wandb.log(
            {
                "train_acc": train_dict,
                "test_acc": test_dict,
                "avg_train_acc": np.mean(list(train_dict.values())),
                "avg_test_acc": np.mean(list(test_dict.values())),
            }
        )
        run.finish()

    wandb.agent(sweep_id, function=main, count=1, project=PROJECT_IMAGE)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_mpnn = subparsers.add_parser("mpnn")
    parser_mpnn.add_argument("--model_name", type=str, required=True, choices=['SGC', 'MLP', 'GAT', 'SAGE', 'GCN', 'GCN2', 'ADGN'])
    parser_mpnn.add_argument("--epochs", type=int, required=True)
    parser_mpnn.add_argument("--layers", type=int, required=True)
    parser_mpnn.add_argument("--device", type=int, required=True)
    parser_mpnn.add_argument("--sweep_id", type=str, default=None)

    parser_attn = subparsers.add_parser("attn")
    parser_attn.add_argument("--epochs", type=int, required=True)
    parser_attn.add_argument("--layers", type=int, required=True)
    parser_attn.add_argument("--device", type=int, required=True)
    parser_attn.add_argument("--model_name", type=str, required=True, choices=["GPS", "GT", "GCN", "TokenGT"])
    parser_attn.add_argument("--no_pe", action="store_true", default=False)
    parser_attn.add_argument("--conv_only", action="store_true", default=False)
    parser_attn.add_argument("--attn_only", action="store_true", default=False)
    parser_attn.add_argument("--no_edge", action="store_true", default=False)
    parser_attn.add_argument("--sweep_id", type=str, default=None)

    parser_pe = subparsers.add_parser("pe")
    parser_pe.add_argument("--model_name", type=str, required=True, choices=['MLP_RWPE', 'MLP_LAPPE'])
    parser_pe.add_argument("--epochs", type=int, required=True)
    parser_pe.add_argument("--layers", type=int, required=True)
    parser_pe.add_argument("--device", type=int, required=True)
    parser_pe.add_argument("--sweep_id", type=str, default=None)

    parser_mlp = subparsers.add_parser("image")
    parser_mlp.add_argument("--epochs", type=int, required=True)
    parser_mlp.add_argument("--sp", action="store_true", default=False)
    parser_mlp.add_argument("--device", type=int, required=True)
    parser_mlp.add_argument("--sweep_id", type=str, default=None)

    args = parser.parse_args()
    if args.mode == "mpnn":
        run_mpnn(args)
    elif args.mode == "attn":
        run_attn(args)
    elif args.mode == "pe":
        run_pe(args)
    elif args.mode == "image":
        run_image(args)


if __name__ == "__main__":
    main()
