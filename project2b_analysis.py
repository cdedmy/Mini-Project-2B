import os, glob, time, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
CHANNELS_14 = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6",
               "O1","O2","P7","P8","T7","T8"]

GAME_TO_LABEL = {1:0, 2:1, 3:2, 4:3}


# ----------------------------
# DATA LOADING
# ----------------------------
def read_one_csv(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if set(CHANNELS_14).issubset(df.columns):
        df = df[CHANNELS_14]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    df = df.fillna(df.mean(numeric_only=True)).fillna(0.0)
    return df


def make_windows(df, win, stride):
    X = df.to_numpy(dtype=np.float32)
    T, C = X.shape
    if T < win:
        return None

    windows = []
    for i in range(0, T - win + 1, stride):
        windows.append(X[i:i+win])

    if not windows:
        return None

    w = np.stack(windows)
    return np.transpose(w, (0, 2, 1))


def load_gameemo(data_root, win, stride):
    X_all, y_all, groups = [], [], []

    subject_dirs = sorted(
        glob.glob(os.path.join(os.path.expanduser(data_root), "(S??)"))
    )
    print("Found subject dirs:", len(subject_dirs))

    for sdir in subject_dirs:
        subj = os.path.basename(sdir).strip("()")
        csv_dir = os.path.join(sdir, "Preprocessed EEG Data", ".csv format")

        for g in [1, 2, 3, 4]:
            f = os.path.join(csv_dir, f"{subj}G{g}AllChannels.csv")
            if not os.path.exists(f):
                continue

            df = read_one_csv(f)
            wins = make_windows(df, win, stride)
            if wins is None:
                continue

            X_all.append(wins)
            y_all.append(np.full((wins.shape[0],), GAME_TO_LABEL[g], dtype=np.int64))
            groups.append(np.full((wins.shape[0],), subj, dtype=object))

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(groups)

    print("X shape:", X.shape)
    print("Unique subjects:", len(np.unique(groups)))

    return X, y, groups


# ----------------------------
# MODEL
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_layers, num_classes=4):
        super().__init__()
        layers = []
        prev = in_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total


def train_and_eval(model, train_loader, test_loader, device, epochs=6):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training time
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    # Evaluation time
    t1 = time.time()
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    eval_time = time.time() - t1

    return train_acc, test_acc, train_time, eval_time


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="~/data/GAMEEMO")
    parser.add_argument("--win", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out_csv", default="project2b_increment_table.csv")
    args = parser.parse_args()

    X, y, groups = load_gameemo(args.data_root, args.win, args.stride)

    # Subject-wise split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=123)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Train subjects:", len(np.unique(groups[train_idx])))
    print("Test subjects:", len(np.unique(groups[test_idx])))

    # Flatten + scale
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte),
                             batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Baseline + 5 increments
    configs = [
        [32],
        [64],
        [128],
        [256],
        [256, 128],
        [512, 256, 128],
    ]

    rows = []

    for i, hidden in enumerate(configs, 1):
        model = MLP(Xtr.shape[1], hidden).to(device)
        params = sum(p.numel() for p in model.parameters())

        train_acc, test_acc, train_time, eval_time = train_and_eval(
            model, train_loader, test_loader, device, args.epochs
        )

        row = {
            "increment": i,
            "hidden_layers": str(hidden),
            "params": params,
            "epochs": args.epochs,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "total_time_sec": train_time + eval_time
        }

        print(row)
        rows.append(row)

    # ----------------------------
    # Final Table
    # ----------------------------
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("\n=== FINAL COMPARATIVE TABLE ===")
    print(df.to_string(index=False))
    print("\nSaved table ->", args.out_csv)

    # ----------------------------
    # 4 Required Plots
    # ----------------------------
    plt.figure()
    plt.plot(df["increment"], df["train_acc"])
    plt.xlabel("Increment")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy vs Model Size")
    plt.savefig("plot_train_accuracy.png")
    plt.show()

    plt.figure()
    plt.plot(df["increment"], df["test_acc"])
    plt.xlabel("Increment")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Model Size")
    plt.savefig("plot_test_accuracy.png")
    plt.show()

    plt.figure()
    plt.plot(df["increment"], df["train_time_sec"])
    plt.xlabel("Increment")
    plt.ylabel("Train Time (sec)")
    plt.title("Training Runtime vs Model Size")
    plt.savefig("plot_train_runtime.png")
    plt.show()

    plt.figure()
    plt.plot(df["increment"], df["total_time_sec"])
    plt.xlabel("Increment")
    plt.ylabel("Total Runtime (sec)")
    plt.title("Total Runtime vs Model Size")
    plt.savefig("plot_total_runtime.png")
    plt.show()

    print("\nSaved plots:")
    print(" - plot_train_accuracy.png")
    print(" - plot_test_accuracy.png")
    print(" - plot_train_runtime.png")
    print(" - plot_total_runtime.png")


if __name__ == "__main__":
    main()
