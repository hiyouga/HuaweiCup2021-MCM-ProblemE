import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


ffn = nn.Sequential(
    nn.Linear(60, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 5)
)


class MyDataset(Dataset):

    def __init__(self, data, label):
        self._data = data
        self._label = label

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':

    train_data, train_label, val_data, val_label = list(), list(), list(), list()
    for C in ['正常', '异常']:
        val_idx = random.sample(list(range(324)), 32)
        for n in range(324):
            if n in val_idx:
                val_data.append(np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.feat.txt"), dtype=np.float32))
                val_label.append(np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.label.txt"), dtype=np.int64))
            else:
                train_data.append(np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.feat.txt"), dtype=np.float32))
                train_label.append(np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.label.txt"), dtype=np.int64))
    train_data, train_label = np.concatenate(train_data), np.concatenate(train_label)
    val_data, val_label = np.concatenate(val_data), np.concatenate(val_label)
    ''' normalize '''
    mean, std = np.mean(train_data, axis=0), np.std(train_data, axis=0)
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    ''' build dataset '''
    trainset = MyDataset(train_data, train_label)
    valset = MyDataset(val_data, val_label)
    train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=ffn.parameters(), lr=1e-2, weight_decay=1e-5)
    ''' model training '''
    best_val_acc, best_model = 0, None
    train_losses, train_accs, val_losses, val_accs = list(), list(), list(), list()
    for epoch in range(100):
        ''' train '''
        ffn.train()
        train_loss, n_correct, n_train = 0, 0, 0
        for i_batch, (inputs, targets) in enumerate(train_dataloader):
            outputs = ffn(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_train += targets.size(0)
        print(f"train loss: {train_loss / n_train:.2f}, acc: {n_correct / n_train:.2f}")
        ''' evaluate '''
        with torch.no_grad():
            ffn.eval()
            val_loss, n_correct, n_val = 0, 0, 0
            for i_batch, (inputs, targets) in enumerate(val_dataloader):
                outputs = ffn(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_val += targets.size(0)
            print(f"val loss: {val_loss / n_val:.2f}, acc: {n_correct / n_val:.2f}")
            if n_correct / n_val > best_val_acc:
                best_val_acc = n_correct / n_val
                best_model = ffn.state_dict()
    print(f"best acc: {best_val_acc:.2f}")
    state_dict = {
        'model': best_model,
        'mean': mean,
        'std': std
    }
    torch.save(state_dict, 'best_model.pt')
    ffn.load_state_dict(best_model)
    with torch.no_grad():
        ffn.eval()
        for C in ['正常', '异常']:
            for n in range(324):
                inputs = torch.from_numpy(np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.feat.txt"), dtype=np.float32))
                inputs = (inputs - mean) / std
                outputs = ffn(inputs)
                predicts = torch.argmax(outputs, dim=-1).numpy()
                np.savetxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.pred.txt"), predicts, fmt='%d', encoding='utf-8')
