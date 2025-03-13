# 围棋对弈数据集（PyTorch神经网络训练专用）

## 概述

本数据集包含从高质量SGF棋谱文件中提取的围棋对局位置，专为训练神经网络而设计。位置根据对局质量分为三个强度类别，每个类别提取了约1000个样本。

## 数据集统计

- **处理的SGF文件总数**：根据原始文件数量而定
- **有效SGF文件**：通过质量筛选的文件数
- **总位置数**：大约3000个（每个强度类别约1000个）
- **处理时间**：取决于实际运行耗时

## 强度类别

数据集根据棋谱质量分为三个强度类别：

- **标准级别** (Quality 80-85)：业余高段和职业初段对局
- **强力级别** (Quality 86-92)：职业中高段对局
- **精英级别** (Quality 93-100)：顶尖职业选手对局

## 目录结构

```
dataset/
├── train/
│   ├── boards.pt      # 棋盘状态张量 (N, C, H, W)
│   ├── moves.pt       # 着法标签 (N,)
│   ├── colors.pt      # 棋手颜色 (N,)
│   └── metadata.json  # 附加信息
├── val/
│   ├── boards.pt
│   ├── moves.pt
│   ├── colors.pt
│   └── metadata.json
├── test/
│   ├── boards.pt
│   ├── moves.pt
│   ├── colors.pt
│   └── metadata.json
├── stats.json         # 处理统计信息
└── README.md          # 本文件
```

## 棋盘表示

棋盘状态表示为具有3个通道的张量：
1. 黑棋（黑子位置为1，其他位置为0）
2. 白棋（白子位置为1，其他位置为0）
3. 下一手（黑方行棋时全部为1，白方行棋时全部为0）

## PyTorch使用示例

```python
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader

class GoDataset(Dataset):
    def __init__(self, data_dir):
        self.boards = torch.load(os.path.join(data_dir, "boards.pt"))
        self.moves = torch.load(os.path.join(data_dir, "moves.pt"))
        self.colors = torch.load(os.path.join(data_dir, "colors.pt"))
        
        with open(os.path.join(data_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.moves)
    
    def __getitem__(self, idx):
        return {
            'board': self.boards[idx],
            'move': self.moves[idx],
            'color': self.colors[idx]
        }

# 创建数据集
train_dataset = GoDataset('dataset/train')
val_dataset = GoDataset('dataset/val')
test_dataset = GoDataset('dataset/test')

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)
```

## 模型训练示例

以下是使用该数据集训练简单围棋策略网络的示例代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的围棋策略网络
class SimplePolicyNet(nn.Module):
    def __init__(self, board_size=19):
        super(SimplePolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * board_size * board_size, board_size * board_size)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimplePolicyNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        boards = batch['board'].to(device)
        moves = batch['move'].to(device)
        
        optimizer.zero_grad()
        outputs = model(boards)
        loss = criterion(outputs, moves)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "go_policy_model.pth")
```

## 使用许可

本数据集仅供研究和教育目的使用。

## 创建日期

数据集创建于：2025-03-13
