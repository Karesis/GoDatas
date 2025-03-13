# Go棋谱数据集构建项目全面文档

## 项目背景

在人工智能和机器学习快速发展的今天，高质量的训练数据集对于深度学习模型的性能至关重要。围棋作为一种极其复杂的策略性博弈游戏，其数据集的构建尤其具有挑战性。本项目旨在从海量而杂乱不堪的的SGF（Smart Game Format）文件中，精心构建一个高质量、结构化的围棋对局数据集，为深度学习研究提供坚实的基础。

## 数据源

基础数据来源：https://homepages.cwi.nl/~aeb/go/games
这个原始数据集是一个全面的围棋对局集合，需要复杂的处理技术将原始SGF文件转换为结构化的、适合机器学习的格式。

## 项目架构

### 总体设计理念
项目采用模块化设计，分为两个核心模块：数据整理（`gogames_organizer.py`）和数据集构建（`go_training_dataset.py`）。这种架构确保了数据处理的灵活性和可扩展性。

### 关键组件详解

#### 1. 数据整理模块 `gogames_organizer.py`

##### 主要功能
- 递归扫描并处理SGF文件
- 元数据自动修复和标准化
- 对局质量评估
- 文件去重和组织
- 生成详细处理报告

##### 核心特性
1. **元数据增强**
   - 内置专业棋手数据库
   - 自动识别和标准化棋手名称
   - 赛事信息规范化
   - 对局日期处理

2. **质量评估机制**
   - 多维度评分系统
   - 考虑因素：
     * 棋手段位
     * 赛事级别
     * 时间跨度
     * 对局注释
     * 棋手知名度

3. **文件组织策略**
   - 支持多种组织模式：
     * 按质量分类
     * 按棋手分类
     * 按日期分类
     * 按赛事分类

#### 2. 数据集构建模块 `go_training_dataset.py`

##### 数据表示创新
- 19路棋盘的标准化张量表示
- 三通道特征编码：
  1. 黑子位置
  2. 白子位置
  3. 轮次信息

##### 数据采样策略
- 跳过开局和尾盘
- 随机抽取关键对局位置
- 保证位置分布的多样性

##### 数据集构建特点
- 生成训练、验证和测试数据集
- 支持多进程并行处理
- 灵活的质量阈值控制

## 技术细节

### 元数据处理技术

#### 棋手名称标准化
- 支持多语言名称映射
- 处理不同语言文字和拼音
- 自动识别知名棋手

#### 日期格式规范化
- 支持多种日期输入格式
- 自动转换为标准 YYYY-MM-DD 格式
- 处理不完整日期信息

#### 赛事信息映射
- 内置权威赛事数据库
- 自动识别和扩展赛事全称
- 处理赛事名称的变体

### 质量评估算法

质量评分基于多维度指标：

1. **棋手信息权重** (30%)
   - 职业段位
   - 国际排名
   - 知名度

2. **赛事级别权重** (25%)
   - 国际大赛
   - 国家级锦标赛
   - 知名杯赛

3. **对局特征权重** (20%)
   - 对局深度
   - 变化图复杂度
   - 注释详细程度

4. **时间相关性** (15%)
   - 对局年代
   - 近期比赛权重提升

5. **文件元数据完整性** (10%)
   - 元数据字段完整程度
   - 关键信息的准确性

### 数据集特征工程

#### 位置采样策略
- 排除开局30手和结束前30手
- 随机抽取代表性位置
- 确保采样位置的战略价值

#### 张量表示优化
- 3通道特征编码
- 浮点数归一化
- 支持PyTorch原生加载

## 使用指南

### 环境准备
- Python 3.7+
- 推荐配置：
  * NumPy
  * PyTorch 1.7+
  * Tqdm
  * Papaparse

### 数据处理流程
1. 数据整理
```bash
python gogames_organizer.py \
    /path/to/raw/sgf/files \
    /path/to/organized/output \
    --quality 80 \
    --organize quality \
    --workers 8
```

2. 数据集构建
```bash
python go_training_dataset.py \
    /path/to/organized/sgf/files \
    /path/to/dataset/output \
    --samples 10000 \
    --val-split 0.1 \
    --test-split 0.1
```

### 数据加载示例
```python
import torch
from torch.utils.data import DataLoader

class GoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.boards = torch.load(f"{data_dir}/boards.pt")
        self.moves = torch.load(f"{data_dir}/moves.pt")
        self.colors = torch.load(f"{data_dir}/colors.pt")
    
    def __len__(self):
        return len(self.moves)
    
    def __getitem__(self, idx):
        return {
            'board': self.boards[idx],
            'move': self.moves[idx],
            'color': self.colors[idx]
        }

# 加载数据集
train_dataset = GoDataset('output/train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## 研究价值

本数据集可应用于：
- 围棋走子预测模型
- 对局策略分析
- 人工智能训练
- 棋局模式识别研究

## 未来展望

- 支持更多棋盘尺寸
- 兼容更多深度学习框架
- 扩展元数据分析维度
- 持续更新棋手和赛事数据库
- (也许)

## 许可

Apache 2.0

## 贡献与反馈

- 欢迎提交Issues
- 鼓励通过Pull Request改进项目
- 期待更多开发者参与

## 版本历史

- v1.0：初始版本
- v1.1：性能优化与bug修复

## 致谢

感谢围棋社区和开源贡献者的支持。
