import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.loader import DataLoader
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
import os

class TransactionGNN(torch.nn.Module):
    """以太坊交易网络的图神经网络模型"""
    
    def __init__(self, num_features, hidden_channels=64, num_classes=2):
        """
        初始化GNN模型
        
        参数:
            num_features (int): 输入特征维度
            hidden_channels (int): 隐藏层维度
            num_classes (int): 输出类别数(2表示二分类)
        """
        super(TransactionGNN, self).__init__()
        
        # 第一层图卷积
        self.conv1 = GCNConv(num_features, hidden_channels)
        # 第二层图卷积
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        # 第三层图卷积
        self.conv3 = GCNConv(hidden_channels//2, num_classes)
        
        # Dropout层防止过拟合
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵
            edge_index: 边的连接关系
        """
        # 第一层卷积+激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层卷积+激活
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第三层卷积
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class TransactionGraphDataset:
    """处理交易网络数据并转换为图数据格式"""
    
    def __init__(self, network_graph, blacklist_addresses):
        """
        初始化数据集
        
        参数:
            network_graph: NetworkX图对象
            blacklist_addresses: 黑名单地址列表
        """
        self.G = network_graph
        self.blacklist_addresses = set(blacklist_addresses)
        
        # 创建地址到索引的映射
        self.addr_to_idx = {addr: idx for idx, addr in enumerate(self.G.nodes())}
        
    def extract_node_features(self):
        """提取节点特征"""
        features = []
        all_values = defaultdict(list)
        
        for node in self.G.nodes():
            # 基础特征
            in_degree = self.G.in_degree(node)
            out_degree = self.G.out_degree(node)
            in_value = sum(d['weight'] for _, _, d in self.G.in_edges(node, data=True))
            out_value = sum(d['weight'] for _, _, d in self.G.out_edges(node, data=True))
            in_freq = len(list(self.G.in_edges(node)))
            out_freq = len(list(self.G.out_edges(node)))
            
            # 收集所有值用于标准化
            all_values['in_degree'].append(in_degree)
            all_values['out_degree'].append(out_degree)
            all_values['in_value'].append(in_value)
            all_values['out_value'].append(out_value)
            all_values['in_freq'].append(in_freq)
            all_values['out_freq'].append(out_freq)
        
        # 计算每个特征的均值和标准差
        stats = {}
        for key, values in all_values.items():
            values = torch.tensor(values, dtype=torch.float)
            stats[key] = {
                'mean': values.mean(),
                'std': values.std() if values.std() > 0 else 1.0
            }
        
        # 标准化特征
        for node in self.G.nodes():
            in_degree = (self.G.in_degree(node) - stats['in_degree']['mean']) / stats['in_degree']['std']
            out_degree = (self.G.out_degree(node) - stats['out_degree']['mean']) / stats['out_degree']['std']
            in_value = (sum(d['weight'] for _, _, d in self.G.in_edges(node, data=True)) - stats['in_value']['mean']) / stats['in_value']['std']
            out_value = (sum(d['weight'] for _, _, d in self.G.out_edges(node, data=True)) - stats['out_value']['mean']) / stats['out_value']['std']
            in_freq = (len(list(self.G.in_edges(node))) - stats['in_freq']['mean']) / stats['in_freq']['std']
            out_freq = (len(list(self.G.out_edges(node))) - stats['out_freq']['mean']) / stats['out_freq']['std']
            
            node_features = [
                float(in_degree),
                float(out_degree),
                float(in_value),
                float(out_value),
                float(in_freq),
                float(out_freq),
            ]
            features.append(node_features)
            
        return torch.tensor(features, dtype=torch.float)
    
    def prepare_data(self):
        """准备PyTorch Geometric数据对象"""
        # 提取特征
        x = self.extract_node_features()
        
        # 构建边索引（将地址转换为索引）
        edge_list = []
        for src, dst in self.G.edges():
            edge_list.append([self.addr_to_idx[src], self.addr_to_idx[dst]])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # 构建标签
        y = torch.tensor([
            1 if node in self.blacklist_addresses else 0 
            for node in self.G.nodes()
        ])
        
        # 创建数据对象
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
        
        return data

    def get_address_from_idx(self, idx):
        """根据索引获取地址"""
        idx_to_addr = {v: k for k, v in self.addr_to_idx.items()}
        return idx_to_addr[idx]

def setup_logger():
    """设置日志"""
    # 创建logs目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 生成日志文件名（使用时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/gnn_analysis_{timestamp}.log'
    
    # 删除旧的日志文件
    for old_log in os.listdir('logs'):
        if old_log.startswith('gnn_analysis_'):
            os.remove(os.path.join('logs', old_log))
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger()

def train_model(model, data, optimizer, epochs=200, patience=20, logger=None):
    """添加早停机制和日志"""
    if logger is None:
        logger = logging.getLogger()
        
    best_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"开始训练模型，总轮数: {epochs}, 早停耐心值: {patience}")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # 早停检查
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}, best loss: {best_loss:.4f}')
            break
            
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')

def evaluate_model(model, data):
    """
    评估模型
    
    参数:
        model: GNN模型
        data: 图数据对象
    """
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # 计算测试集准确率
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
        
        return acc

def predict_suspicious_addresses(model, data, dataset, logger=None):
    """预测可疑地址并记录日志"""
    if logger is None:
        logger = logging.getLogger()
        
    model.eval()
    logger.info("开始预测可疑地址...")
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probabilities = torch.exp(out)[:, 1]  # 获取正类的概率
        
        # 计算概率分布
        probs = probabilities.numpy()
        threshold = np.percentile(probs, 90)  # 使用前10%的概率作为阈值
        logger.info(f"使用阈值: {threshold:.4f}")
        
        # 获取高风险地址
        suspicious_scores = {
            dataset.get_address_from_idx(idx): float(prob)
            for idx, prob in enumerate(probabilities)
            if float(prob) > threshold  # 使用动态阈值
        }
        
        # 按可疑度排序并添加更多信息
        sorted_addresses = sorted(suspicious_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\n发现 {len(sorted_addresses)} 个可疑地址:")
        logger.info("\n可疑地址详细信息:")
        logger.info("-" * 80)
        
        for addr, score in sorted_addresses:
            node_idx = dataset.addr_to_idx[addr]
            in_degree = dataset.G.in_degree(addr)
            out_degree = dataset.G.out_degree(addr)
            in_value = sum(d['weight'] for _, _, d in dataset.G.in_edges(addr, data=True))
            out_value = sum(d['weight'] for _, _, d in dataset.G.out_edges(addr, data=True))
            
            logger.info(f"地址: {addr[:10]}...")
            logger.info(f"可疑度: {score:.4f}")
            logger.info(f"入度: {in_degree}, 出度: {out_degree}")
            logger.info(f"入账总额: {in_value/1e18:.2f} ETH")
            logger.info(f"出账总额: {out_value/1e18:.2f} ETH")
            logger.info("-" * 80)
        
        return suspicious_scores 