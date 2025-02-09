from blockchain_data import BlockchainDataFetcher
from blacklist_data import BlacklistFetcher
from data_processor import TransactionProcessor
from gnn_model import TransactionGNN, TransactionGraphDataset, train_model, evaluate_model, predict_suspicious_addresses, setup_logger
from network_analysis_full import FullTransactionNetwork
import torch
from torch_geometric.transforms import RandomNodeSplit

def main():
    """
    # 数据已下载，data下
    try:
        # 初始化区块链数据获取器
        blockchain_fetcher = BlockchainDataFetcher()
        
        # 获取最近10个区块的交易
        latest_block = blockchain_fetcher.get_latest_block_number()
        start_block = latest_block - 10
        df_transactions = blockchain_fetcher.get_block_transactions(start_block, latest_block)
        print(f"Fetched {len(df_transactions)} transactions")
        
        # 获取黑名单地址
        blacklist_addresses = BlacklistFetcher.get_blacklist_addresses()
        
        # 处理数据
        processor = TransactionProcessor()
        df_labeled = processor.label_transactions(df_transactions, blacklist_addresses)
        
        # 保存数据
        processor.save_to_csv(df_labeled)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    """
    pass

def run_gnn_analysis():
    # 设置日志
    logger = setup_logger()
    logger.info("开始GNN分析...")
    
    # 1. 加载交易网络
    network = FullTransactionNetwork()
    G, _ = network.build_full_network()
    
    # 2. 选择更多的黑名单地址
    import random
    num_nodes = len(G.nodes())
    num_blacklist = int(num_nodes * 0.1)  # 选择10%的节点作为黑名单地址
    blacklist_addresses = random.sample(list(G.nodes()), num_blacklist)
    logger.info(f"选择了 {len(blacklist_addresses)} 个黑名单地址用于训练")
    
    try:
        # 3. 准备数据
        logger.info("准备数据...")
        dataset = TransactionGraphDataset(G, blacklist_addresses)
        data = dataset.prepare_data()
        
        # 4. 添加训练/验证/测试掩码
        transform = RandomNodeSplit(
            num_val=0.1,
            num_test=0.1
        )
        data = transform(data)
        
        # 5. 创建模型
        logger.info("创建模型...")
        model = TransactionGNN(num_features=data.x.size(1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 6. 训练模型
        logger.info("开始训练...")
        train_model(model, data, optimizer, logger=logger)
        
        # 7. 评估模型
        logger.info("评估模型...")
        accuracy = evaluate_model(model, data)
        logger.info(f"测试集准确率: {accuracy:.4f}")
        
        # 8. 预测可疑地址
        logger.info("预测可疑地址...")
        suspicious_addresses = predict_suspicious_addresses(model, data, dataset, logger=logger)
            
    except Exception as e:
        logger.error(f"运行GNN分析时出错: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
    run_gnn_analysis() 