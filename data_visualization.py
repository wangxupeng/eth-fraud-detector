import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class TransactionVisualizer:
    def __init__(self, csv_file='data/labeled_transactions.csv'):
        # 读取时指定数值类型的列
        self.df = pd.read_csv(csv_file, dtype={
            'block_number': int,
            'value': float,
            'gas': int,
            'gas_price': float,
            'timestamp': int,
            'is_blacklisted': bool
        })
        # 创建图表保存目录
        os.makedirs('data/visualizations', exist_ok=True)
        
    def _convert_timestamp(self):
        """将时间戳转换为可读时间"""
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s')
        
    def _convert_to_eth(self):
        """将 Wei 转换为 ETH"""
        try:
            self.df['value_eth'] = self.df['value'].astype(float) / (10 ** 18)
        except Exception as e:
            print(f"转换错误: {e}")
            print("value 列的前几个值:", self.df['value'].head())
            raise
        
    def plot_blacklist_distribution(self):
        """绘制黑名单交易分布"""
        plt.figure(figsize=(10, 6))
        counts = self.df['is_blacklisted'].value_counts()
        
        # 处理可能不存在的值
        normal_count = counts.get(False, 0)
        blacklist_count = counts.get(True, 0)
        
        plt.bar(['正常地址', '黑名单地址'], [normal_count, blacklist_count])
        plt.title('交易地址黑名单分布')
        plt.ylabel('交易数量')
        
        # 添加具体数值标签
        for i, v in enumerate([normal_count, blacklist_count]):
            plt.text(i, v, str(v), ha='center', va='bottom')
            
        plt.savefig('data/visualizations/blacklist_distribution.png')
        plt.close()
        
    def plot_transaction_values(self):
        """绘制交易金额分布"""
        self._convert_to_eth()
        plt.figure(figsize=(12, 6))
        
        # 使用对数刻度，因为交易金额差异可能很大
        non_zero_values = self.df['value_eth'][self.df['value_eth'] > 0]
        if len(non_zero_values) > 0:
            plt.hist(non_zero_values, bins=50, log=True)
            plt.title('交易金额分布 (ETH)')
            plt.xlabel('交易金额 (ETH)')
            plt.ylabel('交易数量 (对数刻度)')
        else:
            plt.text(0.5, 0.5, '没有非零交易金额', ha='center', va='center')
        
        plt.savefig('data/visualizations/transaction_values.png')
        plt.close()
        
    def plot_time_series(self):
        """绘制交易时间序列"""
        self._convert_timestamp()
        plt.figure(figsize=(15, 6))
        
        # 按时间统计交易数量
        time_series = self.df.groupby('datetime').size()
        plt.plot(time_series.index, time_series.values)
        plt.title('交易时间分布')
        plt.xlabel('时间')
        plt.ylabel('交易数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/visualizations/time_series.png')
        plt.close()
        
    def plot_gas_price_distribution(self):
        """绘制 Gas 价格分布"""
        plt.figure(figsize=(12, 6))
        
        # 将 Wei 转换为 Gwei
        gas_price_gwei = self.df['gas_price'].astype(float) / (10 ** 9)
        plt.hist(gas_price_gwei, bins=50, log=True)
        plt.title('Gas 价格分布')
        plt.xlabel('Gas 价格 (Gwei)')
        plt.ylabel('交易数量 (对数刻度)')
        plt.savefig('data/visualizations/gas_price_distribution.png')
        plt.close()
        
    def generate_statistics(self):
        """生成统计信息"""
        self._convert_to_eth()
        stats = {
            '总交易数': len(self.df),
            '黑名单地址交易数': len(self.df[self.df['is_blacklisted'] == True]),
            '正常地址交易数': len(self.df[self.df['is_blacklisted'] == False]),
            '独立发送地址数': self.df['from_address'].nunique(),
            '独立接收地址数': self.df['to_address'].nunique(),
            '总交易金额(ETH)': self.df['value_eth'].sum(),
            '平均交易金额(ETH)': self.df['value_eth'].mean(),
            '最大交易金额(ETH)': self.df['value_eth'].max(),
            '平均 Gas 限制': self.df['gas'].mean(),
            '平均 Gas 价格(Gwei)': (self.df['gas_price'] / (10 ** 9)).mean()
        }
        
        # 将统计信息保存到文件
        with open('data/visualizations/statistics.txt', 'w', encoding='utf-8') as f:
            for key, value in stats.items():
                f.write(f'{key}: {value:,.2f}\n')
        
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("开始生成可视化图表...")
        try:
            self.plot_blacklist_distribution()
            print("1. 黑名单分布图已生成")
            self.plot_transaction_values()
            print("2. 交易金额分布图已生成")
            self.plot_time_series()
            print("3. 时间序列图已生成")
            self.plot_gas_price_distribution()
            print("4. Gas价格分布图已生成")
            self.generate_statistics()
            print("5. 统计信息已生成")
            print("\n所有可视化图表已生成到 data/visualizations 目录")
        except Exception as e:
            print(f"生成过程中出现错误: {e}")
            raise

if __name__ == "__main__":
    visualizer = TransactionVisualizer()
    visualizer.generate_all_visualizations() 