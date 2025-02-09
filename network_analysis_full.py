import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
import os
import numpy as np

class FullTransactionNetwork:
    def __init__(self, csv_file='data/labeled_transactions.csv'):
        """初始化交易网络分析器"""
        self.df = pd.read_csv(csv_file)
        os.makedirs('data/visualizations', exist_ok=True)
        
    def build_full_network(self):
        """构建完整的交易网络"""
        print("正在构建完整交易网络...")
        
        # 构建完整的交易图
        G = nx.DiGraph()
        
        # 计算每个地址的交易统计
        address_stats = defaultdict(lambda: {
            'value': 0.0,
            'frequency': 0,
            'in_value': 0.0,
            'out_value': 0.0
        })
        
        # 添加所有交易
        for _, row in self.df.iterrows():
            # 确保地址是字符串类型
            from_addr = str(row['from_address'])
            to_addr = str(row['to_address'])
            value = float(row['value'])
            
            # 更新地址统计
            address_stats[from_addr]['frequency'] += 1
            address_stats[to_addr]['frequency'] += 1
            address_stats[from_addr]['out_value'] += value
            address_stats[to_addr]['in_value'] += value
            address_stats[from_addr]['value'] += value
            address_stats[to_addr]['value'] += value
            
            # 添加或更新边
            if G.has_edge(from_addr, to_addr):
                G[from_addr][to_addr]['weight'] += value
                G[from_addr][to_addr]['count'] += 1
            else:
                G.add_edge(from_addr, to_addr, weight=value, count=1)
        
        return G, address_stats
    
    def plot_full_network(self):
        """绘制完整的交易网络图"""
        G, address_stats = self.build_full_network()
        
        print(f"总节点数: {G.number_of_nodes()}")
        print(f"总边数: {G.number_of_edges()}")
        
        # 使用 Force Atlas 2 布局
        pos = nx.spring_layout(G, k=2/np.sqrt(G.number_of_nodes()), iterations=50)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_texts = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 节点大小基于交易频率的对数
            size = np.log1p(address_stats[node]['frequency']) * 5
            node_sizes.append(size)
            
            # 节点颜色基于入度/出度比例
            in_value = address_stats[node]['in_value']
            out_value = address_stats[node]['out_value']
            total_value = in_value + out_value
            color = in_value / total_value if total_value > 0 else 0.5
            node_colors.append(color)
            
            # 节点悬停文本
            try:
                addr_str = str(node)
                text = (
                    f"地址: {addr_str[:10]}...\n"
                    f"交易次数: {address_stats[node]['frequency']}\n"
                    f"总交易额: {address_stats[node]['value']/1e18:.2f} ETH\n"
                    f"入账: {in_value/1e18:.2f} ETH\n"
                    f"出账: {out_value/1e18:.2f} ETH"
                )
            except:
                text = f"地址: {node}\n(数据异常)"
            node_texts.append(text)
        
        # 创建节点轨迹
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_texts,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='RdBu',
                colorbar=dict(title='入账比例'),
                line=dict(width=0.5, color='#888'),
                showscale=True
            )
        )
        
        # 准备边数据
        edge_x = []
        edge_y = []
        edge_texts = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # 使用贝塞尔曲线创建弧形边
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # 边的悬停文本
            try:
                from_addr = str(edge[0])
                to_addr = str(edge[1])
                text = (
                    f"从: {from_addr[:10]}...\n"
                    f"到: {to_addr[:10]}...\n"
                    f"交易次数: {edge[2]['count']}\n"
                    f"总金额: {edge[2]['weight']/1e18:.2f} ETH"
                )
            except:
                text = "数据异常"
            edge_texts.extend([text, text, None])
        
        # 为每条边创建单独的轨迹
        edge_traces = []
        max_weight = max(edge[2]['weight'] for edge in G.edges(data=True))

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # 计算这条边的宽度
            width = 0.5 + (edge[2]['weight']/max_weight) * 2
            
            # 创建这条边的轨迹
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(
                    width=width,
                    color='#888'
                ),
                hoverinfo='text',
                text=[f"交易金额: {edge[2]['weight']:.2f} ETH<br>"
                      f"交易次数: {edge[2]['count']}"] * 2 + [None],
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # 创建图形
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='完整以太坊交易网络图',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text=(
                            "节点大小表示交易频率<br>"
                            "节点颜色表示入账比例<br>"
                            "边的颜色表示交易频率<br>"
                            "可缩放和平移查看详情"
                        ),
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0, y=-0.1
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        # 保存为HTML文件
        fig.write_html(
            'data/visualizations/full_network.html',
            include_plotlyjs='cdn',
            full_html=True
        )
        
        print("完整交易网络图已生成到 data/visualizations/full_network.html")
        return fig

def main():
    """主函数"""
    try:
        network = FullTransactionNetwork()
        print("开始生成完整网络分析...")
        network.plot_full_network()
        print("网络分析完成！")
    except Exception as e:
        print(f"生成网络分析时出错: {e}")
        raise

if __name__ == "__main__":
    main() 