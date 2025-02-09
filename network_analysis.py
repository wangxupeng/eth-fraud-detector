import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import os
import matplotlib
import numpy as np  # 添加 numpy 导入
from matplotlib.font_manager import FontProperties
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

# 尝试多个中文字体，按优先级排序
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti', 'Source Han Sans CN']

def get_available_font():
    """获取系统中可用的中文字体"""
    for font in chinese_fonts:
        try:
            FontProperties(font)
            return font
        except:
            continue
    return None

# 设置中文字体
font = get_available_font()
if font:
    matplotlib.rcParams['font.sans-serif'] = [font]
else:
    print("警告：未找到合适的中文字体，文字可能无法正常显示")
    
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')  # 改用默认样式

# 设置绘图样式
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'grid.alpha': 0.2,
    'font.size': 12,  # 增大字体大小
    'axes.titlesize': 14,  # 增大标题字体大小
})

class TransactionNetwork:
    def __init__(self, csv_file='data/labeled_transactions.csv'):
        """初始化交易网络分析器"""
        self.df = pd.read_csv(csv_file)
        os.makedirs('data/visualizations', exist_ok=True)
        
    def build_network(self, max_nodes=100):  # 增加到100个节点
        """构建交易网络"""
        print("正在构建交易网络...")
        
        # 计算地址的各项指标
        address_stats = defaultdict(lambda: {
            'value': 0.0,          # 交易总额
            'frequency': 0,        # 交易频率
            'unique_peers': set()  # 唯一交易对手
        })
        
        # 统计每个地址的交易情况
        for _, row in self.df.iterrows():
            from_addr = row['from_address']
            to_addr = row['to_address']
            value = float(row['value'])
            
            # 更新发送方统计
            address_stats[from_addr]['value'] += value
            address_stats[from_addr]['frequency'] += 1
            address_stats[from_addr]['unique_peers'].add(to_addr)
            
            # 更新接收方统计
            address_stats[to_addr]['value'] += value
            address_stats[to_addr]['frequency'] += 1
            address_stats[to_addr]['unique_peers'].add(from_addr)
        
        # 计算重要性分数
        importance_score = {}
        for addr, stats in address_stats.items():
            # 综合考虑交易金额、频率和交易对手数量
            importance_score[addr] = (
                stats['value'] * 
                stats['frequency'] * 
                len(stats['unique_peers'])
            )
        
        # 选择最重要的地址
        top_addresses = sorted(
            importance_score.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_nodes]
        top_addresses = {addr[0] for addr in top_addresses}
        
        # 构建交易图
        G = nx.DiGraph()
        edge_weights = defaultdict(float)
        
        # 添加边和权重
        for _, row in self.df.iterrows():
            from_addr = row['from_address']
            to_addr = row['to_address']
            
            # 只添加重要地址之间的交易
            if from_addr in top_addresses and to_addr in top_addresses:
                edge_key = (from_addr, to_addr)
                edge_weights[edge_key] += float(row['value'])
        
        # 添加边到图中
        for (source, target), weight in edge_weights.items():
            G.add_edge(source, target, weight=weight)
            
        return G, address_stats
    
    def plot_network(self, max_nodes=100):  # 增加到100个节点
        """绘制交易网络图"""
        G, address_stats = self.build_network(max_nodes)
        
        if not G.nodes():
            print("警告：网络中没有节点")
            return G
            
        plt.figure(figsize=(20, 20))  # 增大图形尺寸
        
        # 计算节点大小和颜色
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            # 使用交易频率和唯一对手数量确定节点大小
            size = (
                address_stats[node]['frequency'] * 
                len(address_stats[node]['unique_peers']) * 
                20  # 调整基础大小
            )
            size = min(size, 3000)  # 限制最大尺寸
            node_sizes.append(size)
            
            # 使用入度出度比例确定颜色
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            ratio = in_degree / (in_degree + out_degree) if (in_degree + out_degree) > 0 else 0.5
            node_colors.append(ratio)
        
        # 使用 spring_layout 布局，增加节点间距和迭代次数
        pos = nx.spring_layout(G, k=3, iterations=200)
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(G, pos,
                                     node_size=node_sizes,
                                     node_color=node_colors,
                                     cmap='coolwarm',
                                     alpha=0.7)
        
        # 绘制边
        if G.edges():
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(edge_weights)
            edge_widths = [max(0.5, (w / max_weight) * 2) for w in edge_weights]
            
            nx.draw_networkx_edges(G, pos,
                                 width=edge_widths,
                                 edge_color='lightgray',
                                 alpha=0.6,
                                 arrows=True,
                                 arrowsize=15)
        
        # 使用 FontProperties 设置标题
        title_font = FontProperties(font) if font else None
        plt.title('以太坊交易网络图\n节点大小表示交易频率，颜色表示入度比例，边的粗细表示交易量',
                 fontproperties=title_font,
                 pad=20,  # 增加标题和图表的间距
                 fontsize=14)
        
        # 为颜色条添加中文标签
        plt.colorbar(nodes, label='入度比例').set_label('入度比例', 
                                                   fontproperties=title_font)
        
        # 为所有节点添加标签
        labels = {node: node[:6] + '...' for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.axis('off')
        
        # 保存高质量图片
        plt.savefig('data/visualizations/transaction_network.png',
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.3)  # 增加边距
        plt.close()
        
        return G
    
    def generate_network_stats(self, G):
        """生成网络统计信息"""
        if not G.nodes():
            print("警告：网络中没有节点")
            return
            
        stats = {
            '节点数量': G.number_of_nodes(),
            '边数量': G.number_of_edges(),
            '平均度': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.nodes() else 0,
            '最大入度': max(dict(G.in_degree()).values()) if G.nodes() else 0,
            '最大出度': max(dict(G.out_degree()).values()) if G.nodes() else 0,
            '连通分量数': nx.number_strongly_connected_components(G),
            '平均聚类系数': nx.average_clustering(G.to_undirected()) if G.nodes() else 0
        }
        
        # 将网络统计信息保存到文件
        with open('data/visualizations/network_stats.txt', 'w', encoding='utf-8') as f:
            f.write("交易网络统计信息:\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:,.2f}\n")
            
            # 添加一些重要节点信息
            f.write("\n重要节点分析:\n")
            
            # 度中心性最高的节点
            degree_centrality = nx.degree_centrality(G)
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("\n度中心性最高的5个节点:\n")
            for addr, cent in top_degree:
                f.write(f"{addr}: {cent:.4f}\n")
            
            # 介数中心性最高的节点
            betweenness_centrality = nx.betweenness_centrality(G)
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("\n介数中心性最高的5个节点:\n")
            for addr, cent in top_betweenness:
                f.write(f"{addr}: {cent:.4f}\n")
        
        print("网络统计信息已生成")
        return stats

    def plot_draggable_network(self, max_nodes=50):  # 减少节点数以提高性能
        """生成可拖动节点的交互式网络图"""
        G, address_stats = self.build_network(max_nodes)
        
        if not G.nodes():
            print("警告：网络中没有节点")
            return G
            
        # 初始化节点位置（随机分布在圆上）
        angles = np.linspace(0, 2*np.pi, len(G.nodes()), endpoint=False)
        radius = 1
        pos = {
            node: (radius * np.cos(angle), radius * np.sin(angle))
            for node, angle in zip(G.nodes(), angles)
        }
        
        # 准备节点数据
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            text=[f"{node[:8]}..." for node in G.nodes()],  # 显示简短地址
            textposition="top center",
            hovertext=[
                f"地址: {node}\n"
                f"交易频率: {address_stats[node]['frequency']}\n"
                f"交易对手数: {len(address_stats[node]['unique_peers'])}\n"
                f"交易总额: {address_stats[node]['value']/1e18:.2f} ETH"
                for node in G.nodes()
            ],
            marker=dict(
                size=[
                    min(address_stats[node]['frequency'] * 2, 30)
                    for node in G.nodes()
                ],
                color=[
                    len(address_stats[node]['unique_peers'])
                    for node in G.nodes()
                ],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='交易对手数量'),
                line=dict(width=1, color='white')
            ),
            customdata=list(G.nodes()),  # 存储完整地址
        )
        
        # 准备边数据
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(
                    width=np.log1p(weight/1e18),  # 使用对数缩放
                    color='rgba(150,150,150,0.5)'
                ),
                hoverinfo='text',
                text=f"交易额: {weight/1e18:.2f} ETH",
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # 创建图形
        fig = go.FigureWidget(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=dict(
                    text='以太坊交易网络图（可拖动节点）',
                    x=0.5,
                    y=0.95
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text="拖动节点可调整位置<br>双击重置视图<br>滚轮缩放",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.01, y=0.01
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                dragmode='pan'  # 允许拖动整个视图
            )
        )
        
        # 添加节点拖动功能
        def update_point(trace, points, selector):
            if points.point_inds:
                ind = points.point_inds[0]
                node = trace.customdata[ind]
                
                # 更新节点位置
                x_new = points.xs[0]
                y_new = points.ys[0]
                pos[node] = (x_new, y_new)
                
                # 更新相关的边
                with fig.batch_update():
                    # 更新节点位置
                    trace.x = list(x for x, y in [pos[n] for n in G.nodes()])
                    trace.y = list(y for x, y in [pos[n] for n in G.nodes()])
                    
                    # 更新边的位置
                    edge_ind = 0
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        fig.data[edge_ind].x = [x0, x1, None]
                        fig.data[edge_ind].y = [y0, y1, None]
                        edge_ind += 1
        
        # 绑定拖动事件
        node_trace.on_click(update_point)
        
        # 保存为独立的HTML文件
        fig.write_html(
            'data/visualizations/draggable_network.html',
            include_plotlyjs='cdn',
            full_html=True
        )
        
        print("可拖动的交互式网络图已生成到 data/visualizations/draggable_network.html")
        return fig

def main():
    """主函数"""
    try:
        network = TransactionNetwork()
        print("开始生成网络分析...")
        
        # 生成静态图
        G = network.plot_network(max_nodes=100)
        
        # 生成可拖动的交互式图
        network.plot_draggable_network(max_nodes=50)
        
        if G and G.nodes():
            network.generate_network_stats(G)
            print("网络分析完成！")
        else:
            print("警告：未能生成有效的网络图")
    except Exception as e:
        print(f"生成网络分析时出错: {e}")
        raise

if __name__ == "__main__":
    main() 