# 以太坊交易数据说明文档

本文档对 `labeled_transactions.csv` 文件中的数据字段进行说明。该数据集包含了以太坊区块链上的交易信息，以及相关的黑名单标记。

## 数据字段说明

1. **block_number** (区块号)
   - 交易所在的区块编号
   - 表示该交易被打包进的区块序号
   - 示例：21807305

2. **transaction_hash** (交易哈希)
   - 交易的唯一标识符
   - 由交易数据计算得出的哈希值
   - 示例：0x379854e906d6dca577ef177b09814e616a1c90ece1cf4cc901574bc6f13bf3dc

3. **from_address** (发送方地址)
   - 发起交易的钱包地址
   - 20字节的以太坊地址，以0x开头
   - 示例：0xe2cA3167B89b8Cf680D63B06E8AeEfc5E4EBe907

4. **to_address** (接收方地址)
   - 接收交易的钱包地址
   - 20字节的以太坊地址，以0x开头
   - 示例：0xE8c060F8052E07423f71D445277c61AC5138A2e5

5. **value** (交易金额)
   - 转账的以太币数量，以 Wei 为单位
   - 1 ETH = 10^18 Wei
   - 示例：11267879211040000 (约等于 0.011 ETH)

6. **gas** (燃料限制)
   - 交易执行的最大燃料限制
   - 表示愿意为这笔交易支付的最大计算资源
   - 示例：21000 (标准转账交易的燃料消耗)

7. **gas_price** (燃料价格)
   - 每单位燃料的价格，以 Wei 为单位
   - 决定交易的优先级，价格越高越优先
   - 示例：2811969867 (约 2.8 Gwei)

8. **timestamp** (时间戳)
   - 交易被打包进区块的 Unix 时间戳
   - 表示交易确认的时间
   - 示例：1739084171

9. **is_blacklisted** (黑名单标记)
   - 布尔值，表示交易地址是否在黑名单中
   - True: 地址在黑名单中
   - False: 地址不在黑名单中

## 数据用途

该数据集可用于：
- 分析以太坊网络上的交易模式
- 监控可疑地址的交易行为
- 研究区块链网络的经济活动
- 识别潜在的风险交易

## 注意事项

1. 金额单位换算：
   - Wei 转换为 ETH：将 value 除以 10^18
   - Gwei 转换为 ETH：将 gas_price 除以 10^9

2. 时间戳解读：
   - 可使用标准时间库将 timestamp 转换为可读时间格式

3. 地址格式：
   - 所有地址都是以太坊格式，以 0x 开头
   - 地址长度固定为 42 个字符（包含 0x 前缀）

## Infura API 限制说明

使用 Infura 免费计划获取数据时，需要注意以下限制：

1. **请求限制**
   - 每天最多 100,000 个请求
   - 每秒最多 10 个请求
   - Archive 数据访问受限（只能访问最近 128 个区块的完整数据）

2. **WebSocket 连接**
   - 最多支持 2 个并发 WebSocket 连接
   - 连接时长限制为 24 小时

3. **数据访问范围**
   - 实时区块数据
   - 交易信息
   - 智能合约状态
   - Gas 价格
   - 账户余额
   - 区块信息

4. **支持的网络**
   - 以太坊主网（Mainnet）
   - 测试网络（Sepolia, Goerli）
   - 第二层网络（Optimism, Arbitrum）
   - 其他 EVM 兼容链

5. **数据延迟**
   - 实时数据延迟通常在 1-2 个区块以内
   - 历史数据查询可能有更长延迟

## 建议
1. 对于大量数据查询，建议：
   - 实现请求限速
   - 使用本地缓存
   - 分批获取数据
   - 错误重试机制

2. 对于实时监控：
   - 使用 WebSocket 而不是 HTTP 轮询
   - 实现断线重连
   - 监控连接状态 