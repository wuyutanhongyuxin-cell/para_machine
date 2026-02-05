# Paradex Trader - 自学习加密货币交易机器人

> 基于 Thompson Sampling 和在线学习的 Paradex DEX 永续合约交易系统

## 目录

1. [项目概述](#项目概述)
2. [核心原理](#核心原理)
3. [系统架构](#系统架构)
4. [安装配置](#安装配置)
5. [使用指南](#使用指南)
6. [策略详解](#策略详解)
7. [学习系统](#学习系统)
8. [风险管理](#风险管理)
9. [配置参数](#配置参数)
10. [注意事项与风险警告](#注意事项与风险警告)
11. [故障排除](#故障排除)
12. [开发者指南](#开发者指南)

---

## 项目概述

Paradex Trader 是一个专为 [Paradex DEX](https://paradex.trade) 设计的自动化交易系统。它结合了经典的量化交易策略与现代机器学习技术，能够：

- **自动选择最优策略**：使用 Thompson Sampling (汤普森采样) 多臂老虎机算法自动发现并利用表现最好的策略
- **实时学习**：通过 River 库实现在线学习，根据每笔交易的结果持续优化信号过滤
- **多层风险控制**：包含回撤控制、冷却期、Kelly 仓位管理等多重保护机制
- **市场状态感知**：实时检测市场处于趋势、震荡还是突破状态，自动调整策略

### 主要特性

| 特性 | 描述 |
|------|------|
| 交易品种 | BTC-USD-PERP (默认)，支持其他 Paradex 永续合约 |
| 策略数量 | 3 个内置策略：趋势跟踪、均值回归、动量 |
| 学习方式 | Thompson Sampling + Online Learning (River) |
| 风险控制 | 3 级回撤保护 + 冷却期 + Kelly 仓位 |
| 数据存储 | SQLite (WAL模式，支持并发) |
| 手续费 | 使用 Interactive Token 实现零交易费 |

---

## 核心原理

### Thompson Sampling (汤普森采样)

Thompson Sampling 是一种贝叶斯方法解决多臂老虎机问题。在我们的场景中：

```
问题定义：
- 有 N 个策略（臂）
- 每个策略有未知的真实胜率
- 目标：最大化总收益

Thompson Sampling 的做法：
1. 为每个策略维护一个 Beta 分布 Beta(α, β)
   - α = 成功次数 + 1
   - β = 失败次数 + 1
2. 从每个分布中采样一个值
3. 选择采样值最高的策略执行
4. 根据结果更新对应策略的分布
```

**优势**：
- 自动平衡探索与利用
- 不确定性越大，探索越多
- 随着数据积累，自动收敛到最优策略

```python
# 简化示例
class ThompsonSampling:
    def select_strategy(self):
        samples = {}
        for name, arm in self.arms.items():
            # 从 Beta 后验分布采样
            samples[name] = np.random.beta(arm.alpha, arm.beta)
        # 选择采样值最高的策略
        return max(samples, key=samples.get)

    def update(self, strategy, is_win):
        arm = self.arms[strategy]
        if is_win:
            arm.alpha += 1  # 增加成功计数
        else:
            arm.beta += 1   # 增加失败计数
```

### 在线学习 (Online Learning)

使用 River 库实现内存高效的增量学习：

```
传统批量学习：
收集数据 → 训练模型 → 部署预测 → 重复

在线学习：
数据流入 → 预测 → 学习 → 立即更新模型
```

**我们的实现**：
- 使用 `LogisticRegression` 进行二分类（预测交易是否会盈利）
- 使用 `StandardScaler` 在线标准化特征
- 使用 `ADWIN` 检测概念漂移（市场状态变化）

### 市场状态检测

系统识别以下市场状态：

| 状态 | 检测条件 | 适合策略 |
|------|----------|----------|
| TRENDING_UP | 趋势强度 > 0.3，价格上升 | 趋势跟踪 |
| TRENDING_DOWN | 趋势强度 > 0.3，价格下降 | 趋势跟踪 |
| RANGING | 趋势强度 < 0.3，波动适中 | 均值回归 |
| HIGH_VOLATILITY | 波动率百分位 > 75% | 动量 |
| LOW_VOLATILITY | 波动率百分位 < 25% | 均值回归 |
| BREAKOUT_UP/DOWN | 突破历史高/低点 | 动量 |

---

## 系统架构

```
paradex_trader/
├── config/
│   └── settings.py          # 配置管理 (Pydantic)
├── core/
│   ├── client.py            # Paradex API 客户端
│   ├── database.py          # SQLite 数据库
│   ├── models.py            # 数据模型
│   └── exceptions.py        # 异常定义
├── strategies/
│   ├── base.py              # 策略基类
│   ├── trend_follow.py      # 趋势跟踪策略
│   ├── mean_reversion.py    # 均值回归策略
│   └── momentum.py          # 动量策略
├── learning/
│   ├── thompson_sampling.py # Thompson Sampling
│   ├── online_filter.py     # 在线学习过滤器
│   ├── feature_engine.py    # 特征工程
│   └── regime_detector.py   # 市场状态检测
├── risk/
│   ├── position_sizer.py    # Kelly 仓位计算
│   ├── drawdown_control.py  # 回撤控制
│   └── cooldown.py          # 冷却期管理
├── indicators/
│   ├── technical.py         # 技术指标
│   ├── microstructure.py    # 订单簿微观结构
│   └── volatility.py        # 波动率指标
├── utils/
│   ├── logger.py            # 日志系统
│   ├── helpers.py           # 辅助函数
│   └── metrics.py           # 性能指标
├── tests/                   # 测试套件
├── main.py                  # 主入口
└── README.md                # 本文档
```

### 数据流程

```
市场数据 → 指标计算 → 特征提取 → 信号生成
                              ↓
                        Thompson Sampling 选择策略
                              ↓
                        在线学习过滤器判断
                              ↓
                        市场状态兼容性检查
                              ↓
                        风险检查 (回撤/冷却期)
                              ↓
                        Kelly 仓位计算
                              ↓
                        订单执行 → 结果反馈 → 学习更新
```

---

## 安装配置

### 系统要求

- Python 3.9+
- 稳定的网络连接
- Paradex 账户 (Starknet L2)

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository_url>
cd paradex_trader

# 2. 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 复制配置文件
cp .env.example .env

# 5. 编辑配置
nano .env  # 或使用任何编辑器
```

### 配置 .env 文件

```bash
# ======================
# API 配置 (必须)
# ======================
PARADEX_API_KEY=your_api_key_here
PARADEX_API_SECRET=your_api_secret_here
PARADEX_ENVIRONMENT=mainnet  # 生产环境用 mainnet

# ======================
# 交易参数
# ======================
PARADEX_MARKET=BTC-USD-PERP  # 交易品种
MIN_ORDER_SIZE=0.001         # 最小订单大小
MAX_POSITION_SIZE=0.1        # 最大持仓 (BTC)
LOOP_INTERVAL_MS=1000        # 主循环间隔 (毫秒)

# ======================
# 风险参数 (重要!)
# ======================
RISK_PER_TRADE=0.02          # 单笔风险 2%
MAX_POSITION_PCT=0.10        # 最大仓位 10%
KELLY_FRACTION=0.25          # Kelly 系数 (保守)

# 回撤控制
DRAWDOWN_LEVEL1=0.03         # 3% 回撤: 减少仓位
DRAWDOWN_LEVEL2=0.05         # 5% 回撤: 暂停交易
DRAWDOWN_LEVEL3=0.10         # 10% 回撤: 停止交易

# 每日亏损限制
DAILY_LOSS_LIMIT=0.05        # 每日最大亏损 5%

# 冷却期
CONSECUTIVE_LOSS_COOLDOWN=3  # 连续亏损 3 次后冷却
COOLDOWN_DURATION=1800       # 冷却时间 30 分钟

# ======================
# 学习参数
# ======================
MIN_SAMPLES_PER_STRATEGY=20  # 每策略最少探索次数
ONLINE_LEARNING_RATE=0.01    # 在线学习速率
SIGNAL_THRESHOLD=0.45        # 信号过滤阈值

# ======================
# 系统配置
# ======================
DATA_DIR=./data              # 数据存储目录
LOG_LEVEL=INFO               # 日志级别
```

---

## 使用指南

### 基本用法

```bash
# 启动交易 (生产模式)
python -m paradex_trader.main

# 模拟模式 (不执行真实交易)
python -m paradex_trader.main --dry-run

# 调试模式 (详细日志)
python -m paradex_trader.main --debug

# 指定配置文件
python -m paradex_trader.main --config /path/to/.env
```

### 监控运行

启动后，系统会输出：

```
=====================================
Initializing Paradex Trader...
=====================================
Database initialized
Connected to Paradex. Account equity: $10,000.00
Strategies loaded: ['trend_follow', 'mean_reversion', 'momentum']
Thompson Sampling state loaded
Online filter state loaded
All components initialized successfully
=====================================
Starting trading loop...
Market: BTC-USD-PERP
Loop interval: 1000ms
=====================================
```

### 停止交易

- 按 `Ctrl+C` 优雅停止
- 系统会自动保存状态

```
Shutdown requested
Shutting down trading engine...
Thompson Sampling state saved
Online filter state saved
=====================================
SESSION SUMMARY
=====================================
Duration: 2h 35m
Total ticks: 9300
Best strategy: trend_follow
Exploration mode: False
  trend_follow: trials=127, win_rate=53.5%, PnL=$523.45
  mean_reversion: trials=89, win_rate=58.4%, PnL=$312.20
  momentum: trials=42, win_rate=47.6%, PnL=-$45.30
Total PnL: $790.35
Win rate: 54.3%
Sharpe: 1.82
Max drawdown: 2.15%
=====================================
Shutdown complete
```

### 数据查看

交易数据存储在 SQLite 数据库中：

```bash
# 查看交易记录
sqlite3 ./data/paradex_trader.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"

# 查看策略统计
sqlite3 ./data/paradex_trader.db "SELECT * FROM strategy_stats;"
```

---

## 策略详解

### 1. 趋势跟踪策略 (Trend Following)

**原理**：捕捉市场趋势，在突破时入场。

**入场条件**：
```
LONG:
- 价格 > Donchian 通道上轨 (20周期)
- RSI < 75 (未超买)
- 趋势强度 > 0.3

SHORT:
- 价格 < Donchian 通道下轨 (20周期)
- RSI > 25 (未超卖)
- 趋势强度 > 0.3
```

**出场规则**：
- 止损：2 x ATR
- 止盈：3 x ATR 或 Donchian 反向突破

**预期表现**：
- 胜率：45-55%
- 盈亏比：2.0+

```python
# 代码逻辑示意
def generate_signal(self, context):
    if context.mid_price > context.donchian_high:
        if context.rsi < 75 and context.trend_strength > 0.3:
            return Signal(direction="LONG", strength=context.trend_strength)
```

### 2. 均值回归策略 (Mean Reversion)

**原理**：价格偏离均值后会回归。

**入场条件**：
```
LONG (价格过低):
- Z-score < -2.0 (价格低于均值 2 标准差)
- RSI < 30 (超卖)
- 点差 < 0.05%

SHORT (价格过高):
- Z-score > 2.0
- RSI > 70 (超买)
- 点差 < 0.05%
```

**出场规则**：
- 止损：Z-score 继续偏离
- 止盈：Z-score 回归到 ±0.5

**预期表现**：
- 胜率：55-65%
- 盈亏比：0.8-1.0

```python
# Z-score 计算
z_score = (price - bollinger_mid) / (bollinger_upper - bollinger_mid) * 2
```

### 3. 动量策略 (Momentum)

**原理**：捕捉短期价格加速运动。

**入场条件**：
```
LONG:
- 1分钟涨幅 > 0.1%
- 成交量 > 2 倍均量
- 订单簿失衡 > 0.2 (买方强)

SHORT:
- 1分钟跌幅 > 0.1%
- 成交量 > 2 倍均量
- 订单簿失衡 < -0.2 (卖方强)
```

**出场规则**：
- 固定止盈：0.15%
- 固定止损：0.08%
- 时间止损：最长持有 3 分钟

**预期表现**：
- 胜率：50-55%
- 盈亏比：1.8

---

## 学习系统

### Thompson Sampling 详解

#### 初始状态

每个策略从 Beta(1, 1) 开始（均匀分布）：

```
策略 A: Beta(1, 1) → 期望胜率 50%
策略 B: Beta(1, 1) → 期望胜率 50%
策略 C: Beta(1, 1) → 期望胜率 50%
```

#### 学习过程

假设策略 A 交易 10 次，赢 6 次：

```
策略 A: Beta(7, 5) → 期望胜率 58.3%
                   → 95% 置信区间: [33.6%, 80.6%]
```

随着更多数据：

```
100 次交易，55 次盈利:
策略 A: Beta(56, 46) → 期望胜率 54.9%
                     → 95% 置信区间: [45.3%, 64.3%]
```

#### 探索与利用

```python
# 探索阶段：确保每个策略至少试用 20 次
if any(arm.trials < 20 for arm in arms):
    return least_tried_strategy()

# 利用阶段：Thompson Sampling
samples = {name: np.random.beta(arm.alpha, arm.beta) for name, arm in arms.items()}
return max(samples, key=samples.get)
```

#### 非平稳适应

使用衰减因子处理市场变化：

```python
# 衰减因子 0.995 意味着：
# - 100 次交易前的数据权重 ≈ 60%
# - 200 次交易前的数据权重 ≈ 36%
arm.alpha = prior + (arm.alpha - prior) * decay_factor
arm.beta = prior + (arm.beta - prior) * decay_factor
```

### 在线学习过滤器

#### 特征输入

```python
features = {
    "price_change_1m": 0.05,      # 1分钟价格变化
    "momentum_1m": 0.3,           # 动量指标
    "rsi_14": 55.0,               # RSI
    "volatility_1m": 0.02,        # 波动率
    "spread_pct": 0.015,          # 点差百分比
    "imbalance": 0.2,             # 订单簿失衡
    "trend_strength": 0.4,        # 趋势强度
    ...
}
```

#### 预测与学习

```python
# 预测
prediction = online_filter.predict(features)
if prediction.probability > 0.45 and prediction.model_ready:
    execute_trade()

# 交易完成后学习
is_profitable = pnl > 0
online_filter.learn(entry_features, is_profitable)
```

#### 概念漂移检测

使用 ADWIN 算法检测市场状态变化：

```python
# 当检测到漂移时
if drift_detector.drift_detected:
    logger.warning("Market conditions changed!")
    # 模型会自动加快适应新环境
```

### 特征重要性

系统跟踪每个特征与交易结果的相关性：

```python
importance = online_filter.get_feature_importance()
# {'trend_strength': 0.85, 'imbalance': 0.72, 'rsi_14': 0.65, ...}
```

---

## 风险管理

### 仓位计算 (Kelly Criterion)

Kelly 公式计算最优仓位：

```
f* = (p * b - q) / b

其中:
- f* = 最优仓位比例
- p = 胜率
- q = 1 - p = 败率
- b = 盈亏比

实际应用：
Kelly 仓位 = f* × Kelly 系数 (0.25) × 信号强度 × 波动率调整
```

**示例**：
```
胜率 = 55%
盈亏比 = 1.5
Kelly 系数 = 0.25

f* = (0.55 × 1.5 - 0.45) / 1.5 = 0.25

最大理论仓位 = 25%
实际仓位 = 25% × 0.25 = 6.25%
```

### 回撤控制

三级保护机制：

```
Level 1: 回撤 3%
└── 动作: 仓位减半
└── 目的: 降低风险暴露

Level 2: 回撤 5%
└── 动作: 暂停新开仓
└── 目的: 防止继续亏损

Level 3: 回撤 10%
└── 动作: 完全停止交易
└── 目的: 资金保护
```

### 冷却期机制

触发条件与响应：

| 触发条件 | 冷却时间 | 原因 |
|----------|----------|------|
| 连续亏损 3 次 | 30 分钟 | 防止连续错误 |
| 单笔亏损 > 3% | 60 分钟 | 市场可能异常 |
| 高波动 (百分位 > 95%) | 15 分钟 | 等待市场稳定 |

### 每日亏损限制

```
每日最大亏损 = 账户权益 × 5%

示例:
账户 $10,000
每日限额 = $500

达到限额后，当日停止交易
```

---

## 配置参数

### 完整参数列表

#### API 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PARADEX_API_KEY` | 必须设置 | API 密钥 |
| `PARADEX_API_SECRET` | 必须设置 | API 密钥 |
| `PARADEX_ENVIRONMENT` | testnet | mainnet/testnet |
| `API_TIMEOUT` | 30 | API 超时 (秒) |
| `MAX_RETRIES` | 3 | 最大重试次数 |

#### 交易参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PARADEX_MARKET` | BTC-USD-PERP | 交易品种 |
| `MIN_ORDER_SIZE` | 0.001 | 最小订单大小 |
| `MAX_POSITION_SIZE` | 0.1 | 最大持仓 |
| `LOOP_INTERVAL_MS` | 1000 | 循环间隔 (ms) |
| `MAX_SPREAD_PCT` | 0.05 | 最大允许点差 |

#### 风险参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RISK_PER_TRADE` | 0.02 | 单笔风险比例 |
| `MAX_POSITION_PCT` | 0.10 | 最大仓位比例 |
| `KELLY_FRACTION` | 0.25 | Kelly 系数 |
| `DRAWDOWN_LEVEL1` | 0.03 | 一级回撤阈值 |
| `DRAWDOWN_LEVEL2` | 0.05 | 二级回撤阈值 |
| `DRAWDOWN_LEVEL3` | 0.10 | 三级回撤阈值 |
| `DAILY_LOSS_LIMIT` | 0.05 | 每日亏损限制 |

#### 学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MIN_SAMPLES_PER_STRATEGY` | 20 | 最少探索次数 |
| `THOMPSON_DECAY` | 0.995 | 衰减因子 |
| `ONLINE_LEARNING_RATE` | 0.01 | 学习速率 |
| `ONLINE_MIN_SAMPLES` | 50 | 模型启用阈值 |
| `SIGNAL_THRESHOLD` | 0.45 | 过滤阈值 |

#### 策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DONCHIAN_PERIOD` | 20 | Donchian 周期 |
| `RSI_PERIOD` | 14 | RSI 周期 |
| `ATR_PERIOD` | 14 | ATR 周期 |
| `BB_PERIOD` | 20 | 布林带周期 |
| `BB_STD` | 2.0 | 布林带标准差 |

---

## 注意事项与风险警告

### 重要风险警告

```
⚠️  警告：加密货币交易具有高风险！

1. 永续合约带有杠杆，可能导致超过本金的损失
2. 过去表现不代表未来结果
3. 自动化交易系统可能出现意外行为
4. 请只使用可以承受损失的资金
5. 在 testnet 充分测试后再使用 mainnet
```

### 使用前检查清单

- [ ] 在 testnet 运行至少 1 周
- [ ] 理解所有风险参数的含义
- [ ] 设置合理的风险限制
- [ ] 确认 API 权限配置正确
- [ ] 备份 .env 文件
- [ ] 设置监控告警

### 最佳实践

#### 1. 从小资金开始

```bash
# 建议初始配置
MAX_POSITION_PCT=0.05    # 仓位不超过 5%
RISK_PER_TRADE=0.01      # 单笔风险 1%
```

#### 2. 保守的回撤设置

```bash
# 保守配置
DRAWDOWN_LEVEL1=0.02     # 2% 减仓
DRAWDOWN_LEVEL2=0.03     # 3% 暂停
DRAWDOWN_LEVEL3=0.05     # 5% 停止
```

#### 3. 充分的探索期

```bash
# 让每个策略都有足够样本
MIN_SAMPLES_PER_STRATEGY=30
```

#### 4. 定期检查

- 每天查看交易日志
- 每周分析策略表现
- 每月评估参数设置

### 常见陷阱

| 陷阱 | 说明 | 避免方法 |
|------|------|----------|
| 过度交易 | 循环间隔过短导致频繁交易 | 设置 `LOOP_INTERVAL_MS >= 1000` |
| 仓位过大 | Kelly 全仓导致大幅波动 | 使用 `KELLY_FRACTION = 0.25` |
| 忽略点差 | 高点差时仍然交易 | 设置 `MAX_SPREAD_PCT < 0.05` |
| 追涨杀跌 | 在极端行情中持续加仓 | 启用回撤控制和冷却期 |

### API 限制

Paradex API 有速率限制：

- 每秒最多 10 个请求
- 建议 `LOOP_INTERVAL_MS >= 500`

---

## 故障排除

### 常见错误

#### 1. API 连接失败

```
错误: APIError: Connection refused
```

**解决方法**：
1. 检查网络连接
2. 验证 API 密钥正确
3. 确认环境设置 (mainnet/testnet)

#### 2. 余额不足

```
错误: InsufficientBalanceError
```

**解决方法**：
1. 检查账户余额
2. 减少 `MAX_POSITION_PCT`
3. 确保有足够保证金

#### 3. 订单被拒绝

```
错误: OrderRejectedError: Size too small
```

**解决方法**：
1. 增加 `MIN_ORDER_SIZE`
2. 增加账户余额

#### 4. 状态文件损坏

```
错误: Failed to load Thompson state
```

**解决方法**：
```bash
# 删除状态文件重新学习
rm ./data/thompson_state.json
rm ./data/online_filter.pkl
```

### 日志分析

```bash
# 查看最近的错误
grep ERROR ./data/paradex_trader.log | tail -20

# 查看交易执行
grep "ENTRY\|EXIT" ./data/paradex_trader.log

# 查看策略选择
grep "Thompson selected" ./data/paradex_trader.log
```

### 性能监控

```python
# 在代码中添加性能监控
import cProfile
cProfile.run('engine.run()', 'output.prof')
```

---

## 开发者指南

### 添加新策略

1. 创建策略文件：

```python
# strategies/my_strategy.py
from strategies.base import BaseStrategy, Signal, TradeContext

class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def generate_signal(self, context: TradeContext) -> Signal:
        # 实现你的策略逻辑
        if your_condition:
            return Signal(
                direction="LONG",
                strategy=self.name,
                strength=0.8,
                reason="My signal reason",
            )
        return Signal(direction="HOLD", strategy=self.name, strength=0)

    def get_exit_levels(self, entry_price, direction, context):
        # 定义出场规则
        return ExitLevels(
            stop_loss=entry_price * 0.98,
            take_profit=entry_price * 1.03,
        )
```

2. 注册策略：

```python
# strategies/__init__.py
from .my_strategy import MyStrategy

def create_all_strategies(settings):
    return {
        "trend_follow": TrendFollowStrategy(settings),
        "mean_reversion": MeanReversionStrategy(settings),
        "momentum": MomentumStrategy(settings),
        "my_strategy": MyStrategy(settings),  # 添加这行
    }
```

### 添加新指标

```python
# indicators/my_indicator.py
class MyIndicator:
    def __init__(self, period: int = 14):
        self.period = period
        self.values = deque(maxlen=period)

    def update(self, price: float) -> Optional[float]:
        self.values.append(price)
        if len(self.values) < self.period:
            return None
        return self._calculate()

    def _calculate(self) -> float:
        # 实现计算逻辑
        pass
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_integration.py -v

# 生成覆盖率报告
pytest tests/ --cov=paradex_trader --cov-report=html
```

### 代码风格

项目使用：
- Black 格式化
- isort 导入排序
- mypy 类型检查

```bash
# 格式化代码
black paradex_trader/
isort paradex_trader/

# 类型检查
mypy paradex_trader/
```

---

## 支持与反馈

- 问题报告：请在 GitHub Issues 中提交
- 功能建议：欢迎提交 Pull Request

---

## 许可证

MIT License

---

## 免责声明

本软件按"原样"提供，不提供任何明示或暗示的保证。使用本软件进行的任何交易风险由用户自行承担。开发者不对任何直接或间接损失负责。

在使用本软件进行真实交易之前，请确保您完全理解加密货币交易的风险，并在测试网络上进行充分测试。
