# 算法逻辑说明 (Algorithm Logic & Data Translation)

本文档详细记录了如何将《药品基础信息表》(drug_info.csv) 和 《外部影响因子表》(external_factors.csv) 中的业务字段，转化为数学模型中的驱动参数。

## 1. 核心模型架构 (Core Architecture)

基于马尔科夫链蒙特卡洛 (MCMC) 方法，每个【诊所-药品】对(Clinic-Drug Pair)被建模为一个独立的状态机。

### 状态转移方程 (State Transition Equation)
$$ S_{t+1} = S_t + A_t - D'_t - L_t $$

其中:
- $S_t$: 第 $t$ 天结束时的库存状态 (Inventory Level)
- $A_t$: 第 $t$ 天的**实际到货量** (Arrivals)，由 $R_{t-\tau}$ ( $\tau$ 天前的补货决策) 决定。
- $D'_t$: 第 $t$ 天的**实际销量** (Sales)，$D'_t = \min(S_t + A_t, D_t)$
- $D_t$: 第 $t$ 天的**理论需求** (Theoretical Demand)，由环境因子驱动。
- $L_t$: 第 $t$ 天的**损耗** (Loss/Expiry)。

---

## 2. 字段映射逻辑 (Field Mapping Logic)

### 2.1 药品属性映射 (Drug Attributes -> Model Parameters)

我们从 `drug_info.csv` 中提取关键字段，并将其“翻译”为算法的超参数。

| 原始字段 (CSV Original) | 算法参数 (Algorithm Param) | 映射逻辑 (Mapping Logic) | 数学作用 (Mathematical Role) |
| :--- | :--- | :--- | :--- |
| **日均销量** | $BaseDemand$ | 直接读取，且根据诊所规模缩放 ($Scale_{clinic}$) | 需求的基准均值 |
| **效期（月）** | $T_{validity}$ | Value * 30 days | 决定 $L_t$ (损耗) 的触发时间 |
| **波动区间分类** | $\sigma_{mult}$ (Noise Multiplier) | 高波动 $\to 2.0$, 低波动 $\to 0.5$ | 调节 $Noise \sim N(0, \sigma)$ 的方差 |
| **药品品类** | $C_{func}$ (Functional Category) | 关键词匹配 (e.g., "呼吸" $\to$ Respiratory) | 决定 $f(Temp)$ 和 $f(Flu)$ 的敏感度系数 |

#### 波动区间分类详细映射表
- **高波动**: $\sigma \times 2.0$ (需求极不稳定)
- **中高波动**: $\sigma \times 1.5$
- **中波动**: $\sigma \times 1.0$ (基准)
- **低中波动**: $\sigma \times 0.8$
- **低波动**: $\sigma \times 0.5$ (需求非常平稳，如慢病用药)

#### 药品品类详细映射表 (Category Classification)
- **呼吸系统/感冒/解热 (Respiratory)**:
    - 关键词: `呼吸`, `感冒`, `咳`, `肺`, `炎`, `抗生素`
    - 算法行为: 对 **气温下降** 敏感 ($\beta_{temp} > 0$)，对 **流感爆发** 敏感 ($\beta_{flu} > 0$)。
- **慢性病 (Chronic)**:
    - 关键词: `慢病`, `心血管`, `降压`, `降糖`, `降脂`
    - 算法行为: 对外部因子**不敏感** ($\beta \approx 0$)，主要受基准需求驱动。
- **其他 (Other)**:
    - 算法行为: 仅受随机噪声影响。

### 2.2 外部因子映射 (External Factors -> Drivers)

从 `external_factors.csv` 提取环境驱动力。

| 原始字段 | 算法变量 | 作用机制 |
| :--- | :--- | :--- |
| **平均气温** | $Temp_t$ | 当 $Temp_t < Threshold$ 时，呼吸类药物需求非线性上升。 |
| **ILI% (流感百分比)** | $Flu_t$ | 直接乘数效应: $D_t = D_{base} \times (1 + \beta \times Flu_t)$ |
| **季节因子** | $Season_t$ | 调节基准需求 (冬季上浮，夏季下调)。 |

---

## 3. 算法实现伪代码 (Algorithm Pseudocode)

```python
# 初始化
Noise_Level = Config.sigma * Map(波动区间分类)
Category_Type = Map(药品品类)

# 每日循环
For each Day t:
    # 1. 计算环境乘数
    Effect_Temp = f_temp(Temp_t, Category_Type)
    Effect_Flu = f_flu(Flu_t, Category_Type)
    Effect_Season = f_season(Month_t, Category_Type)
    
    # 2. 生成理论需求 (The Generation Step)
    # 核心公式：基准 * 季节 * 气温 * 流感 + 随机噪声
    Demand_Mean = BaseDemand * Effect_Season * Effect_Temp * Effect_Flu
    Real_Demand = Normal(Demand_Mean, Noise_Level)
    
    # 3. 执行状态转移 (The Transition Step)
    Inventory_Next = Inventory_Current + Arrivals - Min(Inventory, Real_Demand) - Loss
```

---

## 4. 结论验证 (Thesis Validation)

通过上述映射，我们确保生成的每一条数据都携带了特定的“基因特征”：
1.  **慢病药 (低波动)**：数据曲线平滑，人工补货容易，极少缺货。
2.  **感冒药 (高波动 + 敏感)**：数据随气温剧烈震荡，且受流感爆发冲击。由于人工补货周期 ($R=14$) 的滞后性，必然会在流感高峰期出现 **缺货 (Stockout)**。

这种差异性正是为了证明您 ARIMA + 外部因子模型在“复杂场景”下的优越性。

---

## 5. 补货策略优化 (Optimization of Replenishment Strategy)

为了更真实地模拟现实世界中的人工补货行为（即“对照组”逻辑），我们在 v2 版本生成器中引入了更复杂的 **混合补货策略 (Hybrid Replenishment Policy)**。该策略旨在复现人工操作的滞后性、恐慌性补货以及对供应链不确定性的反应。

### 5.1 核心逻辑变更 (Core Logic Changes)

原有的简单周期性检查（Period Review, $R, S$）被升级为 **带紧急触发的周期性检查 (Periodic Review with Emergency Trigger)**。

#### A. 紧急补货机制 (Emergency Replenishment)
模拟药店管理者在发现货架空置时的“恐慌性”补货行为。
- **触发条件**: 感知库存 (Physical Inventory) < **紧急阈值 (3天销量)**。
- **行为**:
    1. 检查是否有在途订单 (Incoming Orders)。
    2. 如果 (现有库存 + 在途库存) 仍低于 **7天安全水平**，则立即触发紧急订单。
    3. **补货目标**: 补足至 **14天** 的用量。
    4. **人为延迟**: 设置 **50% 概率** 触发，模拟人工可能因忙碌或疏忽而未及时发现缺货。

#### B. 常规周期补货 (Regular Replenishment)
模拟定期的盘点和订货计划。
- **触发时间**: 每 $R$ 天 (默认为 30 天) 的固定盘点日。
- **计算逻辑**:
    - **预测依据**: 仅依赖 **过去30天的历史平均销量** (Naive Moving Average)。这刻意忽略了季节性趋势和外部因子（气温/流感），从而为 ARIMA 模型留出超越空间。
    - **安全库存**: $AvgSales \times R \times SafetyFactor$。
        - *修正*: 针对高波动药品，人工往往会因为过度担忧缺货而增加安全系数 (Hoarding Behavior)。
    - **补货量**: $Order = Target - (Inventory + Reference\_Pending)$。

### 5.2 供应链不确定性 (Supply Chain Uncertainty)

#### C. 浮动提前期 (Floating Lead Time)
不再使用固定的提前期，而是引入随机波动，模拟物流延误。
- **公式**: $LeadTime = Base(4) + Noise \in \{-1, 0, 1, 2\}$
- **结果**: 订单会在 **3 到 6 天** 后随机到达。这种不确定性会导致即使发出了补货指令，仍可能因为到货晚了一天而发生缺货。

#### D. 信息滞后 (Information Lag)
- 补货决策不是基于实时的“上帝视角”库存，而是基于 $t-Lag$ 天的库存记录 (Lag $\approx$ 2-3 天)。
- 这导致在库存快速下降时，补货指令发出得太晚。

### 5.3 优化目的 (Goal of Optimization)
这套逻辑生成的 **Stockout (缺货)** 和 **Overstock (积压)** 模式更加接近真实的业务痛点：
1.  **突发缺货**: 在流感爆发初期，由于只看历史均值，补货量不足，且带有信息滞后，导致库存断崖式下跌。
2.  **长鞭效应 (Bullwhip Effect)**: 在缺货发生后，管理者恐慌性通过“紧急补货”拉高库存，导致流感结束后库存大量积压，最终过期损耗。

这为论文中 **"利用 ARIMA 预测提前感知趋势，平滑库存波动"** 提供了极佳的对照和改进空间。

