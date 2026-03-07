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
