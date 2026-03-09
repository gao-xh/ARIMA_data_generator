# 需求生成算法 (Demand Generation Logic)

本文档详细描述了系统用于模拟真实世界药品销量的生成模型。该模型通过结合基础需求、季节性趋势、环境影响因子（气温、流感）以及随机波动，为仿真实验提供符合流行病学特征的合成数据。

---

## 1. 核心生成公式 (Core Equation)

第 $t$ 天的实际销量 $D_t$ 由以下分量合成：

$$ D_t = D_{base} \times f_{season}(m) \times f_{temp}(T_t) \times f_{flu}(F_t) + \epsilon_t $$

其中：
*   $D_{base}$: 药品的**基准日均销量**。
*   $f_{season}(m)$: **季节性调节因子**（月度指数）。
*   $f_{temp}(T_t)$: **气温敏感因子**。
*   $f_{flu}(F_t)$: **流感爆发因子**。
*   $\epsilon_t$: **随机噪声项** (Random Noise)。

---

## 2. 因子详细定义 (Factor Definitions)

### 2.1 基础需求 ($D_{base}$)
从 `drug_info.csv` 直接读取。
*   **规模缩放**: 为模拟不同规模的诊所，系统应用缩放系数 $Scale$。
    $$ D_{base}^{adj} = D_{base}^{raw} \times Scale_{clinic} $$

### 2.2 季节性因子 ($f_{season}$)
根据该药品所属的治疗领域（Category），赋予不同的月度曲线。
*   **呼吸类 (Respiratory)**: 冬季 (12-2月) 系数 $> 1.2$，夏季 (6-8月) 系数 $< 0.8$。
*   **慢病类 (Chronic)**: 全年系数波动极小 ($\approx 1.0$)。

### 2.3 气温敏感函数 ($f_{temp}$)
模拟寒潮对发病率的影响。
$$ f_{temp}(T_t) = 1 + \max(0, (Threshold - T_t)) \times \beta_{temp} $$
*   $T_t$: 当日平均气温。
*   $Threshold$: 触发阈值（例如 10°C）。
*   $\beta_{temp}$: 药品的**气温敏感度**。
    *   感冒药 $\beta \approx 0.05$ (气温每降1度，需求增5%)。
    *   降压药 $\beta \approx 0$。

### 2.4 流感爆发函数 ($f_{flu}$)
模拟流感季的冲击。
$$ f_{flu}(F_t) = 1 + F_t \times \beta_{flu} $$
*   $F_t$: 当日流感样病例百分比 (ILI%)。
*   $\beta_{flu}$: 药品的**流感敏感度**。
    *   抗病毒药、解热药 $\beta > 2.0$。

### 2.5 随机噪声 ($\epsilon_t$)
模拟日常的不可预测波动。
$$ \epsilon_t \sim \mathcal{N}(0, \sigma^2) $$
$$ \sigma = D_{base} \times CV \times \text{NoiseLevel} $$
*   **$CV$ (变异系数)**: 决定药品的波动类别。
    *   High: $CV > 0.5$
    *   Medium: $0.2 < CV < 0.5$
    *   Low: $CV < 0.2$

---

## 3. 药品分类特征映射 (Category Mapping)

系统根据药品名称关键词，自动映射参数特征：

| 药品类别 | 关键词示例 | 气温敏感度 ($\beta_{temp}$) | 流感敏感度 ($\beta_{flu}$) | 随机波动 ($CV$) |
| :--- | :--- | :--- | :--- | :--- |
| **高度敏感 (Respiratory)** | 感冒, 咳, 肺, 炎, 病毒 | **高** (0.05~0.1) | **极高** (2.0+) | **高** (High) |
| **中度敏感 (General)** | 头孢, 阿莫西林, 消炎 | **中** (0.02) | **中** (1.0) | **中** (Medium) |
| **不敏感 (Chronic)** | 血压, 糖, 心脑, 慢病 | **无** (0.0) | **无** (0.0) | **低** (Low) |

---

## 4. 算法流程 (Algorithm Flow)

```python
For each day t:
    1. 获取外部数据: Temp_t, ILI_t
    2. 确定药品参数: Beta_temp, Beta_flu, BaseDemand
    3. 计算乘数效应:
       M_temp = 1 + max(0, 15 - Temp_t) * Beta_temp
       M_flu = 1 + ILI_t * Beta_flu
    4. 合成理论均值:
       Mu_t = BaseDemand * M_temp * M_flu
    5. 加入随机波动:
       Demand_t = Max(0, Normal(Mu_t, Sigma))
    6. 取整输出:
       Final_Demand = Round(Demand_t)
```

这种生成机制确保了数据既有**统计学规律**（适合 ARIMA 学习），又有**随机不可预测性**（挑战库存策略），完美服务于论文的对比实验。
