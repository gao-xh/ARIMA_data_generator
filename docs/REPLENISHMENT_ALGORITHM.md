# 补货策略算法详细文档 (Replenishment Strategy Algorithm)

本文档详细记录了系统中实现的两种独立补货策略：**基准策略 (Baseline Strategy)** 与 **优化策略 (Optimized Strategy)**。这两种策略在平行模拟环境中运行，共享相同的外部环境（需求、天气、流感），以确保对比的公平性。

---

## 1. 符号定义 (Notation)

| 符号 | 含义 | 单位/说明 |
| :--- | :--- | :--- |
| $t$ | 当前时间步 (Current Day) | 天 (Day) |
| $I_t$ | 当前库存水平 (On-Hand Inventory) | 单位 (Units) |
| $IP_t$ | 库存头寸 (Inventory Position) | $IP_t = I_t + \text{InTransit}_t$ |
| $D_t$ | 每日实际需求 (Daily Demand) | 单位 (Units) |
| $\hat{D}_{t+\tau}$ | 预测的未来需求 (Forecasted Demand) | 单位 (Units) |
| $R$ | 补货周期 (Review Period) | 天 (Days) |
| $L$ | 补货提前期 (Lead Time) | 天 (Days, $\in [3, 6]$) |
| $SS$ | 安全库存 (Safety Stock) | 单位 (Units) |
| $Target$ | 目标库存水平 (Target Level / Order-Up-To Level) | 单位 (Units) |
| $\alpha$ | 效期衰减系数 (Validity Decay Coefficient) | 无量纲 ($\in [0, 1]$) |
| $V_{min}$ | 当前库存最小剩余效期 (Min Validity) | 天 (Days) |
| $CV$ | 变异系数 (Coefficient of Variation) | $CV = \sigma / \mu$ |

---

## 2. 基准策略 (Baseline Strategy) - 模拟人工经验

该策略模拟传统药店的人工管理模式，具有**反应滞后**、**依赖经验值**和**恐慌性补货**的特征。

### 2.1 核心逻辑：固定周期检查 (Fixed Periodic Review)

*   **检查频率**: 每 $R=30$ 天检查一次。
*   **预测方法**: 简单移动平均 (SMA)。仅根据过去 30 天的历史销量均值 $\bar{D}_{hist}$ 来估算未来需求。
    *   *注：完全忽略季节性、流感爆发和气温变化。*
*   **目标库存 ($Target_{base}$)**:
    $$ Target_{base} = \bar{D}_{hist} \times (R + L) + SS_{manual} $$
    *   $SS_{manual}$: 人工设定的安全库存，通常为固定值（如 14 天用量）。

### 2.2 补货公式

在检查日 $t$ (当 $t \mod R == 0$):

$$ Order_t = \max(0, Target_{base} - IP_t) $$

### 2.3 紧急补货机制 (Panic Ordering)

为了模拟人工在发现货架空置时的恐慌行为，基准策略包含一个**每日触发**的紧急检查：

1.  **触发条件**: 如果 $IP_t < 3 \times \bar{D}_{hist}$ (库存不足 3 天)。
2.  **检查在途**: 如果加上在途订单 ($IP_t$) 依然 $< 7 \times \bar{D}_{hist}$。
3.  **行动**: 立即发出紧急订单，补足至 14 天用量。
    $$ Order_{emergency} = (14 \times \bar{D}_{hist}) - IP_t $$
    *   *注：这种机制虽然能救急，但往往导致“牛鞭效应”，在需求高峰后造成库存积压。*

---

## 3. 优化策略 (Optimized Strategy) - 改进型 ARIMA

该策略引入了机器学习预测、动态安全库存和效期约束机制。

### 3.1 核心逻辑：动态周期与 AI 预测

*   **检查频率**:
    *   高波动药品 (High CV): $R=15$ 天 (更频繁的检查)。
    *   低/中波动药品: $R=30$ 天。
*   **预测方法**: **Seasonal ARIMAX** (Improved ARIMA)。
    *   引入外部因子 $X$: 气温 ($Temp$), 流感指数 ($Flu$), 节假日 ($Holiday$)。
    *   预测未来 $T = R + L$ 天的每日需求序列 $\hat{D}_{t+1}, \dots, \hat{D}_{t+T}$。

### 3.2 效期衰减机制 (Validity Decay) - "软着陆"优化版

为了防止临期药品积压，系统在计算需求预测时引入衰减系数 $\alpha$。

1.  **计算最小效期** $V_{min}$: 扫描仓库中所有批次，找到最早过期的天数。
2.  **计算基础系数** $\alpha_{base}$ (线性平滑):
    *   若 $V_{min} > 60$: $\alpha_{base} = 1.0$ (不衰减)
    *   若 $V_{min} \le 60$: 线性下降。
    $$ \alpha_{base} = \text{MinFloor} + \frac{(1.0 - \text{MinFloor})}{60} \times V_{min} $$
3.  **波动性保护地板 (Volatility Floor)**:
    *   高波动药品 ($CV > 0.5$): Floor = $0.8$ (允许少量衰减，优先保供)。
    *   低波动药品 ($CV < 0.2$): Floor = $0.4$ (激进衰减，防止报废)。
4.  **最终预测修正**:
    在此基础上，对 ARIMA 的预测值进行加权：
    $$ \hat{D}_{final} = \hat{D}_{raw} \times \alpha_{base} \times (1 + \beta \times CV) $$
    *   *(注: $\beta \approx 0.1$ 为波动性补偿，确保极高波动时不会因为效期而过度砍单)*

### 3.3 补货公式 (The Optimized Formula)

在检查日 $t$:

1.  **周期需求 (Cycle Stock)**:
    $$ CS = \sum_{k=1}^{R+L} \hat{D}_{fresh} $$
    *   $\hat{D}_{fresh}$: 基于最新鲜效期 (Nominal Shelf Life) 的 ARIMA 预测值。即使现有库存快过期，我们也不降低对未来需求的预期，因为补进来的全是新货。这确保了补货目标足以支撑真实市场需求 (保供逻辑)。

2.  **动态安全库存 (Dynamic SS)**:
    不再是固定值，而是基于预测误差的标准差 $\sigma_{err}$ 动态计算：
    $$ SS_{dyn} = Z \times \sigma_{err} \times \sqrt{R + L} $$
    *   $Z$: 服务水平系数 (1.65 for 95%, 2.33 for 99%)。

3.  **有效库存计算 (Effective Inventory)**:
    为了应对临期风险，系统会扫描现有库存批次，并在计算补货量时**扣除**在保护期 $(R+L)$ 内即将过期的库存。
    $$ I_{eff} = (I_{on\_hand} + I_{pipeline}) - \sum Expiring(R+L) $$
    *   **逻辑**: 如果某批次将在下一次补货到货前过期，它实际上无法满足该周期的需求。因此，我们将其视为“无效库存”，通过增加补货量来提前补偿这部分损失，防止未来的断货。

4.  **最终订单 (Final Order)**:
    $$ Order_t = \max(0, (CS + SS_{dyn}) - I_{eff}) $$

### 3.4 紧急补货机制 (Emergency Ordering - Optimized)

为了防止预测误差导致的断货风险，优化策略同样引入了紧急补货机制，但其触发阈值是动态的：

1.  **动态触发阈值**:
    不再是固定的“3天用量”，而是基于当前的**补货提前期** ($L$) 和**安全库存** ($SS$):
    $$ Threshold_{emer} = \hat{D}_{daily} \times (L + 1) + 0.5 \times SS_{dyn} $$
    *   **逻辑**: 只要现有库存不足以支撑到下一批货到货，就立即补货。

2.  **补货量**:
    $$ Order_{emer} = Target_{optimized} - I_{eff} $$
    *   直接补足至目标库存水平，确保恢复服务能力。

---

## 4. 策略对比总结

| 特征 | 基准策略 (Baseline) | 优化策略 (Optimized) |
| :--- | :--- | :--- |
| **驱动数据** | 仅历史均值 (SMA) | 历史趋势 + 气温 + 流感 + 季节性 |
| **补货周期** | 固定 30 天 | 动态 (15/30 天) |
| **安全库存** | 静态经验值 | 动态计算 (基于 $Z$-Score) |
| **效期管理** | 无 (直到过期才报废) | **双重保险**: <br>1. 预测使用 Nominal Shelf Life 防止被旧货误导 (Availability Priority)。<br>2. **扣除即将过期库存**，提前补足缺口。 |
| **紧急应对** | 恐慌性补货 (固定 < 3天触发) | **动态安全网**: 基于提前期与预测偏差触发 |
| **抗波动性** | 差 (滞后) | 强 (提前感知环境变化 & 效期风险) |
