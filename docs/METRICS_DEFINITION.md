# 演化模拟统计指标说明 (Evolution Metrics Guide)

本文档详细定义了 **Evolution Simulation (二阶段演化模拟)** 报告中的关键统计指标及其计算方法。这些指标用于量化对比 **Baseline (2024 Q4)** 与 **Optimized (2025 Q4)** 两个阶段的运营绩效。

---

## 📅 统计区间 (Time Periods)
为了消除季节性偏差，对比仅针对每年相同的 **Q4 (9月-12月)** 高波动时段进行：
*   **Baseline (人工对照组)**: `2024-09-01` 至 `2024-12-31`
*   **Optimized (AI优化组)**: `2025-09-01` 至 `2025-12-31`

---

## 📊 核心指标定义 (Key Performance Indicators)

### 1. Loss Rate (损耗率)
衡量因药品过期而被迫废弃的库存比例。
*   **定义**: 区间内产生的报损数量占（销售数量+报损数量）的百分比。
*   **公式**:
    $$ \text{Loss Rate} = \frac{\sum \text{Loss}_{qty}}{\sum \text{Sales}_{qty} + \sum \text{Loss}_{qty}} \times 100\% $$
*   **解读**: 越低越好。AI 通过效期衰减 (Validity Decay) 机制，在临期前抑制补货，从而降低此项。

### 2. Stockout Rate (缺货率)
衡量因库存不足导致无法满足患者需求的频率。
*   **定义**: 区间内发生缺货的天数占总天数的百分比。
*   **公式**:
    $$ \text{Stockout Rate} = \frac{\text{Count}(\text{Days where Stock} < \text{Demand})}{\text{Total Days}} \times 100\% $$
*   **解读**: 越低越好。AI 通过提前感知外部因子（如流感爆发、降温）来提前备货，降低此项。

### 3. Turnover Days (周转天数)
衡量库存转化为销售的速度，反映资金效率。
*   **定义**: 平均库存能维持多少天的平均销售。
*   **公式**:
    $$ \text{Turnover} = \frac{\text{Avg Daily Inventory}}{\text{Avg Daily Sales}} $$
*   **解读**: 越低代表效率越高，资金占用越少。但过低可能导致缺货风险，需维持在合理区间 (40-50天)。

### 4. Backlog / Avg Inv (平均库存积压)
衡量仓库中平均积压的药品数量。
*   **定义**: 区间内每日库存水平的算术平均值。
*   **公式**:
    $$ \text{Avg Inv} = \frac{\sum_{t=1}^{N} \text{Inventory}_t}{N} $$
*   **解读**: 越低越好（在保证不缺货的前提下）。反映了去库存的效果。

### 5. Funds Occupied (资金占用)
衡量压在库存上的资金成本。
*   **定义**: 平均库存数量乘以单价。
*   **公式**:
    $$ \text{Funds} = \text{Avg Inv} \times \text{Unit Price} $$
*   **解读**: 直接的财务指标，越低越好。

### 6. Model MAPE (预测误差率)
衡量模型对未来需求预测的偏差程度。
*   **全称**: Mean Absolute Percentage Error (平均绝对百分比误差)。
*   **定义**: 预测值与真实销量之间偏差的平均百分比。
*   **公式**:
    $$ \text{MAPE} = \frac{1}{n} \sum \left| \frac{\text{Actual} - \text{Forecast}}{\text{Actual}} \right| \times 100\% $$
    *(注: 实际计算时会排除销量为0的样本以防除零错误)*
*   **解读**: **越低越好**。Baseline 通常基于简单平均，误差较大；Optimized 基于 ARIMA，误差应显著降低。

### 7. Forecast Metric (预测准确度)
描述预测值是如何生成的，以及它如何影响库存决策。
*   **Baseline (人工经验)**:
    *   **公式**: $F_t = \text{SMA}_{30} (D_{t-30:t})$
    *   简单移动平均 (Simple Moving Average, SMA)，仅看过去30天均值，对突发流感反应迟钝。
*   **Optimized (AI预测)**:
    *   **公式**: $F_t = \text{ARIMAX}(p,d,q) + \text{Climate} + \text{Flu} + \text{ValidityDecay}$
    *   综合考虑历史趋势、季节性、天气预报及效期衰减因子 $\alpha$，能精准捕捉未来波动。

---

## 📉 对比逻辑 (Comparison Logic)
UI 表格中的 **"Change"** 列逻辑如下：

*   **Positive Metric (越高越好)**: `Change = Opt - Base`
    *   *暂无此类指标*
*   **Inverse Metric (越低越好)**: `Change = Opt - Base`
    *   若 Change < 0 (数值下降): 显示为 **绿色 (Good)** ✅
    *   若 Change > 0 (数值上升): 显示为 **红色 (Bad)** ❌

例如：如果 `Stockout Rate` 从 `5.0%` 变为 `2.0%`，Change 为 `-3.0%`，标记为 **Good**。

**注意**: 平行世界的所有外部因子（气温、流感、随机种子）在两个年份是受控且相似的，因此该对比具备统计学意义上的公平性。