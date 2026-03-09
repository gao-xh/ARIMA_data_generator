# 算法逻辑说明 (Algorithm Logic & Data Translation)

本文档详细记录了系统如何利用 《药品基础信息表》(drug_info.csv) 和 《外部影响因子表》(external_factors.csv) 构建 **2024-2025 双年度演化模拟**。

## 1. 核心模型架构 (Core Architecture)

基于马尔科夫链蒙特卡洛 (MCMC) 方法，系统构建了 **双模态验证框架 (Dual-Mode Validation Framework)**，以全面评估算法性能。

### 1.1 实验模式定义 (Mode Definitions)

为了满足不同维度的验证需求，系统实现了两种独立的模拟模式：

#### A. 平行对照模式 (Parallel Benchmark Mode)
- **目的**: 验证算法在相同环境下的理论优越性。
- **逻辑**: 构建两个平行的“平行世界”：
  - **World A (Baseline)**: 全程 (2024-2025) 运行 **人工经验策略**。
  - **World B (AI-Optimized)**: 全程 (2024-2025) 运行 **改进型 ARIMA 策略**。
- **控制变量**: 两个世界共享完全相同的**随机种子 (Random Seed)**，确保每日的气温、流感爆发、顾客流量完全一致。
- **应用场景**: 对应系统界面的 **Tab 1: Strategy Benchmark**。

#### B. 演化模拟模式 (Evolutionary Simulation Mode)
- **目的**: 模拟真实世界部署效果（Before/After 对比）。
- **逻辑**: 在单一时间轴上进行策略切换：
  - **Phase 1 (2024, Baseline)**: 系统运行在**人工模式**，积累销售数据，训练模型。
  - **Phase 2 (2025, Optimized)**: 系统切换至**AI 托管模式**，利用 Phase 1 的数据进行冷启动，并实时滚动优化。
- **应用场景**: 对应系统界面的 **Tab 2: Evolution Simulation**。

### 1.2 状态转移方程
$$ S_{t+1} = S_t + A_t - D'_t - L_t $$
- $S_t$: 库存状态 (Inventory Level)
- $A_t$: 实际到货量 (Arrivals)，受 $L$ (Lead Time, 默认4-7天) 延迟影响。
- $D'_t$: 实际满足销量，$\min(S_t + A_t, D_t)$。
- $L_t$: 过期损耗 (Expired Loss)。

---

## 2. 字段映射逻辑 (Field Mapping Logic)

### 2.1 外部因子映射 (External Factors)
从 `external_factors.csv` (2024-2025) 提取环境驱动力。

| 原始字段 | 数据变换 (Transformation) | 算法变量 | 作用机制 |
| :--- | :--- | :--- | :--- |
| **日期** | **索引对齐**: 确保 2024-01-01 至 2025-12-31 连续 | $t$ | 时间序列基准 |
| **平均气温** | **Z-Score 标准化**: $T' = \frac{T - \mu}{\sigma}$ | $Temp\_Std$ | 影响呼吸类药物需求 |
| **平均降水量** | **对数变换**: $R' = \ln(Rain + 1)$ | $Rain\_Log$ | 修正右偏分布，作为 ARIMA 外生变量 |
| **ILI%** | 直接引用 (0-100) | $Flu_t$ | 直接乘数效应: $D_t \propto (1 + \beta \cdot Flu_t)$ |

### 2.2 药品属性映射 (Drug Attributes)
| 原始字段 | 算法参数 | 映射逻辑 |
| :--- | :--- | :--- |
| **日均销量** | $BaseDemand$ | 需求的基准均值 $\mu$ |
| **波动区间分类** | $\sigma_{mult}$ | 高波动($2.0$), 中($1.0$), 低($0.5$) |
| **药品品类** | $Category$ | 呼吸/感冒 $\to$ 气温敏感; 慢病 $\to$ 稳定 |

---

## 3. 需求生成模型 (Demand Generation)

$$ D_t = \text{Base} \times f_{season}(t) \times f_{temp}(T_t) \times f_{flu}(F_t) + \mathcal{N}(0, \sigma^2) $$

- **2024年 (Baseline)**: 生成原始需求，不仅用于库存模拟，还作为**训练集 (Ground Truth)**。
- ***Noise Injection***: 所有因子均引入随机扰动，模拟现实世界的不可预测性。
- **2025年 (Optimized)**: 同样的需求生成逻辑继续运行，但**库存控制器**开始使用预测值 $\hat{D}_t$ 进行干预。

---

## 4. 补货策略对比 (Replenishment Strategies)

### 4.1 Baseline 策略 (2024)
模拟传统人工管理：
- **固定周期**: 每 $R=30$ 天检查一次。
- **简单预测**: 仅使用过去 30 天的历史平均销量 (SMA)。
- **被动响应**: 仅在物理库存低于 3 天用量时触发“紧急补货”，但往往为时已晚（受限于提前期 $L$）。

### 4.2 Optimized 策略 (2025)
基于 ARIMA 的主动管理：
- **动态周期**: 高波动药品缩短为 $R=7$ 或 $R=14$ 天。
- **ARIMA 预测**: 
    - 输入: 历史销量 + 未来气温/流感预报。
    - 输出: 未来 $R+L$ 天的总需求预测 $\sum \hat{D}$。
- **效期修正**: 若批次平均效期 $<60$ 天，强制降低订货量，优先去库存。
- **主动响应**: 预测到未来流感爆发时，提前 $L$ 天（提前期）发出订单，避免缺货。

---

## 5. 关键算法实现细节

### 5.1 数据对齐与清洗
为防止 ARIMA 模型报错 (e.g., "Ambiguous Index"), 系统在训练前执行严格的预处理：
```python
# 修复索引歧义 (Fix Ambiguous Date Index)
exog_hist = exog_hist.reset_index(drop=True)
future_exog = future_exog.reset_index(drop=True)

# 确保列名一致 (CSV -> Constants)
col_map = {
    '平均气温2m(℃)': C.EXT_TEMP,
    'ILI%': C.EXT_FLU,
    ...
}
```

### 5.2 滚动预测 (Rolling Forecast)
在 2025 年模拟中，系统执行 **Walk-Forward Validation**：
1.  **T时刻**: 收集 2024-01-01 至 $T$ 的所有数据。
2.  **Retrain**: 重新训练/更新 ARIMA 模型参数。
3.  **Predict**: 预测 $T+1$ 至 $T+L+R$ 的需求。
4.  **Order**: 生成最优订货量 $Q^* = \hat{D}_{total} + SS_{dyn} - Inventory$。

---

## 6. 验证指标 (Validation Metrics)

系统自动计算并对比 2024 与 2025 的关键指标：
1.  **缺货率 (Stockout Rate)**: 缺货天数 / 总天数。
2.  **周转天数 (Turnover Days)**: 平均库存 / 日均销量。
3.  **服务水平 (Service Level)**: 满足需求量 / 总需求量。
4.  **GMROI**: 毛利回报率（与库存成本挂钩）。

通过 2025 年显著优于 2024 年的指标，证明算法有效性。

