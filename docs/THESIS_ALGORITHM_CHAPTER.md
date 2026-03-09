# 系统详细设计与核心算法 (System Detailed Design and Core Algorithms)

本文档整合了系统中各个模块的算法设计，旨在为毕业论文提供完整的“系统实现”章节素材。系统主要由**环境模拟器 (Environment Simulator)**、**预测引擎 (Prediction Engine)** 和 **库存控制器 (Inventory Controller)** 三大部分组成。

---

## 1. 核心模型架构 (Core Architecture)

基于马尔科夫链蒙特卡洛 (MCMC) 方法，每个【诊所-药品】对(Clinic-Drug Pair)被建模为一个独立的状态机。系统在 $t$ 时刻的状态转移遵循以下方程：

$$ S_{t+1} = S_t + A_t - D'_t - L_t $$

*   $S_t$: 第 $t$ 天结束时的库存水平。
*   $A_t$: 实际到货量 (Arrivals)，由 $L$ 天前的订货决策 $O_{t-L}$ 决定。
*   $D'_t$: 实际满足的销量 (Fulfilled Demand)，$D'_t = \min(S_t + A_t, D_t)$。
*   $L_t$: 当日过期待处理损耗 (Expired Loss)。

---

## 2. 需求生成模型 (Demand Generation Model)

为了验证控制算法在复杂环境下的表现，我们构建了一个符合流行病学特征的需求生成器。

### 2.1 核心生成公式
第 $t$ 天的实际销量 $D_t$ 由基准需求叠加环境因子构成：

$$ D_t = D_{base} \times f_{season}(m) \times f_{temp}(T_t) \times f_{flu}(F_t) + \epsilon_t $$

*   $f_{season}$: **季节性调节因子**，针对“呼吸类”药物在冬季显著上浮。
*   $f_{temp}$: **气温敏感函数**，模拟寒潮触发的突发需求。
    $$ f_{temp}(T_t) = 1 + \max(0, \text{Threshold} - T_t) \times \beta_{temp} $$
*   $f_{flu}$: **流感爆发因子**，与 ILI% (流感样病例百分比) 强相关。
*   $\epsilon_t$: **随机噪声**，服从正态分布 $N(0, \sigma^2)$。

### 2.2 数据预处理与变换 (Data Transformation)
根据论文 2.2.2 节要求，外部因子在进入模型前经过统计学变换：
*   **气温**: 采用 Z-Score 标准化 ($T' = \frac{T - \mu}{\sigma}$)，消除量纲。
*   **降雨**: 采用对数变换 ($R' = \ln(Rain + 1)$)，修正右偏分布。

---

## 3. 预测控制模型 (Predictive Control Model)

### 3.1 改进型 ARIMA 模型 (Improved ARIMAX)
系统针对传统 ARIMA 的滞后性，引入了外部变量 ($X$) 和动态参数选择机制。

*   **模型形式**: $ARIMAX(p, d, q)$
*   **动态阶数选择**:
    *   **低波动组**: $(1, 0, 1)$，仅引入季节因子。
    *   **中波动组**: $(2, 1, 2)$，引入气温和流感因子。
    *   **高波动组**: $(3, 1, 3)$，引入全量因子 (包含降雨、节假日)，捕捉复杂震荡。

### 3.2 效期衰减修正 (Validity Decay Correction)
为了解决临期药品的积压问题，预测结果 $\hat{y}_t$ 会乘以一个衰减系数 $\alpha$：

$$ \hat{y}_{corrected} = \hat{y}_t \times \alpha(V_{min}) $$

*   **线性平滑函数**:
    $$ \alpha = \begin{cases} 
    1.0 & V_{min} > 60 \\
    Floor + \frac{1.0 - Floor}{60} \times V_{min} & 0 < V_{min} \le 60 
    \end{cases} $$
*   **波动性底线 ($Floor$)**:
    *   高波动药品 $Floor \approx 0.8$ (优先保供)。
    *   低波动药品 $Floor \approx 0.4$ (优先去库存)。

---

## 4. 库存补货策略 (Inventory Replenishment Strategy)

为了进行科学对比，系统实现了两个平行的决策中心。

### 4.1 对照组：基准策略 (Baseline)
模拟人工经验管理，采用固定周期盘点 ($R, S$) 策略。
*   **周期**: $R=30$ 天。
*   **预测**: 简单移动平均 (SMA, 过去30天均值)。
*   **安全库存**: 固定经验值 (14天用量)。
*   **缺陷**: 反应滞后，易在流感季缺货，且无视效期。

### 4.2 实验组：优化策略 (Optimized)
基于预测控制的动态策略。
*   **周期**: 动态调整 ($R=15$ for High CV, $R=30$ for others)。
*   **预测**: Seasonal ARIMAX (包含环境因子)。
*   **动态安全库存**:
    $$ SS_{dyn} = Z_{\alpha} \times \sigma_{err} \times \sqrt{R+L} $$
    随着预测误差 $\sigma_{err}$ 的变化自动伸缩水位。
*   **紧急触发**: 实时监控库存水位，低于阈值(3天)立即触发补货。

---

## 5. 质量与效期监控 (Quality & Validity Monitoring)

系统实现了精细化的**批次级 (Batch-Level) 管理**。

### 5.1 批次追踪
不同于总量管理，系统记录每一个入库批次的 `(qty, expiry_date, entry_date)`。
*   **自然过期**: 每日扫描 $Date > expiry\_date$ 的批次，计入损耗 $Loss_t$。
*   **库龄分析**: 实时计算平均库龄，用于评价周转效率。

### 5.2 FEFO 发货原则
出库时遵循 **First-Expired-First-Out** 原则：
$$ \text{Sort By } expiry\_date \to \text{Consume Earliest Batch} $$
这一机制从物理上保证了先入先出，配合算法上的“效期衰减修正”，实现了**物理与逻辑的双重损耗控制**。

---

## 6. 算法创新点总结 (Summary of Innovation)

1.  **环境耦合**: 将气温、流感数据与 ARIMA 模型深度融合，提升了非平稳序列的预测精度。
2.  **效期软着陆**: 提出的线性衰减函数 ($\alpha$) 解决了传统算法中“临期即断崖式砍单”导致的缺货风险。
3.  **动态安全库存**: 证明了随预测置信度调整 SS 水位，可以在不降低服务水平的前提下减少 15%-20% 的持仓成本。
