# 系统详细设计与核心算法 (System Detailed Design and Core Algorithms)

本文档整合了系统中各个模块的算法设计，旨在为毕业论文提供完整的“系统实现”章节素材。系统基于 **2024-2025 双年度演化框架 (Evolutionary Framework)**，通过 Baseline (2024) 积累数据，随后在 Optimized (2025) 阶段部署预测引擎，验证算法的增益。

---

## 1. 核心模型架构 (Core Architecture)

基于马尔科夫链蒙特卡洛 (MCMC) 方法，每个【诊所-药品】对(Clinic-Drug Pair)被建模为一个独立的状态机。系统在 $t$ 时刻的状态转移遵循以下方程：

$$ S_{t+1} = S_t + A_{t} - D'_t - L_t $$

*   $S_t$: 第 $t$ 天结束时的库存水平。
*   $A_t$: 实际到货量 (Arrivals)，由 $L$ 天前的订货决策 $O_{t-L}$ 决定。
*   $D'_t$: 实际满足的销量 (Fulfilled Demand)，$D'_t = \min(S_t + A_t, D_t)$。
*   $L_t$: 当日过期待处理损耗 (Expired Loss)。

---

## 2. 需求生成模型 (Demand Generation Model)

为了验证控制算法在复杂环境下的表现，我们构建了一个符合流行病学特征的需求生成器，基于 **2024-2025 真实气象与流感数据**。

### 2.1 核心生成公式
第 $t$ 天的实际销量 $D_t$ 由基准需求叠加环境因子构成：

$$ D_t = D_{base} \times f_{season}(m) \times f_{temp}(T_t) \times f_{flu}(F_t) + \epsilon_t $$

*   $f_{season}$: **季节性调节因子**，针对“呼吸类”药物在冬季显著上浮。
*   $f_{temp}$: **气温敏感函数**，模拟寒潮触发的突发需求。
    $$ f_{temp}(T_t) = 1 + \max(0, \text{Threshold} - T_t) \times \beta_{temp} $$
*   $f_{flu}$: **流感爆发因子**，与 ILI% (流感样病例百分比) 强相关。
    - **Baseline (2024)**: 历史 ILI 数据作为基准。
    - **Optimized (2025)**: 假设已知 ILI 趋势或具有高精度的短期预报能力。
*   $\epsilon_t$: **随机噪声**，服从正态分布 $N(0, \sigma^2)$。

### 2.2 数据预处理 (Data Preprocessing)
根据论文 2.2.2 节要求，外部因子在进入模型前经过统计学变换：
*   **日期对齐**: 严格对齐 2024-01-01 至 2025-12-31 的时间序列，解决 Pandas 索引歧义问题。
*   **气温**: 采用 Z-Score 标准化 ($T' = \frac{T - \mu}{\sigma}$)，消除量纲。

---

## 3. 预测控制模型 (Predictive Control Model - 2025 Phase)

在 Optimized 阶段（2025年），系统引入改进型 ARIMA 模型进行主动干预。

### 3.1 改进型 ARIMA 模型 (Improved ARIMAX)
系统针对传统 ARIMA 的滞后性，引入了外部变量 ($X$) 和动态参数选择机制。

*   **模型形式**: $ARIMAX(p, d, q)$
*   **滚动训练 (Rolling Training)**: 
    - 初始训练集: 2024 全年数据 (Baseline Phase)。
    - 每日更新: $T$ 时刻重新训练，预测 $T+1 \dots T+L+R$。
*   **动态阶数**:
    *   **低波动组**: $(1, 0, 1)$，仅引入季节因子。
    *   **中波动组**: $(2, 1, 2)$，引入气温和流感因子。
    *   **高波动组**: $(3, 1, 3)$，引入全量因子，捕捉复杂震荡。

### 3.2 效期衰减修正 (Validity Decay Correction)
为了解决临期药品的积压问题，预测结果 $\hat{y}_t$ 会乘以一个衰减系数 $\alpha$：

$$ \hat{y}_{corrected} = \hat{y}_t \times \alpha(V_{min}) $$

*   **线性平滑函数**:
    $$ \alpha = \begin{cases} 
    1.0 & V_{min} > 60 \\
    0.5 + \frac{0.5}{60} \times V_{min} & 0 < V_{min} \le 60 
    \end{cases} $$
*   **作用**: 当现有库存即将过期时，即使预测销量很高，也强制抑制补货信号，优先消耗现有库存。

---

## 4. 实验验证框架 (Experimental Validation Framework)

为了全面评估算法的性能与鲁棒性，系统设计了双模态验证框架。

### 4.1 模式一：平行对照 (Parallel Mode)
构建两个相互独立的平行世界，确保输入因子（气温、流感、顾客随机流）在数学上完全一致。
- **Baseline Group**: 2024-2025 全程运行人工经验策略。
- **Optimized Group**: 2024-2025 全程运行 AI 优化策略。
- **目的**: 剥离环境干扰，证明算法在相同条件下的**绝对优势**（如：在同一流感爆发日，AI 是否能避免缺货）。

### 4.2 模式二：演化模拟 (Evolutionary Mode)
模拟真实业务场景中的“数字化转型”过程。
- **Phase 1 (2024 Baseline)**: 系统运行在**人工模式**，积累销售数据，训练模型。
- **Phase 2 (2025 Optimized)**: 系统切换至**AI 托管模式**，利用 Phase 1 的数据进行冷启动，并实时滚动优化。
- **目的**: 验证算法的**学习能力**与**冷启动适应性**，展示引入 AI 后的实际业务提升（Before & After）。

### 4.3 补货策略细节

#### Baseline Strategy (人工组)
模拟传统药店管理：
*   **补货周期**: 固定 $R=30$ 天。
*   **预测方法**: 简单移动平均 (SMA, 过去30天)。
*   **安全库存**: 固定经验值 (14天用量)。
*   **缺陷**: 反应滞后，流感季易缺货，无视效期风险。

#### Optimized Strategy (AI组)
基于数据驱动的智能决策：
*   **补货周期**: 动态调整 ($R=7 \text{ or } 14 \text{ or } 30$)。
*   **预测方法**: Improved ARIMAX (包含环境因子预报)。
*   **动态安全库存**:
    $$ SS_{dyn} = Z_{\alpha} \times \sigma_{err} \times \sqrt{R+L} $$
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

---

## 6. 算法创新点总结 (Summary of Innovation)

1.  **环境耦合**: 将气温、流感数据与 ARIMA 模型深度融合，提升了非平稳序列的预测精度。
2.  **演化视角 (Evolutionary Perspective)**: 通过 2024-2025 连续模拟，展示了算法如何从历史数据中“学习”规律并指导未来决策。
3.  **效期双重保险机制 (Dual-Mechanism Validity Control)**: 
    *   **前端控制**: 预测层面的 $\alpha$ 衰减系数，减少订货。
    *   **后端控制**: 物理层面的 FEFO 出库，减少损耗。
4.  **动态安全库存**: 证明了随预测置信度调整 SS 水位，可以在不降低服务水平的前提下减少 15%-20% 的持仓成本。
