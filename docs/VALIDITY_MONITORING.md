# 全生命周期效期监控与管理算法 (Lifecycle Validity Monitoring Algorithm)

本文档详细记录了系统中用于模拟药品从入库到出库（或过期）全过程的监控逻辑。该机制是 **Optimized 策略** 能够降低损耗率的核心基础。

---

## 1. 核心数据结构：批次追踪 (Batch-Level Tracking)

不同于简单的总量库存管理 ($Total = \sum q_i$)，本系统采用了 **批次列表 (List of Batches)** 的方式存储库存数据。

### 数据定义
每个库存记录是一个包含以下字段的独立对象：

```python
batch = {
    'qty': float,        # 剩余数量 (Units)
    'expiry_day': int,   # 绝对失效日期 (Day Index, e.g., Day 730)
    'entry_day': int     # 入库日期 (用于计算库龄)
}
```
系统维护一个全局列表 `self.inventory_batches` 存储所有当前在库的批次。

---

## 2. 每日状态流转 (Daily State Transition)

在模拟的一天 ($t$) 开始时，系统首先执行状态检查，处理自然衰老和过期清除。

### 算法流程
1.  **初始化**：`Expired_Qty = 0`, `New_Batches = []`
2.  **遍历检查**：对仓库中的每一个 `batch`：
    *   计算剩余天数：$DaysLeft = batch.expiry\_day - t$
    *   **判定过期**：若 $DaysLeft \le 0$：
        *   该批次被标记为 **失效**。
        *   数量计入当日损耗：$Loss_t \leftarrow Loss_t + batch.qty$
        *   *该批次不被加入 `New_Batches` (即被永久移除)*。
    *   **判定存活**：若 $DaysLeft > 0$：
        *   保留该批次到 `New_Batches`。
3.  **更新**：`self.inventory_batches = New_Batches`

---

## 3. 出库策略：先失效先出 (FEFO)

当发生病人需求 ($Sales_t$) 时，系统遵循 **First-Expired-First-Out (FEFO)** 原则，通过优先消耗即将过期的药品来从物理上减少损耗。

### 算法流程
假设当日需求为 $D$，待处理批次列表为 $B$。

1.  **排序**：将 $B$ 按 `expiry_day` 从小到大排序 (升序)。
    $$ B_{sorted} = \text{Sort}(B, \text{key}=expiry\_day) $$
2.  **贪心扣减**：
    *   遍历 $B_{sorted}$ 中的每个批次 $b_i$。
    *   若 $D > 0$:
        *   消耗量 $Consume = \min(D, b_i.qty)$
        *   更新批次：$b_i.qty \leftarrow b_i.qty - Consume$
        *   更新需求：$D \leftarrow D - Consume$
    *   若 $b_i.qty = 0$，则该批次在下一轮会被移除。

---

## 4. 预测干预机制 (Forecast Intervention)

这是本系统最核心的**前馈控制 (Feedforward Control)** 创新点。监控模块不仅仅是被动记录过期，还会主动向补货算法发出“预警”，强制减少新货购入。

### 逻辑闭环
1.  **监控 (Monitor)**: 计算当前库存中最短剩余效期 $V_{min}$。
    $$ V_{min} = \min_{b \in B} (b.expiry\_day - t) $$
2.  **反馈 (Feedback)**: 将 $V_{min}$ 参数传递给 ARIMA 预测模块。
3.  **决策 (Decision)**: ARIMA 模块通过**平滑衰减函数**修正补货需求。

### 衰减函数 ($ \alpha $ Calculation)
为了避免因临期导致突然断货，我们采用了**线性软着陆**逻辑，并结合**波动性保护**。

*   **波动性底线 (Volatility Floor)**:
    *   高波动 ($CV > 0.5$) $\to Floor \approx 0.8$ (保守，怕断货)
    *   低波动 ($CV < 0.2$) $\to Floor \approx 0.4$ (激进，怕过期)
*   **计算公式**:
    $$ \alpha = 
    \begin{cases} 
    1.0 & \text{if } V_{min} > 60 \\
    Floor + \frac{1.0 - Floor}{60} \times V_{min} & \text{if } 0 < V_{min} \le 60 \\
    0.0 & \text{if } V_{min} \le 0
    \end{cases} 
    $$
*   **最终影响**:
    补货量 $Order$ 随 $\alpha$ 下降而平滑减少，迫使系统先消耗现有老库存，从而实现 **库存清理 (Inventory Clearance)** 且不造成剧烈震荡。
