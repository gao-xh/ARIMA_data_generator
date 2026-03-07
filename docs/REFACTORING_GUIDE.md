# 项目重构与开发指南 (Refactoring & Development Guide)

本文档记录了项目在 **2025-03-06** 进行的架构重构要求及核心统计约束，作为后续开发的基准。

## 1. 架构重构要求 (Architectural Requirements)

为了解决单文件 (`generator_v2.py`) 难以维护且无法进行自检 (Check-first) 的问题，系统已被拆分为模块化的工具包。

### 1.1 模块化结构
代码需严格遵循以下分层结构：

1.  **Generation Algorithm (生成算法)**:
    *   **位置**: `src/core/algorithms/demand.py`
    *   **职责**: 仅负责 **Operator A** (需求生成)。
    *   **逻辑**: 基于季节性、气温、流感及随机噪声生成每日需求。
    
2.  **Recursive Algorithm (递归/控制算法)**:
    *   **位置**: `src/core/algorithms/inventory_control.py`
    *   **职责**: 仅负责 **Operator B** (库存控制)。
    *   **逻辑**: 实现周期性盘点 (Periodic Review) 策略，计算补货量 (`OrderQty`) 和安全库存 (`SafetyStock`)。
    
3.  **Transition Kernel (状态转移核心)**:
    *   **位置**: `src/core/algorithms/mcmc_transition.py`
    *   **职责**: **Operator T** (状态转移)。
    *   **逻辑**: 协调 Demand 和 Inventory Control，维护每日的 `Inventory`, `Pipeline`, `Stockout` 状态。

4.  **Self-Check Tool (自检工具)**:
    *   **位置**: `src/core/tools/validator.py`
    *   **职责**: 在大规模生成数据前，对小样本进行仿真，验证是否符合统计指标。

---

## 2. 统计指标约束 (Statistical Constraints)

所有算法调整必须以满足以下论文基准为目标（允许极小误差）：

| 指标 (Metric) | 目标值 (Target) |说明 (Note) |
| :--- | :--- | :--- |
| **报损率 (Loss Rate)** | **17.2%** | 通过缩短高波动药品的有效效期 (`validity_days` -> 180) 和增加积压实现。 |
| **缺货率 (Stockout Rate)** | **3.1%** | 即使在高波动下也需保持较低缺货，要求 `SafetyFactor` 对 High 类设为 2.0+。 |
| **周转天数 (Turnover Days)** | **44.6 Days** | 通过 `ReplenishmentPeriod` (R=30) 和较高的安全库存系数 (`SafetyFactor` ~ 2.0) 锚定。 |

---

## 3. 药品分类逻辑 (Categorization Logic)

为确保统计结果的确定性，必须严格执行以下分类比例（共 128 SKUs）：

*   **分类方法**: 使用 `hash(DrugName) % 128` 进行确定性哈希分类。

| 波动类别 (Volatility) | 数量 (Count) | 特征 (Characteristics) | 算法参数配置 |
| :--- | :--- | :--- | :--- |
| **Low (低波动)** | **41** | CV < 0.2, 需求平稳 | `Noise=0.2`, `Season=0.2`, `Validity=720` |
| **Medium (中波动)** | **63** | 0.2 <= CV <= 0.5 | `Noise=0.6`, `Season=0.5`, `Validity=360` |
| **High (高波动)** | **24** | CV > 0.5, 需求剧烈 | `Noise=2.5`, `Season=1.0`, `Validity=180` (易过期) |

> **注意**: 高波动药品 (High Volatility) 是产生 17.2% 报损率的主要来源，主要通过较短的效期和极高的安全库存（导致积压）来实现。

---

## 4. 开发工作流 (Workflow)

鉴于严格的统计要求，后续修改必须遵循以下流程：

1.  **修改代码** (`algorithms/` 目录)。
2.  **运行自检** (`tools/validator.py`)。
3.  **确认指标** (Loss ~17.2%, Stockout ~3.1%)。
4.  **全量生成**。
