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

## 2. 统计参考目标 (Statistical Reference Targets)

系统生成的数据将与以下论文基准进行对比验证，供分析使用（非强制约束）：

| 指标 (Metric) | 参考值 (Target) |说明 (Note) |
| :--- | :--- | :--- |
| **报损率 (Loss Rate)** | **17.2%** | 主要受高波动药品的有效效期 (`validity_days`) 和积压情况影响。 |
| **缺货率 (Stockout Rate)** | **3.1%** | 受安全库存系数 (`SafetyFactor`) 和需求波动 (`Noise`) 的共同影响。 |
| **周转天数 (Turnover Days)** | **44.6 Days** | 这一指标反映了库存周转效率，与订货周期相关。 |

---

## 3. 药品分类逻辑 (Categorization Logic)

系统采用随机或基于名称的规则进行波动性分类，参数配置如下：

*   **分类方法**: 优先匹配特定关键词（如感冒、慢病），其余采用哈希分布。

| 波动类别 (Volatility) | 特征 (Characteristics) | 算法参数配置 (默认) |
| :--- | :--- | :--- |
| **Low (低波动)** | CV < 0.2, 需求平稳 | `Noise=0.2`, `Season=0.2`, `Validity=720` |
| **Medium (中波动)** | 0.2 <= CV <= 0.5 | `Noise=0.6`, `Season=0.5`, `Validity=360` |
| **High (高波动)** | CV > 0.5, 需求剧烈 | `Noise=2.5`, `Season=1.0`, `Validity=180` |

> **注意**: 高波动药品 (High Volatility) 是产生损耗的主要来源，系统通过较短的效期和随机爆发 (`Burst`) 来模拟。

---

## 4. 开发工作流 (Workflow)

生成数据的标准流程如下：

1.  **运行生成脚本** (`src/scripts/generate_thesis_dataset.py`)。
    *   该脚本会自动调用 `SimulationTuner` (运行纯模拟) 和 `ThesisValidator` (进行后验证)。
2.  **查看验证报告** (`docs/THESIS_VALIDATION_REPORT_FINAL.md`)。
    *   检查各项指标与参考值的偏差。
3.  **调整参数 (可选)**。
    *   若偏差过大，可手动调整 `src/core/thesis_params.py` 中的默认参数，然后重新生成。
