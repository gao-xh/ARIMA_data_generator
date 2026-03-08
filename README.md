# 改进型 ARIMA 库存优化系统 (Improved ARIMA Inventory System)

本项目旨在通过改进后的 ARIMA 模型（引入外部因子与效期衰减机制）来优化社区诊所的药品库存管理。项目包含一个完整的 **数据生成器** 和 **模型验证工具**，并通过图形用户界面（GUI）进行操作。

---

## 🚀 快速开始

### 启动方法
本项目内置了自动环境配置脚本，无需手动安装依赖。

1.  双击运行 **`run_app.bat`**。
2.  脚本会自动创建虚拟环境 (`.venv`) 并安装所需库。
3.  启动后会出现 GUI 界面。

> **注意**：如果遇到环境报错，请运行 `clean_setup.bat` 进行彻底重置。

---

## 🛠️ 功能模块

### 1. 数据生成器 (Simulation & Generator)

#### 方法 A: GUI 界面 (可视化调参)
在 "Simulation & Generator" 标签页中调整参数并点击生成。此模块已升级为 **平行世界 (Parallel Worlds)** 模拟架构，可同时生成 Baseline 和 Optimized 两组数据进行对比。

*   **平行对照逻辑**：
    *   **Baseline**：模拟人工经验补货（固定30天周期 + 历史均值 + 还有恐慌性紧急订货）。
    *   **Optimized**：使用改进型 **ARIMAX** 模型（动态周期 + AI预测 + 效期衰减）。
    *   **控制变量**：两组实验共享相同的随机种子，确保每日的各类随机外部事件（如气温、流感、客流）完全一致。

#### 方法 B: 批处理脚本 (推荐用于模型训练)
直接运行项目根目录下的 `src/scripts/generate_thesis_dataset.py` 脚本，可快速生成符合论文逻辑的全量数据集。
```bash
python src/scripts/generate_thesis_dataset.py
```
生成的数据将保存在 `data/processed/thesis_dataset_final.csv`，格式完全符合模板要求。

*   **核心逻辑**：
    *   **基础销量**：服从泊松分布 $P(\lambda)$。
    *   **季节性调整**：根据月度指数 (S-Index) 自动修正。
    *   **外部冲击**：$Demand_{Final} = Demand_{Base} \times (1 + \alpha_{Flu} + \beta_{Temp})$。

### 2. 模型验证 (Model Validation 标签页)
用于对比和评估改进型 ARIMA 模型的预测效果。

*   **输入**：
    *   点击 "Load Dataset" 加载生成的 `data/processed/synthetic_sales.csv`。
    *   选择具体的诊所 (Clinic) 和药品 (Drug)。

---

## 📄 算法亮点 (Thesis Highlights)

1.  **外部因子嵌入**：根据药品波动性自动引入环境因子（季节、气温、流感发病率）。
2.  **动态参数优化**：自动适配波动性类别，选择最优的 $(p,d,q)$ 参数组合。
3.  **效期衰减修正**：在订货公式中引入剩余效期约束，防止临期库存积压。
4.  **真实模拟场景**：包含了浮动提前期 (Floating Lead Time) 和信息滞后 (Information Lag) 等现实供应链约束。
