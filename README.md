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
在 "Simulation & Generator" 标签页中调整参数并点击生成。界面主要用于**小规模预览**参数效果。

#### 方法 B: 批处理脚本 (推荐用于模型训练)
直接运行项目根目录下的 `generate_dataset.py` 脚本，可快速生成全量数据集。
```bash
python generate_dataset.py
```
生成的数据将保存在 `data/processed/synthetic_sales.csv`，格式完全符合模板要求。

*   **参数调节**：
    *   **流感爆发阈值 (Flu Threshold)**：设定流感率达到多少时触发销量激增。
    *   **气温敏感度 (Temp Impact)**：设定低温对呼吸道药物销量的影响权重。
    *   **补货周期 (Replenishment Cycle)**：模拟诊所的订货频率（默认 14 天）。
*   **核心逻辑**：
    *   **基础销量**：服从泊松分布 $P(\lambda)$。
    *   **季节性调整**：根据月度指数 (S-Index) 自动修正。
    *   **外部冲击**：
        $$ Demand_{Final} = Demand_{Base} \times (1 + \alpha_{Flu} + \beta_{Temp}) $$
        确保在流感高发期或寒潮来袭时，数据能够呈现真实的爆炸性增长。

### 2. 模型验证 (Model Validation 标签页)
用于对比和评估改进型 ARIMA 模型的预测效果。

*   **输入**：
    *   点击 "Load Dataset" 加载生成的 `data/processed/synthetic_sales.csv`。
    *   选择具体的诊所 (Clinic) 和药品 (Drug)。
*   **仅针对验证的参数**：
    *   **剩余效期 (Validity Days)**：测试不同效期下的预测策略（效期越短，预测越保守）。
    *   **波动分类 (Volatility Class)**：手动指定药品的波动类型（低/中/高），强制模型采用不同的 (p,d,q) 参数组。
*   **结果展示**：
    *   **可视化图表**：真实销量 vs 预测销量趋势图。
    *   **核心指标**：
        *   **MAPE** (平均绝对百分比误差)：预测偏差率。
        *   **RMSE** (均方根误差)：误差绝对值。
        *   **R² Score**：模型拟合优度。
        *   **Explained Variance**：解释方差。

---

## 📋 目录结构

```
ARIMA/
├── data lib/               # 原始数据字典 (药品信息、外部因子)
├── data/processed/         # [自动生成] 存放生成好的销售数据
├── src/
│   ├── core/               # 核心算法 (生成器逻辑, 常量定义)
│   ├── models/             # 模型实现 (ImprovedARIMA 类)
│   ├── ui/                 # 界面代码 (PySide6)
│   │   ├── common/         # 通用组件 (图表, 滑动条)
│   │   ├── generation/     # 生成器界面
│   │   └── validation/     # 验证界面
│   └── evaluation/         # 评价指标计算
├── run_app.bat             # [入口] 启动脚本
├── clean_setup.bat         # [工具] 环境清理脚本
└── requirements.txt        # 依赖列表
```

## 📝 技术细节

### 核心算法：ImprovedARIMA
模型基于 `statsmodels` 库实现，针对传统 ARIMA 进行了以下改进：

1.  **动态阶数选择**：
    *   **低波动** (CV < 0.2)：使用 ARIMA(1, 0, 1)，仅考虑季节性。
    *   **中波动** (0.2 ≤ CV ≤ 0.5)：使用 ARIMA(2, 1, 2)，引入气温、降水、流感率。
    *   **高波动** (CV > 0.5)：使用 ARIMA(3, 1, 3)，全量引入所有外部变量。

2.  **效期衰减系数 (Validity Decay)**：
    *   当库存即将过期（如剩余效期 < 30 天）时，为了避免呆滞库存，预测值会被自动调低。
    *   衰减公式：$\alpha = \alpha_0 \times (1 + 0.2 \times CV')$

---
**版本**: 1.0.0
**最后更新**: 2026-03-06
