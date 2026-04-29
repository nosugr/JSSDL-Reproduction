# JSSDL Reproduction

基于论文《Industrial Process Modeling and Monitoring Based on Jointly Specific and Shared Dictionary Learning》的复现项目。

本仓库按 `document/JSSDL_Project_Structure.pdf` 中给出的结构组织，提供：

- JSSDL 主算法实现
- 数值仿真实验
- PCA / Robust PCA / 传统字典学习对比基线
- 单元测试与可视化工具

## 目录概览

```text
.
├── config.yaml
├── requirements.txt
├── data/
├── jssdl/
├── baselines/
├── experiments/
├── tests/
├── notebooks/
└── outputs/
```

## 安装

推荐使用项目自带虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 快速开始

数值仿真实验：

```powershell
.\.venv\Scripts\python.exe experiments/exp1_numerical.py
```

参数敏感性分析：

```powershell
.\.venv\Scripts\python.exe experiments/sensitivity_analysis.py
```

单元测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 数据说明

- `data/raw/numerical/`：数值仿真数据，可由 `exp1_numerical.py` 自动生成

## 实现说明

- JSSDL 采用论文中的交替迭代优化流程
- `D1` 使用行级硬阈值保留每个原子的稀疏支撑
- `P` 使用奇异值软阈值更新
- `D2` 使用闭式解更新
- 稀疏编码采用三步 OMP 更新
- 监测阈值默认通过 KDE 估计，若环境缺少 `scipy` 则自动退化为经验分位数

## 输出

运行实验后会在 `outputs/` 下生成：

- `figures/`：监测曲线、热力图、敏感性分析图
- `tables/`：FDR / FAR 指标表
- `checkpoints/`：训练得到的字典与阈值
