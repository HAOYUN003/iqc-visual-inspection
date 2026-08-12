# IQC 智能来料检测与质量追溯系统

面向半导体设备厂来料检验（IQC）的 AI 视觉检测原型系统，融合 **光学成像 × 深度学习 × 质量工程**，打通"采集 → AI检测 → 人工复核 → 质量追溯"的完整闭环。

> Phase 0（当前）：纯软件原型，用手机/电脑镜头 + 模拟/公开数据打通流程。
> Phase 1（后续）：工业相机 + 镜头 + 光源，尺寸精密测量，落地产线。

---

## 系统架构

```
供应商来料 → 自动拍摄/图像采集 → 质检数据库（料号/批次/供应商/照片）
         → AI 初筛 ─┬─ 正常品自动通过
                    └─ 异常品人工复核 → 复核结果回灌 → 模型持续优化
```

### 四大模块

| 模块 | 功能 | 技术 |
|---|---|---|
| 规格防错 | 识别标准件规格（M3/M4/M5/M6/M8），防止拿错混料 | 传统CV判型+测径反推（主） / CNN 兜底 |
| 表面缺陷检测 | 检测划伤/裂纹/麻面/氧化皮等缺陷 | YOLOv8（紧固件域模型）+ 程序生成数据 |
| 尺寸测量 | 亚像素测外径/孔径，超差判定 | 传统 CV + 标定系数 |
| 质量追溯 | 批次/供应商/料号/时间维度追溯 | SQLite |
| 报表分析 | 供应商质量排行、缺陷分布、报告导出 | Streamlit + pandas |

### 双端界面

- **检验员端**：登记批次 → 拍照/上传 → AI 检测 → 记录入库 → 人工复核
- **主管端**：质量总览、供应商/批次报表、追溯查询、报告导出

---

## 快速开始

### 1. 环境

```bash
conda create -n iqc python=3.11 -y
conda activate iqc
pip install -r requirements.txt
# 若用 GPU（推荐 RTX 系列）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. 准备数据（Phase 0 无硬件）

```bash
# 生成标准件规格模拟数据（M3~M8 螺钉/垫片/螺母）
python src/data/make_standard_parts.py

# 下载 NEU-DET 表面缺陷公开数据集（自动转 YOLO 格式）
python src/detection/defect_model.py --download
```

### 3. 训练模型

```bash
# 规格识别（ResNet18，约3分钟 GPU）
python src/detection/spec_model.py --epochs 25 --backbone resnet18

# 缺陷检测（YOLOv8n，约5-10分钟 GPU）
python src/detection/defect_model.py --train --epochs 40
```

### 4. 启动界面

```bash
streamlit run app/app.py
```

浏览器打开 → 检验员端登记批次、上传/拍照零件 → 开始检测 → 记录入库；主管端查看统计报表与追溯。

---

## 项目结构

```
iqc_vision/
├── app/app.py                 # Streamlit 双端界面
├── src/
│   ├── config.py              # 全局配置
│   ├── data/
│   │   ├── quality_db.py      # SQLite 质检数据库（批次/记录/追溯）
│   │   └── make_standard_parts.py  # 标准件模拟数据集生成器
│   ├── detection/
│   │   ├── spec_model.py      # 规格识别（CNN/ResNet18）
│   │   ├── defect_model.py    # 缺陷检测（YOLOv8 + NEU-DET）
│   │   └── pipeline.py        # 统一检测流水线
│   └── report.py              # 报表与追溯
├── data/                      # 数据与数据库
├── models/                    # 训练好的模型权重
├── docs/光学方案设计.md        # Phase1 光学成像方案（镜头/光源/标定）
└── requirements.txt
```

---

## 技术亮点（简历 / 答辩用）

1. **光学成像 × AI 融合叙事**：先算光学（最小缺陷 → 物面分辨率 → 放大倍率 → 镜头/光源选型），再谈 AI（见 `docs/光学方案设计.md`）
2. **AI 不"猜"尺寸**：尺寸测量走传统 CV 亚像素边缘 + 标定，AI 只做外观分类/检测，二者分工。**规格防错 = 判型 + 测外径 → 反推规格**（传统CV主通道，确定性可解释），CNN 仅兜底
3. **质量数据资产**：批次/供应商/料号全程留痕，支持追溯与持续优化（回灌训练）
4. **AI 初筛 + 工程师决策**：不是替代检验员，而是降低人工负担、提高一致性

---

## Phase 1 展望

- 工业相机 + 低角度环形光 + 背光 + 标定板 → 尺寸测量与缺陷检测一体化
- 真实零件数据采集 → 同架构重训练（数据源无缝替换）
- 对接企业 MES/ERP，节拍优化与产线集成

> 详细光学选型计算见 [docs/光学方案设计.md](docs/光学方案设计.md)
