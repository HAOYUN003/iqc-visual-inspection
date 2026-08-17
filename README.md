# IQC 智能来料检测与质量追溯系统

面向半导体设备厂来料检验（IQC）的 AI 视觉检测原型系统，融合 **光学成像 × 深度学习 × 质量工程**，打通"采集 → AI检测 → 人工复核 → 质量追溯"的完整闭环。

> Phase 0（当前）：软件原型，用手机/电脑镜头 + 模拟/公开数据打通流程，**网页默认本地免费引擎**。
> Phase 1（后续）：工业相机 + 镜头 + 光源，尺寸精密测量，落地产线（见 `docs/光学方案设计.md`）。

---

## 系统架构

```
供应商来料 → 图像采集（上传/拍照） → AI 初筛
  ├─ 本地免费引擎（默认，RTX 4060）：规格CNN + 尺寸反推 + 缺陷YOLO + 螺纹分类
  ├─ 视觉大模型引擎（可选，~0.02元/件）：Qwen-VL 读卡尺LCD / 加工件多角度
  └─ 判定 OK / NG / UNSURE → 入库 → 人工复核 → 复核回灌重训 → 越用越准
```

### 检测能力

| 模块 | 功能 | 判定方式 |
|---|---|---|
| 规格防错 | 标准件 M3~M8 识别，防拿错混料 | 尺寸反推（主）/ CNN 兜底 |
| 表面缺陷 | 划伤/麻面/裂纹/锈蚀/毛刺等 | 本地 YOLO / 视觉大模型 |
| 螺纹状态 | 侧视图 正常/缺牙/烂牙 | 本地 CNN 分类 |
| 尺寸测量 | 外径/孔径，超差判定 | 卡尺读数或 CV + 公差比对 |
| 图纸清单校验 | 按检验标准逐项判定 | 表面类视觉 + 尺寸类实测值量化判定 |

### 双端界面

- **检验员端**：登记批次 → 检测模式（标准件/加工件/螺纹）→ 引擎选择 → 检测 → 入库 → 复核
- **主管端**：质量总览（判定/缺陷/良率趋势/复核一致性）、物料白名单、检验标准编辑器、追溯查询、数据回灌

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

### 3. 训练本地模型（免费，用自己 GPU）

```bash
# 规格识别（ResNet18）
python src/detection/spec_model.py --epochs 25 --backbone resnet18
# 紧固件表面缺陷（YOLO）
python src/detection/fastener_defect_model.py --train --epochs 40
# 螺纹状态（CNN）
python src/detection/thread_model.py --train --epochs 30
```

### 4. 启动界面

```bash
streamlit run app/app.py
```

浏览器打开 → 检验员端登记批次、检测；主管端配置白名单/检验标准、看报表。

---

## 项目结构

```
iqc_vision/
├── app/app.py                  # Streamlit 双端界面
├── src/
│   ├── config.py               # 全局配置 + 中文路径兼容 patch_cv_io
│   ├── cv_io.py                # OpenCV 中文路径读写兼容层
│   ├── data/
│   │   ├── quality_db.py       # SQLite 质检数据库（批次/记录/物料白名单/追溯）
│   │   └── make_standard_parts.py  # 标准件模拟数据集生成器
│   ├── detection/
│   │   ├── engine.py           # 引擎统一入口（本地免费 / 视觉大模型）
│   │   ├── pipeline.py         # 本地引擎：规格+尺寸+缺陷+螺纹
│   │   ├── spec_model.py       # 规格识别（CNN/ResNet18）
│   │   ├── fastener_defect_model.py  # 紧固件缺陷（YOLO）
│   │   ├── thread_model.py     # 螺纹状态（CNN）
│   │   ├── dimension_model.py  # 尺寸测量（CV 亚像素）
│   │   └── vision_detector.py  # 视觉大模型引擎（Qwen-VL）
│   ├── inspection_checklist.py # 检验标准（位置+阈值+量化判定）
│   ├── export_training.py      # 复核数据导出训练集（数据回灌）
│   ├── report.py               # 报表与追溯（趋势/一致性/缺陷明细）
│   └── bench_engine.py         # 本地 vs 视觉 API 精度对比
├── data/                       # 数据、数据库、训练集
├── models/                     # 训练好的本地模型权重
├── docs/光学方案设计.md         # Phase1 光学成像方案（镜头/光源/标定）
└── requirements.txt
```

---

## 技术亮点（简历 / 答辩用）

1. **光学成像 × AI 融合叙事**：先算光学（最小缺陷 → 物面分辨率 → 放大倍率 → 镜头/光源），再谈 AI
2. **AI 不"猜"尺寸**：尺寸走卡尺读数/亚像素 CV + 公差比对，AI 只做外观分类，二者分工
3. **本地免费引擎**：规格/缺陷/螺纹/尺寸全走本地模型（RTX 4060），视觉大模型仅疑难件可选
4. **检验标准结构化**：检验部位 + 图纸阈值 + 量化判定，机器判定覆盖视觉 UNSURE
5. **质量数据资产**：批次/供应商/料号全程留痕，复核回灌 → 本地模型越用越准
6. **AI 初筛 + 工程师决策**：不是替代检验员，而是降低负担、提高一致性

---

## 已知边界与后续

- **本地缺陷模型**：当前用程序生成数据训练，真实照片误检偏高 → 靠复核回灌攒真实缺陷数据后重训
- **卡尺读数**：OCR 对七段数码管识别率低 → 用视觉大模型读（0.02元/件），或 Phase1 做专用数码管识别器
- **Phase 1**：工业相机 + 治具固定工位 + 光源 → 像素级 ROI、亚像素精密测量（详见 `docs/光学方案设计.md`）
