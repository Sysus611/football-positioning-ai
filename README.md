# Football Positioning AI

基于深度学习的足球球员跑位AI，通过连续时间段内球员在场上的相对位置信息训练模型。

## 项目结构

```
football-positioning-ai/
├── data/
│   ├── raw/              # 原始追踪数据 (Metrica CSV / SkillCorner JSONL)
│   ├── processed/        # 预处理后的 Parquet/HDF5
│   └── tensors/          # 训练样本 .pt
├── src/
│   ├── extraction/       # 视频 → 坐标提取 (Phase 3)
│   ├── preprocessing/    # 数据预处理 (归一化/插值/校验)
│   ├── features/         # 特征工程 (相对位置/速度/滑动窗口)
│   ├── model/            # 模型定义 (LSTM → Transformer)
│   ├── training/         # 训练脚本与配置
│   └── export/           # ONNX 导出
├── notebooks/            # 探索性分析
└── tests/
```

## 数据来源

- [Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data) — 3场比赛, CSV, 25Hz
- [SkillCorner Open Data](https://github.com/SkillCorner/opendata) — 9场比赛, JSONL

## 实施路线

| 阶段 | 内容 |
|------|------|
| Phase 1 | Metrica 数据 → 预处理 → LSTM baseline |
| Phase 2 | Transformer + GNN 架构 |
| Phase 3 | 视频提取管线 (可选) |
| Phase 4 | ONNX 导出 → 游戏引擎集成 |
