"""
特征工程 + 滑动窗口切片 (单球员模型)

为每个球员生成独立的训练数据:
  输入: 其他 21 名球员坐标 + 球坐标 + 目标球员当前坐标 (上下文)
  输出: 目标球员未来 2 秒的位置轨迹

输出 .pt 文件，每个球员一个，可直接用 PyTorch DataLoader 加载。

使用方法:
    python src/features/build_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def load_config():
    """加载配置"""
    config_path = os.path.join(PROJECT_ROOT, "src", "training", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_player_ids(df: pd.DataFrame) -> list:
    """
    从 DataFrame 列名提取球员 ID 列表。
    返回格式: ["home_11", "home_1", ..., "away_25", ...]
    """
    ids = []
    for col in df.columns:
        if col.endswith("_x") and not col.startswith("ball") and not col.startswith("game"):
            pid = col[:-2]  # "home_11_x" -> "home_11"
            ids.append(pid)
    return ids


def build_context_features(
    df: pd.DataFrame,
    target_pid: str,
    all_pids: list,
    pitch_length: float,
    pitch_width: float,
    sample_rate: int,
) -> np.ndarray:
    """
    为目标球员构建每帧的上下文特征向量。

    每帧特征 = [其他21人的(x,y), 球的(x,y), 目标球员的(x,y, vx,vy)]
    维度 = 21*2 + 2 + 4 = 48

    Args:
        df: 预处理后的 DataFrame
        target_pid: 目标球员 ID (如 "home_11")
        all_pids: 所有球员 ID 列表
        pitch_length: 场地长度 (米)
        pitch_width: 场地宽度 (米)
        sample_rate: 采样频率 (Hz)

    Returns:
        特征矩阵 [n_frames, feature_dim]
    """
    n_frames = len(df)
    N_OTHER_SLOTS = 21  # 固定 21 个"其他球员"slot，保证特征维度一致

    # --- 其他球员的坐标 (固定 21 人 × 2 = 42 维) ---
    # 不同比赛球员数不同 (22~28)，但特征维度必须固定
    # 多余的截断，不足的补零
    other_pids = [pid for pid in all_pids if pid != target_pid]
    other_coords = np.zeros((n_frames, N_OTHER_SLOTS * 2), dtype=np.float32)

    for i, pid in enumerate(other_pids[:N_OTHER_SLOTS]):
        x = df[f"{pid}_x"].values
        y = df[f"{pid}_y"].values
        other_coords[:, i * 2] = x
        other_coords[:, i * 2 + 1] = y

    # --- 球的坐标 (2维) ---
    ball_x = df["ball_x"].values
    ball_y = df["ball_y"].values

    # --- 目标球员的坐标和速度 (4维) ---
    target_x = df[f"{target_pid}_x"].values
    target_y = df[f"{target_pid}_y"].values

    # 速度 = 位移差分
    # np.diff 计算相邻元素的差值，结果比原数组少 1 个元素
    # np.insert 在开头补 0，使长度一致
    target_vx = np.insert(np.diff(target_x), 0, 0.0)
    target_vy = np.insert(np.diff(target_y), 0, 0.0)

    # 拼接: [42 + 2 + 4 = 48 维]
    tail_features = np.column_stack([ball_x, ball_y, target_x, target_y, target_vx, target_vy])
    all_features = np.hstack([other_coords, tail_features])

    # 替换 NaN 为 0
    all_features = np.nan_to_num(all_features, nan=0.0)

    return all_features.astype(np.float32)


def build_target(df: pd.DataFrame, target_pid: str) -> np.ndarray:
    """
    提取目标球员的位置序列 (用于构建预测标签)。

    Returns:
        位置矩阵 [n_frames, 2]  (x, y)
    """
    x = df[f"{target_pid}_x"].values
    y = df[f"{target_pid}_y"].values
    target = np.column_stack([x, y])
    target = np.nan_to_num(target, nan=0.0)
    return target.astype(np.float32)


def sliding_window(
    features: np.ndarray,
    target_pos: np.ndarray,
    obs_frames: int,
    pred_frames: int,
    stride: int,
) -> tuple:
    """
    滑动窗口切片，生成训练样本。

    从连续的时间序列中，按固定窗口大小和步长切出样本对:
      X = features[t : t + obs_frames]         输入 (观测窗口)
      Y = target_pos[t + obs_frames : t + obs_frames + pred_frames]  标签 (预测窗口)

    ┌──────[obs 75帧]──────┬──────[pred 50帧]──────┐
    │      输入 X          │      输出 Y            │
    └──────────────────────┴───────────────────────┘
                 ← stride=5 →  滑动

    Args:
        features: 上下文特征 [n_frames, feat_dim]
        target_pos: 目标球员位置 [n_frames, 2]
        obs_frames: 观测窗口帧数
        pred_frames: 预测窗口帧数
        stride: 滑动步长

    Returns:
        (X, Y):
          X shape = [n_samples, obs_frames, feat_dim]
          Y shape = [n_samples, pred_frames, 2]
    """
    total = len(features)
    window = obs_frames + pred_frames

    X_list = []
    Y_list = []

    for start in range(0, total - window + 1, stride):
        x = features[start : start + obs_frames]
        y = target_pos[start + obs_frames : start + obs_frames + pred_frames]

        # 跳过包含太多零值的样本 (球员可能不在场)
        if np.mean(y == 0) > 0.3:
            continue

        X_list.append(x)
        Y_list.append(y)

    if not X_list:
        return np.array([]), np.array([])

    X = np.stack(X_list)   # [n_samples, obs_frames, feat_dim]
    Y = np.stack(Y_list)   # [n_samples, pred_frames, 2]

    return X, Y


def build_player_dataset_single_game(
    df: pd.DataFrame,
    target_pid: str,
    config: dict,
) -> tuple:
    """
    从单场比赛构建一个球员的训练数据。

    重要: 不同比赛的同号球员是不同的人!
    所以必须按单场比赛生成数据，不能跨场合并。

    划分策略: 按 10 分钟循环 (8 分钟训练 + 2 分钟验证)
    ┌──TRAIN 8min──┬──VAL 2min──┬──TRAIN 8min──┬──VAL 2min──┬...
    这样训练数据占 80%，且验证集分布在各个时段。
    不需要单独的测试集（学习的是单个球员的个人跑位习惯）。

    Args:
        df: 单场比赛的 DataFrame
        target_pid: 目标球员 ID
        config: 配置字典

    Returns:
        (X_train, Y_train, X_val, Y_val) numpy 数组
    """
    obs_frames = config["window"]["obs_seconds"] * config["window"]["sample_rate"]
    pred_frames = config["window"]["pred_seconds"] * config["window"]["sample_rate"]
    stride = config["window"]["stride_frames"]
    sample_rate = config["window"]["sample_rate"]
    all_pids = get_player_ids(df)

    # 检查目标球员是否在比赛中
    if f"{target_pid}_x" not in df.columns:
        return None, None, None, None

    # 检查有效率
    active_col = f"{target_pid}_active"
    if active_col in df.columns:
        active_ratio = df[active_col].mean()
        if active_ratio < 0.5:
            print(f"    跳过: 有效率 {active_ratio:.0%} < 50%")
            return None, None, None, None

    # 循环划分参数 (单位: 帧)
    TRAIN_MINUTES = 8
    VAL_MINUTES = 2
    CYCLE_FRAMES = (TRAIN_MINUTES + VAL_MINUTES) * 60 * sample_rate
    TRAIN_FRAMES = TRAIN_MINUTES * 60 * sample_rate

    train_X_list, train_Y_list = [], []
    val_X_list, val_Y_list = [], []

    # 按半场分别处理 (避免跨中场休息的窗口)
    for period in sorted(df["period"].unique()):
        period_df = df[df["period"] == period].reset_index(drop=True)
        n_frames = len(period_df)

        features = build_context_features(
            period_df, target_pid, all_pids,
            config["pitch"]["length"],
            config["pitch"]["width"],
            sample_rate,
        )
        target_pos = build_target(period_df, target_pid)

        # 先生成所有滑动窗口样本
        X_all, Y_all = sliding_window(features, target_pos, obs_frames, pred_frames, stride)

        if len(X_all) == 0:
            continue

        # 每个样本按其起始帧在循环中的位置分配到 train 或 val
        for i in range(len(X_all)):
            # 样本起始帧 = i * stride
            start_frame = i * stride
            pos_in_cycle = start_frame % CYCLE_FRAMES

            if pos_in_cycle < TRAIN_FRAMES:
                train_X_list.append(X_all[i])
                train_Y_list.append(Y_all[i])
            else:
                val_X_list.append(X_all[i])
                val_Y_list.append(Y_all[i])

    if not train_X_list or not val_X_list:
        return None, None, None, None

    X_train = np.stack(train_X_list)
    Y_train = np.stack(train_Y_list)
    X_val = np.stack(val_X_list)
    Y_val = np.stack(val_Y_list)

    ratio = len(X_train) / (len(X_train) + len(X_val))
    print(f"    划分: {len(X_train)} train + {len(X_val)} val ({ratio:.0%}/{1-ratio:.0%})")

    return X_train, Y_train, X_val, Y_val


def main():
    """
    为每场比赛的每个球员构建训练数据。

    输出文件命名: game1_home_11.npz (比赛名_球员ID)
    每个文件包含: X_train, Y_train, X_val, Y_val
    """
    config = load_config()
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    output_dir = os.path.join(PROJECT_ROOT, "data", "tensors")
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有 Parquet 文件
    parquet_files = sorted([
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir)
        if f.endswith(".parquet")
    ])

    print(f"数据文件: {[os.path.basename(p) for p in parquet_files]}")
    print(f"观测窗口: {config['window']['obs_seconds']}s = {config['window']['obs_seconds'] * config['window']['sample_rate']} 帧")
    print(f"预测窗口: {config['window']['pred_seconds']}s = {config['window']['pred_seconds'] * config['window']['sample_rate']} 帧")
    print(f"滑动步长: {config['window']['stride_frames']} 帧")

    total_datasets = 0

    for parquet_path in parquet_files:
        game_id = os.path.basename(parquet_path).replace(".parquet", "")
        df = pd.read_parquet(parquet_path)
        all_pids = get_player_ids(df)

        print(f"\n{'='*50}")
        print(f"处理 {game_id}: {len(all_pids)} 个球员")
        print(f"{'='*50}")

        for pid in all_pids:
            print(f"\n  --- {game_id}_{pid} ---")

            result = build_player_dataset_single_game(df, pid, config)
            if result[0] is None:
                print(f"    跳过: 无有效样本")
                continue

            X_train, Y_train, X_val, Y_val = result

            # 保存为 .npz ，包含已划分的 train/val
            output_path = os.path.join(output_dir, f"{game_id}_{pid}.npz")
            np.savez_compressed(
                output_path,
                X_train=X_train, Y_train=Y_train,
                X_val=X_val, Y_val=Y_val,
            )

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"    训练集: {X_train.shape[0]} 样本")
            print(f"    验证集: {X_val.shape[0]} 样本")
            print(f"    X shape: {X_train.shape}  (样本, 帧, 特征)")
            print(f"    已保存: {output_path} ({size_mb:.1f} MB)")
            total_datasets += 1

    print(f"\n{'='*50}")
    print(f"特征工程完成! 共 {total_datasets} 个数据集")
    print(f"输出: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

