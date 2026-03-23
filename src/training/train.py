"""
LSTM 训练脚本 (单球员模型)

为指定球员训练 LSTM 跑位预测模型。
支持训练所有球员或指定单个球员。

使用方法:
    python src/training/train.py                # 训练所有球员
    python src/training/train.py home_11        # 只训练 home_11
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.model.lstm_baseline import PlayerLSTM


def load_config():
    config_path = os.path.join(PROJECT_ROOT, "src", "training", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_player_data(player_id: str, data_dir: str) -> tuple:
    """
    加载球员的训练数据。

    Args:
        player_id: 球员 ID (如 "home_11")
        data_dir: .npz 文件所在目录

    Returns:
        (X_train, Y_train, X_val, Y_val) torch Tensor
    """
    path = os.path.join(data_dir, f"{player_id}.npz")
    if not os.path.exists(path):
        return None, None, None, None

    print(f"  [DEBUG] 加载数据: {path}")
    data = np.load(path)
    X = data["X"]   # [n_samples, obs_frames, feat_dim]
    Y = data["Y"]   # [n_samples, pred_frames, 2]
    print(f"  [DEBUG] 原始数据 X: {X.shape}, Y: {Y.shape}")
    print(f"  [DEBUG] X 范围: [{X.min():.4f}, {X.max():.4f}], 均值: {X.mean():.4f}")
    print(f"  [DEBUG] Y 范围: [{Y.min():.4f}, {Y.max():.4f}], 均值: {Y.mean():.4f}")
    print(f"  [DEBUG] X 中 NaN 数: {np.isnan(X).sum()}, 零值占比: {(X==0).mean():.2%}")
    print(f"  [DEBUG] Y 中 NaN 数: {np.isnan(Y).sum()}, 零值占比: {(Y==0).mean():.2%}")

    # 按 80/20 划分训练集和验证集
    n = len(X)
    split = int(n * 0.8)

    # 随机打乱
    indices = np.random.permutation(n)
    X = X[indices]
    Y = Y[indices]

    X_train = torch.from_numpy(X[:split])
    Y_train = torch.from_numpy(Y[:split])
    X_val = torch.from_numpy(X[split:])
    Y_val = torch.from_numpy(Y[split:])

    print(f"  [DEBUG] 划分完成: 训练 {len(X_train)}, 验证 {len(X_val)}")
    return X_train, Y_train, X_val, Y_val


def train_player(player_id: str, config: dict, data_dir: str, model_dir: str):
    """
    训练一个球员的 LSTM 模型。

    训练流程讲解:
    1. 数据加载 → DataLoader (批量读取)
    2. 模型前向传播 → 计算预测值
    3. 计算损失 (MSE)
    4. 反向传播 → 自动计算梯度
    5. 优化器更新参数
    6. 重复 2-5 直到 loss 收敛

    Args:
        player_id: 球员 ID
        config: 配置字典
        data_dir: 训练数据目录
        model_dir: 模型保存目录
    """
    print(f"\n{'='*50}")
    print(f"训练 {player_id}")
    print(f"{'='*50}")

    # 加载数据
    X_train, Y_train, X_val, Y_val = load_player_data(player_id, data_dir)
    if X_train is None:
        print(f"  跳过: 无训练数据")
        return

    if len(X_train) < 50:
        print(f"  跳过: 样本太少 ({len(X_train)})")
        return

    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  输入维度: {X_train.shape[2]}")

    # 参数
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    pred_frames = config["window"]["pred_seconds"] * config["window"]["sample_rate"]

    print(f"  [CONFIG] batch_size={batch_size}, epochs={epochs}, lr={lr}")
    print(f"  [CONFIG] pred_frames={pred_frames}, obs_frames={X_train.shape[1]}")

    # DataLoader: 把数据打包成小批量 (mini-batch)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    print(f"  [DEBUG] DataLoader: 训练 {n_train_batches} batches, 验证 {n_val_batches} batches")

    # 设备选择: 优先用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [DEVICE] {device}")
    if device.type == "cuda":
        print(f"  [DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  [DEVICE] GPU Memory: {gpu_mem:.1f} GB")

    # 创建模型
    model = PlayerLSTM(
        input_dim=X_train.shape[2],
        pred_frames=pred_frames,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [MODEL] 参数量: {param_count:,}")
    print(f"  [MODEL] 结构: LSTM({X_train.shape[2]}->128, 2层) + FC(128->64->{pred_frames*2})")

    # 损失函数: MSE (均方误差)
    # 预测位置与真实位置的平方差的平均值
    criterion = nn.MSELoss()

    # 优化器: AdamW
    # 负责根据梯度更新模型参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 学习率调度器: ReduceLROnPlateau
    # 当验证 loss 不再下降时，自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 训练循环
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    max_patience = 25   # 早停: 连续 25 个 epoch 无改善就停止

    start_time = time.time()
    print(f"\n  {'Epoch':>5s} | {'Train Loss':>10s} | {'Val Loss':>10s} | {'Dist(m)':>8s} | {'LR':>10s} | {'Time':>6s} | Status")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}-+-------")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches

        # 把 MSE loss 换算成直观的距离误差 (米)
        # MSE 是归一化坐标的均方误差, 开根号×场地尺寸 ≈ 平均误差米数
        avg_dist_m = (val_loss ** 0.5) * ((105 + 68) / 2)

        scheduler.step(val_loss)

        # 每 5 个 epoch 或首末 epoch 打印
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time

        # 保存最佳模型
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            is_best = True

            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{player_id}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": X_train.shape[2],
                "pred_frames": pred_frames,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
                "player_id": player_id,
            }, model_path)
        else:
            patience_counter += 1

        # 打印进度 (每 5 个 epoch 或第 1 个)
        if epoch % 5 == 0 or epoch == 1 or is_best and epoch <= 5:
            lr_now = optimizer.param_groups[0]["lr"]
            status = "<< BEST" if is_best else f"patience {patience_counter}/{max_patience}"
            print(f"  {epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {avg_dist_m:6.2f}m | {lr_now:10.7f} | {elapsed:5.0f}s | {status}")

        if patience_counter >= max_patience:
            print(f"  [STOP] 早停! 连续 {max_patience} 个 epoch 无改善")
            break

    elapsed = time.time() - start_time
    best_dist_m = (best_val_loss ** 0.5) * ((105 + 68) / 2)
    print(f"  {'='*70}")
    print(f"  [DONE] {player_id} 训练完成!")
    print(f"  [DONE] 最佳 epoch: {best_epoch}/{epoch}")
    print(f"  [DONE] 最佳 val_loss: {best_val_loss:.6f} (~{best_dist_m:.2f}m 平均误差)")
    print(f"  [DONE] 总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  [DONE] 模型已保存: {model_dir}/{player_id}.pt")


def main():
    config = load_config()
    data_dir = os.path.join(PROJECT_ROOT, "data", "tensors")
    model_dir = os.path.join(PROJECT_ROOT, "data", "models")

    print(f"{'='*60}")
    print(f"  Football Player Positioning AI - LSTM Training")
    print(f"{'='*60}")
    print(f"  [ENV] Python: {sys.version.split()[0]}")
    print(f"  [ENV] PyTorch: {torch.__version__}")
    print(f"  [ENV] CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [ENV] GPU: {torch.cuda.get_device_name(0)}")
    print(f"  [ENV] 数据目录: {data_dir}")
    print(f"  [ENV] 模型目录: {model_dir}")

    # 检查是否指定了球员
    target_player = sys.argv[1] if len(sys.argv) > 1 else None

    total_start = time.time()

    if target_player:
        train_player(target_player, config, data_dir, model_dir)
    else:
        npz_files = sorted([
            f.replace(".npz", "")
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ])
        print(f"\n  [PLAN] 找到 {len(npz_files)} 个球员: {npz_files}")

        for i, pid in enumerate(npz_files):
            print(f"\n  [PROGRESS] ===== 球员 {i+1}/{len(npz_files)} =====")
            train_player(pid, config, data_dir, model_dir)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  全部训练完成!")
    print(f"  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  模型目录: {model_dir}")
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        print(f"  已训练模型: {len(models)} 个")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
