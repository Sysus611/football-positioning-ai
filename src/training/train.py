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

    data = np.load(path)
    X = data["X"]   # [n_samples, obs_frames, feat_dim]
    Y = data["Y"]   # [n_samples, pred_frames, 2]

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

    # DataLoader: 把数据打包成小批量 (mini-batch)
    # TensorDataset 把 X 和 Y 配对
    # DataLoader 负责批量加载、打乱顺序
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,    # 每个 epoch 随机打乱顺序
    )

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 设备选择: 优先用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # 创建模型
    model = PlayerLSTM(
        input_dim=X_train.shape[2],
        pred_frames=pred_frames,
    ).to(device)   # .to(device) 把模型移到 GPU 或 CPU

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {param_count:,}")

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

    for epoch in range(1, epochs + 1):
        # --- 训练阶段 ---
        model.train()   # 切换到训练模式 (启用 Dropout)
        train_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            # 把数据移到设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播: 模型预测
            pred = model(batch_x)

            # 计算损失
            loss = criterion(pred, batch_y)

            # 反向传播 + 更新参数
            optimizer.zero_grad()   # 清除上一步的梯度
            loss.backward()         # 反向传播计算梯度
            # 梯度裁剪: 防止梯度爆炸 (LSTM 常见问题)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()        # 用梯度更新参数

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # --- 验证阶段 ---
        model.eval()    # 切换到评估模式 (关闭 Dropout)
        val_loss = 0.0
        n_val_batches = 0

        # torch.no_grad() 禁止梯度计算，节省内存和时间
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches

        # 更新学习率
        scheduler.step(val_loss)

        # 打印进度 (每 10 个 epoch)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {lr_now:.6f} | {elapsed:.0f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

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
            if patience_counter >= max_patience:
                print(f"  早停! 连续 {max_patience} 个 epoch 无改善")
                break

    elapsed = time.time() - start_time
    print(f"  完成! 最佳 epoch: {best_epoch}, "
          f"最佳 val_loss: {best_val_loss:.6f}, 耗时: {elapsed:.0f}s")


def main():
    config = load_config()
    data_dir = os.path.join(PROJECT_ROOT, "data", "tensors")
    model_dir = os.path.join(PROJECT_ROOT, "data", "models")

    # 检查是否指定了球员
    target_player = sys.argv[1] if len(sys.argv) > 1 else None

    if target_player:
        # 训练指定球员
        train_player(target_player, config, data_dir, model_dir)
    else:
        # 训练所有球员
        npz_files = sorted([
            f.replace(".npz", "")
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ])
        print(f"找到 {len(npz_files)} 个球员的训练数据")

        for pid in npz_files:
            train_player(pid, config, data_dir, model_dir)

    print(f"\n{'='*50}")
    print(f"全部训练完成! 模型目录: {model_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
