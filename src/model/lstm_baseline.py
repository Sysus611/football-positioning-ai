"""
LSTM Baseline 模型定义 (单球员跑位预测)

模型结构:
  输入 → LSTM(多层) → 全连接层 → 输出

  输入: [batch, obs_frames, feat_dim]  = [batch, 75, 48]
  输出: [batch, pred_frames, 2]        = [batch, 50, 2]

LSTM 语法讲解:
  LSTM 是一种循环神经网络 (RNN)，专门处理时间序列。
  它逐时间步读取输入，记住过去的信息，生成隐藏状态。

  nn.LSTM 参数:
    - input_size: 每个时间步的输入维度 (48)
    - hidden_size: LSTM 内部隐藏状态的维度 (128)
    - num_layers: 堆叠几层 LSTM (2)
    - batch_first=True: 输入 shape 为 [batch, seq_len, features]
    - dropout: 层间 dropout 概率，防止过拟合
"""

import torch
import torch.nn as nn


class PlayerLSTM(nn.Module):
    """
    单球员跑位预测 LSTM 模型。

    nn.Module 是 PyTorch 中所有神经网络的基类。
    自定义模型需要:
      1. 继承 nn.Module
      2. 在 __init__ 中定义网络层
      3. 在 forward 中定义前向计算逻辑
    """

    def __init__(
        self,
        input_dim: int = 48,      # 输入特征维度
        hidden_dim: int = 128,    # LSTM 隐藏层维度
        num_layers: int = 2,      # LSTM 层数
        pred_frames: int = 50,    # 预测帧数
        dropout: float = 0.2,     # Dropout 概率
    ):
        """
        初始化模型各层。

        super().__init__() 是 Python 继承语法，必须在子类的 __init__ 中调用。
        它初始化父类 nn.Module 的内部状态。
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_frames = pred_frames

        # LSTM 编码器: 读取过去 75 帧的上下文
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,     # 让输入 shape 为 [batch, seq, feat]
            dropout=dropout if num_layers > 1 else 0,
        )

        # 全连接解码器: 把 LSTM 的隐藏状态映射为未来位置
        # nn.Sequential 把多个层按顺序串联
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),    # 线性变换: 128维 → 64维
            nn.ReLU(),                     # 激活函数: max(0, x)
            nn.Dropout(dropout),           # 随机丢弃一部分神经元，防止过拟合
            nn.Linear(64, pred_frames * 2),  # 64维 → 100维 (50帧 × 2坐标)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输入 → LSTM → 全连接 → 输出。

        forward 方法定义了数据如何流过网络。PyTorch 会自动跟踪计算图
        用于反向传播 (自动求导)。

        Args:
            x: 输入张量 [batch, obs_frames, input_dim]

        Returns:
            预测位置 [batch, pred_frames, 2]
        """
        batch_size = x.size(0)

        # LSTM 编码
        # lstm_out: 每个时间步的输出 [batch, obs_frames, hidden_dim]
        # (h_n, c_n): 最后一个时间步的隐藏状态和细胞状态
        lstm_out, (h_n, c_n) = self.encoder(x)

        # 取最后一个时间步的输出作为整个序列的编码
        # lstm_out[:, -1, :] 选取序列最后一帧，shape = [batch, hidden_dim]
        last_hidden = lstm_out[:, -1, :]

        # 全连接解码: [batch, hidden_dim] → [batch, pred_frames * 2]
        output = self.decoder(last_hidden)

        # 重塑输出: [batch, pred_frames * 2] → [batch, pred_frames, 2]
        # view 是 PyTorch 的 reshape 操作
        output = output.view(batch_size, self.pred_frames, 2)

        return output


if __name__ == "__main__":
    # 快速测试模型是否能正常前向传播
    model = PlayerLSTM(input_dim=48, pred_frames=50)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟一个 batch 的输入
    dummy_input = torch.randn(4, 75, 48)   # batch=4, 75帧, 48维
    output = model(dummy_input)
    print(f"输入: {dummy_input.shape}")
    print(f"输出: {output.shape}")
