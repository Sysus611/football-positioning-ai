"""
数据预处理主管线

流程:
1. 解析原始数据 (CSV / EPTS) → 统一 DataFrame
2. 坐标归一化 (Metrica 已经是 [0,1]，验证即可)
3. 数据质量校验 (速度异常/缺失帧过滤)
4. 输出为 Parquet 格式 (按比赛存储)

使用方法:
    python -m src.preprocessing.preprocess
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from typing import List, Tuple

# 添加项目根目录到 sys.path，使 import 能正常工作
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.parsers import parse_game


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件。

    Args:
        config_path: config.yaml 路径，默认使用项目根目录下的

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "src", "training", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_player_columns(df: pd.DataFrame) -> List[str]:
    """
    从 DataFrame 中提取所有球员 ID。

    列名格式: "{team}_{jersey}_x" / "{team}_{jersey}_y"
    球员 ID 格式: "{team}_{jersey}"

    Args:
        df: 包含球员列的 DataFrame

    Returns:
        去重的球员 ID 列表，如 ["home_11", "home_1", "away_25", ...]
    """
    player_ids = []
    for col in df.columns:
        # 匹配 xxx_xxx_x 或 xxx_xxx_y 模式 (不包括 ball_x, ball_y)
        if col.endswith("_x") and not col.startswith("ball"):
            # "home_11_x" -> "home_11"
            pid = col[:-2]  # 去掉 "_x"
            player_ids.append(pid)
    return player_ids


def validate_coordinates(df: pd.DataFrame, player_ids: List[str]) -> pd.DataFrame:
    """
    验证坐标是否在合理范围内 [0, 1]。
    Metrica 数据已经归一化，这里做边界检查。

    超出 [0, 1] 的值设为 NaN (表示不在场上)。

    Args:
        df: 数据 DataFrame
        player_ids: 球员 ID 列表

    Returns:
        修正后的 DataFrame
    """
    out_of_range_count = 0
    coord_cols = []

    for pid in player_ids:
        coord_cols.extend([f"{pid}_x", f"{pid}_y"])
    coord_cols.extend(["ball_x", "ball_y"])

    for col in coord_cols:
        if col in df.columns:
            mask = (df[col] < -0.1) | (df[col] > 1.1)
            n_bad = mask.sum()
            if n_bad > 0:
                out_of_range_count += n_bad
                df.loc[mask, col] = np.nan

    if out_of_range_count > 0:
        print(f"  坐标越界修正: {out_of_range_count} 个值")
    else:
        print(f"  坐标范围检查: 全部在 [0, 1] 内 OK")

    return df


def validate_speed(
    df: pd.DataFrame,
    player_ids: List[str],
    max_speed: float,
    pitch_length: float,
    pitch_width: float,
    sample_rate: int,
) -> pd.DataFrame:
    """
    速度异常检测: 如果相邻帧间距离隐含的速度 > max_speed，标记为异常。

    计算原理:
        速度 = 两帧间的位移 / 时间间隔
        位移 = sqrt((Δx * pitch_length)² + (Δy * pitch_width)²)
        时间间隔 = 1 / sample_rate

    异常点的坐标设为 NaN，后续可通过插值修复。

    Args:
        df: 数据 DataFrame
        player_ids: 球员 ID 列表
        max_speed: 最大允许速度 (m/s)
        pitch_length: 场地长度 (米)
        pitch_width: 场地宽度 (米)
        sample_rate: 采样率 (Hz)

    Returns:
        修正后的 DataFrame
    """
    dt = 1.0 / sample_rate
    total_anomalies = 0

    for pid in player_ids:
        x_col = f"{pid}_x"
        y_col = f"{pid}_y"

        if x_col not in df.columns:
            continue

        # 计算相邻帧的位移 (米)
        # diff() 是 pandas 的差分函数: 当前帧减去上一帧
        dx = df[x_col].diff() * pitch_length   # 归一化坐标 → 米
        dy = df[y_col].diff() * pitch_width

        # 计算速度 (m/s)
        speed = np.sqrt(dx**2 + dy**2) / dt

        # 标记异常帧
        anomaly_mask = speed > max_speed
        n_anomalies = anomaly_mask.sum()

        if n_anomalies > 0:
            df.loc[anomaly_mask, x_col] = np.nan
            df.loc[anomaly_mask, y_col] = np.nan
            total_anomalies += n_anomalies

    if total_anomalies > 0:
        print(f"  速度异常修正: {total_anomalies} 个点 (>{max_speed} m/s)")
    else:
        print(f"  速度检查: 无异常 OK")

    return df


def interpolate_missing(
    df: pd.DataFrame,
    player_ids: List[str],
    max_gap: int,
) -> pd.DataFrame:
    """
    对缺失数据 (NaN) 做线性插值修复。

    只修复短暂缺失 (连续 NaN 帧数 ≤ max_gap)，
    长时间缺失保留 NaN (可能是换人/罚下)。

    Args:
        df: 数据 DataFrame
        player_ids: 球员 ID 列表
        max_gap: 最大允许插值的连续缺失帧数

    Returns:
        插值后的 DataFrame
    """
    total_interpolated = 0

    all_cols = []
    for pid in player_ids:
        all_cols.extend([f"{pid}_x", f"{pid}_y"])
    all_cols.extend(["ball_x", "ball_y"])

    for col in all_cols:
        if col not in df.columns:
            continue

        before_nan = df[col].isna().sum()

        # limit 参数限制连续插值的最大帧数
        df[col] = df[col].interpolate(method="linear", limit=max_gap)

        after_nan = df[col].isna().sum()
        total_interpolated += (before_nan - after_nan)

    print(f"  线性插值修复: {total_interpolated} 个缺失值")
    return df


def compute_active_mask(
    df: pd.DataFrame,
    player_ids: List[str],
) -> pd.DataFrame:
    """
    计算每个球员的 is_active 标记。

    如果某帧某球员的 x 和 y 都是 NaN，则 is_active=0 (不在场)。
    否则 is_active=1。

    这处理了红牌罚下、换人等情况。

    Args:
        df: 数据 DataFrame
        player_ids: 球员 ID 列表

    Returns:
        新增 is_active 列后的 DataFrame
    """
    for pid in player_ids:
        x_col = f"{pid}_x"
        y_col = f"{pid}_y"
        # 两个坐标都不是 NaN 才算在场
        # .astype(int) 将 True/False 转为 1/0
        df[f"{pid}_active"] = (~(df[x_col].isna() & df[y_col].isna())).astype(int)

    return df


def generate_summary(df: pd.DataFrame, player_ids: List[str], game_id: str):
    """
    打印数据摘要统计信息。
    """
    print(f"\n--- {game_id} 数据摘要 ---")
    print(f"  总帧数: {len(df)}")
    print(f"  时间跨度: {df['timestamp'].max():.1f} 秒")
    print(f"  球员数: {len(player_ids)}")

    # 统计每个球员的有效帧占比
    home_players = [p for p in player_ids if p.startswith("home")]
    away_players = [p for p in player_ids if p.startswith("away")]

    print(f"  主队球员: {len(home_players)}")
    for pid in home_players:
        active_pct = df[f"{pid}_active"].mean() * 100
        print(f"    {pid}: {active_pct:.1f}% 有效帧")

    print(f"  客队球员: {len(away_players)}")
    for pid in away_players:
        active_pct = df[f"{pid}_active"].mean() * 100
        print(f"    {pid}: {active_pct:.1f}% 有效帧")

    ball_valid = (~df["ball_x"].isna()).mean() * 100
    print(f"  球追踪有效率: {ball_valid:.1f}%")


def preprocess_game(game_dir: str, game_id: str, config: dict) -> pd.DataFrame:
    """
    对单场比赛执行完整预处理流水线。

    Args:
        game_dir: 比赛原始数据目录
        game_id: 比赛标识
        config: 配置字典

    Returns:
        预处理后的 DataFrame
    """
    # Step 1: 解析原始数据
    df = parse_game(game_dir, game_id)

    # 获取球员列表
    player_ids = get_player_columns(df)

    # Step 2: 坐标范围验证
    print("\n[2] 坐标验证...")
    df = validate_coordinates(df, player_ids)

    # Step 3: 速度异常检测
    print("\n[3] 速度异常检测...")
    df = validate_speed(
        df,
        player_ids,
        max_speed=config["validation"]["max_speed"],
        pitch_length=config["pitch"]["length"],
        pitch_width=config["pitch"]["width"],
        sample_rate=config["window"]["sample_rate"],
    )

    # Step 4: 缺失值插值
    print("\n[4] 缺失值插值...")
    max_gap_frames = int(
        config["validation"]["max_gap_seconds"] * config["window"]["sample_rate"]
    )
    df = interpolate_missing(df, player_ids, max_gap=max_gap_frames)

    # Step 5: 生成 is_active 掩码
    print("\n[5] 生成 is_active 掩码...")
    df = compute_active_mask(df, player_ids)

    # 打印摘要
    generate_summary(df, player_ids, game_id)

    return df


def main():
    """
    主入口: 预处理所有比赛数据并保存为 Parquet。
    """
    config = load_config()

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw", "metrica")
    output_dir = os.path.join(PROJECT_ROOT, "data", "processed")

    # 自动发现所有比赛目录
    game_dirs = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    print(f"发现 {len(game_dirs)} 场比赛: {game_dirs}")

    for game_id in game_dirs:
        game_dir = os.path.join(raw_dir, game_id)

        # 执行预处理
        df = preprocess_game(game_dir, game_id, config)

        # 保存为 Parquet
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{game_id}.parquet")
        df.to_parquet(output_path, index=False)
        print(f"\n>> 已保存: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

    print(f"\n{'='*50}")
    print(f"全部预处理完成! 输出目录: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
