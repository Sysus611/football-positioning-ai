"""
Metrica Sports 原始数据解析器

支持两种格式:
1. CSV 格式 (Game 1 & 2): Home/Away 分开的 CSV 文件
2. EPTS FIFA 格式 (Game 3): tracking.txt + metadata.xml

所有解析器输出统一的 DataFrame 格式:
- 列: frame, period, timestamp, {player_id}_x, {player_id}_y, ball_x, ball_y
- 坐标: 归一化 [0, 1]
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
from typing import Tuple, Dict, List


def parse_metrica_csv(home_csv_path: str, away_csv_path: str) -> pd.DataFrame:
    """
    解析 Metrica CSV 格式的追踪数据 (Game 1 & 2)。

    CSV 文件结构:
        - 第1行: 队伍标识 (Home/Away)
        - 第2行: 球员球衣号码
        - 第3行: 列名 (Period, Frame, Time [s], PlayerX, ..., Ball)
        - 第4行起: 数据，每个球员占2列 (x, y)

    Args:
        home_csv_path: 主队追踪数据 CSV 路径
        away_csv_path: 客队追踪数据 CSV 路径

    Returns:
        统一格式的 DataFrame
    """
    # ---- 解析单个 CSV ----
    def _read_single_csv(path: str, team_prefix: str) -> pd.DataFrame:
        """
        读取单个队伍的 CSV 文件。

        这里用了 pandas 的 read_csv 函数。几个参数说明:
        - header=None: 不自动解析表头 (因为有3行复合表头)
        - skiprows=3: 跳过前3行表头，直接读数据行
        """
        # 先读表头来获取球员 ID
        header_df = pd.read_csv(path, header=None, nrows=3)

        # 第3行 (index=2) 是列名: Period, Frame, Time [s], Player11, , Player1, ...
        col_names = header_df.iloc[2].values

        # 提取球员 ID: 列名格式是 "PlayerXX"，每个球员占2列 (x, y)
        player_ids = []
        for col_name in col_names:
            if isinstance(col_name, str) and col_name.startswith("Player"):
                # 提取号码: "Player11" -> "11"
                pid = col_name.replace("Player", "")
                if pid not in player_ids:
                    player_ids.append(pid)

        # 读取数据行 (跳过3行表头)
        data = pd.read_csv(path, header=None, skiprows=3)

        # 构建结果 DataFrame
        result = pd.DataFrame()
        result["period"] = data.iloc[:, 0].astype(int)
        result["frame"] = data.iloc[:, 1].astype(int)
        result["timestamp"] = data.iloc[:, 2].astype(float)

        # 每个球员占2列 (x, y)，从第4列 (index=3) 开始
        col_idx = 3
        for pid in player_ids:
            result[f"{team_prefix}_{pid}_x"] = data.iloc[:, col_idx].astype(float)
            result[f"{team_prefix}_{pid}_y"] = data.iloc[:, col_idx + 1].astype(float)
            col_idx += 2

        # 球的位置在最后2列之前 (倒数第3和倒数第2列，最后一列是空的)
        result["ball_x"] = data.iloc[:, -2].astype(float)
        result["ball_y"] = data.iloc[:, -1].astype(float)

        return result, player_ids

    # 解析主客队数据
    home_df, home_players = _read_single_csv(home_csv_path, "home")
    away_df, away_players = _read_single_csv(away_csv_path, "away")

    # 合并: 以 frame 为主键，拼接主客队球员列
    # 只保留一份 period/frame/timestamp/ball 列
    away_player_cols = [c for c in away_df.columns if c.startswith("away_")]
    merged = home_df.copy()
    for col in away_player_cols:
        merged[col] = away_df[col].values

    # 覆盖球的位置（两个文件应该一致，取一个即可）
    print(f"  主队球员: {home_players}")
    print(f"  客队球员: {away_players}")
    print(f"  总帧数: {len(merged)}")

    return merged


def parse_metrica_epts(tracking_path: str, metadata_path: str) -> pd.DataFrame:
    """
    解析 Metrica EPTS FIFA 格式的追踪数据 (Game 3)。

    tracking.txt 格式 (每行一帧):
        帧号:x1,y1;x2,y2;...;x22,y22:ball_x,ball_y

    metadata.xml 包含:
        - 帧率 (25Hz)
        - 上下半场帧范围
        - 球员 ID 和球队信息

    Args:
        tracking_path: tracking.txt 路径
        metadata_path: metadata.xml 路径

    Returns:
        统一格式的 DataFrame
    """
    # ---- 解析 metadata.xml ----
    tree = ET.parse(metadata_path)
    root = tree.getroot()

    # XML 命名空间处理 (Metrica 用了带命名空间的 XML)
    # 直接去掉命名空间前缀来简化解析
    ns = ""
    for elem in root.iter():
        if "}" in elem.tag:
            ns = elem.tag.split("}")[0] + "}"
            break

    # 提取帧率
    frame_rate_elem = root.find(f".//{ns}FrameRate")
    frame_rate = int(frame_rate_elem.text) if frame_rate_elem is not None else 25

    # 提取上下半场帧范围
    params = {}
    for param in root.iter(f"{ns}ProviderParameter"):
        name_elem = param.find(f"{ns}Name")
        value_elem = param.find(f"{ns}Value")
        if name_elem is not None and value_elem is not None and value_elem.text:
            params[name_elem.text] = value_elem.text

    half1_start = int(params.get("first_half_start", 1))
    half1_end = int(params.get("first_half_end", 0))
    half2_start = int(params.get("second_half_start", 0))
    half2_end = int(params.get("second_half_end", 0))

    # 提取球员信息
    # XML 结构: <Player id="P3578" teamId="FIFATMA">
    #             <Name>Player 11</Name>
    #             <ShirtNumber>11</ShirtNumber>
    #             ...
    #           </Player>
    # 注意: id 和 teamId 是 XML 属性 (attribute)，不是子元素
    player_info = {}     # pid -> (team_prefix, jersey)
    player_order = []    # 保持顺序，对应 tracking 数据中的坐标顺序

    for player in root.iter(f"{ns}Player"):
        # elem.attrib 是一个字典，存储 XML 元素的属性
        # 例如 <Player id="P3578" teamId="FIFATMA"> 的 attrib = {"id": "P3578", "teamId": "FIFATMA"}
        pid = player.attrib.get("id")
        team_id = player.attrib.get("teamId", "unknown")

        # 球衣号码在子元素 <ShirtNumber> 里
        jersey_elem = player.find(f"{ns}ShirtNumber")
        jersey = jersey_elem.text.strip() if jersey_elem is not None else pid

        if pid:
            # FIFATMA = Team A (home), FIFATMB = Team B (away)
            team_prefix = "home" if team_id.upper() in ("HOME", "FIFATMA") else "away"
            player_info[pid] = (team_prefix, jersey)
            player_order.append(pid)

    print(f"  帧率: {frame_rate}Hz")
    print(f"  上半场: 帧 {half1_start}-{half1_end}")
    print(f"  下半场: 帧 {half2_start}-{half2_end}")
    print(f"  球员数: {len(player_order)}")

    # ---- 解析 tracking.txt ----
    frames = []
    periods = []
    timestamps = []
    player_positions = {pid: ([], []) for pid in player_order}
    ball_x_list = []
    ball_y_list = []

    with open(tracking_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 格式: 帧号:p1x,p1y;p2x,p2y;...;pNx,pNy:ballx,bally
            parts = line.split(":")

            frame_id = int(parts[0])

            # 判断上下半场
            if half1_start <= frame_id <= half1_end:
                period = 1
                elapsed = (frame_id - half1_start) / frame_rate
            elif half2_start <= frame_id <= half2_end:
                period = 2
                elapsed = (frame_id - half2_start) / frame_rate
            else:
                continue  # 跳过非比赛帧

            frames.append(frame_id)
            periods.append(period)
            timestamps.append(round(elapsed, 4))

            # 解析球员位置
            player_data = parts[1].split(";")
            for i, pid in enumerate(player_order):
                if i < len(player_data):
                    coords = player_data[i].split(",")
                    try:
                        px, py = float(coords[0]), float(coords[1])
                    except (ValueError, IndexError):
                        px, py = np.nan, np.nan
                else:
                    px, py = np.nan, np.nan
                player_positions[pid][0].append(px)
                player_positions[pid][1].append(py)

            # 解析球位置
            if len(parts) > 2:
                ball_coords = parts[2].split(",")
                try:
                    bx, by = float(ball_coords[0]), float(ball_coords[1])
                except (ValueError, IndexError):
                    bx, by = np.nan, np.nan
            else:
                bx, by = np.nan, np.nan
            ball_x_list.append(bx)
            ball_y_list.append(by)

    # 构建 DataFrame
    result = pd.DataFrame({
        "frame": frames,
        "period": periods,
        "timestamp": timestamps,
    })

    for pid in player_order:
        team_prefix, jersey = player_info[pid]
        result[f"{team_prefix}_{jersey}_x"] = player_positions[pid][0]
        result[f"{team_prefix}_{jersey}_y"] = player_positions[pid][1]

    result["ball_x"] = ball_x_list
    result["ball_y"] = ball_y_list

    print(f"  总帧数: {len(result)}")
    return result


def parse_game(game_dir: str, game_id: str) -> pd.DataFrame:
    """
    自动检测数据格式并解析。

    Args:
        game_dir: 比赛数据目录 (如 data/raw/metrica/game1)
        game_id: 比赛标识 (如 "game1")

    Returns:
        统一格式的 DataFrame
    """
    print(f"\n{'='*50}")
    print(f"解析 {game_id}: {game_dir}")
    print(f"{'='*50}")

    # 检查是 CSV 格式还是 EPTS 格式
    home_csv = os.path.join(game_dir, "tracking_home.csv")
    away_csv = os.path.join(game_dir, "tracking_away.csv")
    tracking_txt = os.path.join(game_dir, "tracking.txt")
    metadata_xml = os.path.join(game_dir, "metadata.xml")

    if os.path.exists(home_csv) and os.path.exists(away_csv):
        print("  格式: Metrica CSV")
        df = parse_metrica_csv(home_csv, away_csv)
    elif os.path.exists(tracking_txt) and os.path.exists(metadata_xml):
        print("  格式: EPTS FIFA")
        df = parse_metrica_epts(tracking_txt, metadata_xml)
    else:
        raise FileNotFoundError(f"无法在 {game_dir} 找到有效的追踪数据文件")

    # 添加 game_id 列
    df.insert(0, "game_id", game_id)

    return df
