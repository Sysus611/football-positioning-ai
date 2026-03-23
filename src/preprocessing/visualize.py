"""
比赛追踪数据可视化 - 生成模拟视频

从 Parquet 预处理数据读取球员位置，在虚拟球场上绘制球员移动轨迹，
输出 240p MP4 视频。

使用 OpenCV (cv2) 绘制每一帧，效率远高于 matplotlib。

使用方法:
    python src/preprocessing/visualize.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========== 视频参数 ==========
VIDEO_W = 426           # 240p 宽度 (16:9 近似)
VIDEO_H = 240           # 240p 高度
VIDEO_FPS = 30          # 输出帧率
SPEED_MULTIPLIER = 15   # 加速倍率: 15x (90分钟 -> ~6分钟视频)

# 球场绘制区域 (留边距)
MARGIN = 20
PITCH_X1 = MARGIN
PITCH_Y1 = MARGIN + 15   # 上方留空给标题
PITCH_X2 = VIDEO_W - MARGIN
PITCH_Y2 = VIDEO_H - MARGIN

# 颜色定义 (BGR 格式，OpenCV 用 BGR 不是 RGB)
COLOR_BG = (34, 40, 49)         # 深灰背景
COLOR_PITCH = (50, 120, 50)     # 草地绿
COLOR_LINES = (200, 200, 200)   # 白色线条
COLOR_HOME = (60, 60, 230)      # 红色 (主队)
COLOR_AWAY = (230, 180, 50)     # 蓝色 (客队)
COLOR_BALL = (0, 255, 255)      # 黄色 (球)
COLOR_TEXT = (220, 220, 220)    # 浅灰文字
COLOR_INACTIVE = (100, 100, 100)  # 灰色 (不在场)


def draw_pitch(frame: np.ndarray):
    """
    在帧上绘制足球场地。

    cv2 绘图函数说明:
    - cv2.rectangle(img, pt1, pt2, color, thickness)
      绘制矩形. pt1=左上角, pt2=右下角, thickness=-1表示填充
    - cv2.line(img, pt1, pt2, color, thickness)
      绘制直线
    - cv2.circle(img, center, radius, color, thickness)
      绘制圆. thickness=-1表示填充
    - cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
      绘制椭圆弧

    Args:
        frame: numpy 数组，代表一帧图像 (H, W, 3)
    """
    pw = PITCH_X2 - PITCH_X1   # 球场像素宽度
    ph = PITCH_Y2 - PITCH_Y1   # 球场像素高度

    # 填充草地
    cv2.rectangle(frame, (PITCH_X1, PITCH_Y1), (PITCH_X2, PITCH_Y2), COLOR_PITCH, -1)

    # 外边框
    cv2.rectangle(frame, (PITCH_X1, PITCH_Y1), (PITCH_X2, PITCH_Y2), COLOR_LINES, 1)

    # 中线
    mid_x = PITCH_X1 + pw // 2
    cv2.line(frame, (mid_x, PITCH_Y1), (mid_x, PITCH_Y2), COLOR_LINES, 1)

    # 中圈 (半径约 9.15m，场地105m，比例 = 9.15/105)
    center_radius = int(pw * 9.15 / 105)
    cv2.circle(frame, (mid_x, PITCH_Y1 + ph // 2), center_radius, COLOR_LINES, 1)

    # 中点
    cv2.circle(frame, (mid_x, PITCH_Y1 + ph // 2), 2, COLOR_LINES, -1)

    # 禁区 (长 16.5m, 宽 40.3m)
    pen_w = int(pw * 16.5 / 105)
    pen_h = int(ph * 40.3 / 68)
    pen_y1 = PITCH_Y1 + (ph - pen_h) // 2
    pen_y2 = pen_y1 + pen_h

    # 左禁区
    cv2.rectangle(frame, (PITCH_X1, pen_y1), (PITCH_X1 + pen_w, pen_y2), COLOR_LINES, 1)
    # 右禁区
    cv2.rectangle(frame, (PITCH_X2 - pen_w, pen_y1), (PITCH_X2, pen_y2), COLOR_LINES, 1)

    # 小禁区 (长 5.5m, 宽 18.3m)
    spen_w = int(pw * 5.5 / 105)
    spen_h = int(ph * 18.3 / 68)
    spen_y1 = PITCH_Y1 + (ph - spen_h) // 2
    spen_y2 = spen_y1 + spen_h

    cv2.rectangle(frame, (PITCH_X1, spen_y1), (PITCH_X1 + spen_w, spen_y2), COLOR_LINES, 1)
    cv2.rectangle(frame, (PITCH_X2 - spen_w, spen_y1), (PITCH_X2, spen_y2), COLOR_LINES, 1)


def norm_to_pixel(x_norm: float, y_norm: float) -> tuple:
    """
    将归一化坐标 [0,1] 转换为像素坐标。

    Metrica 坐标系: (0,0)=左上, (1,1)=右下
    像素坐标: (PITCH_X1, PITCH_Y1) 到 (PITCH_X2, PITCH_Y2)

    Args:
        x_norm: 归一化 x 坐标 [0, 1]
        y_norm: 归一化 y 坐标 [0, 1]

    Returns:
        (px, py) 像素坐标
    """
    pw = PITCH_X2 - PITCH_X1
    ph = PITCH_Y2 - PITCH_Y1
    px = int(PITCH_X1 + x_norm * pw)
    py = int(PITCH_Y1 + y_norm * ph)
    return (px, py)


def generate_video(parquet_path: str, output_path: str, game_id: str):
    """
    从 Parquet 追踪数据生成模拟视频。

    Args:
        parquet_path: 预处理后的 Parquet 文件路径
        output_path: 输出 MP4 文件路径
        game_id: 比赛标识 (用于显示标题)
    """
    print(f"\n{'='*50}")
    print(f"生成视频: {game_id}")
    print(f"{'='*50}")

    # 读取数据
    df = pd.read_parquet(parquet_path)
    total_frames = len(df)

    # 获取球员列表
    player_cols = {}   # pid -> (x_col, y_col, active_col, team)
    for col in df.columns:
        if col.endswith("_x") and not col.startswith("ball") and not col.startswith("game"):
            pid = col[:-2]
            x_col = f"{pid}_x"
            y_col = f"{pid}_y"
            active_col = f"{pid}_active"
            team = "home" if pid.startswith("home") else "away"
            if x_col in df.columns and y_col in df.columns:
                player_cols[pid] = (x_col, y_col, active_col, team)

    print(f"  球员数: {len(player_cols)}")
    print(f"  原始帧数: {total_frames}")

    # 计算抽帧间隔
    # 原始 25Hz, 输出 30fps, 加速 SPEED_MULTIPLIER 倍
    # 每帧对应原始数据的步长 = 25 * SPEED_MULTIPLIER / 30
    frame_step = max(1, int(25 * SPEED_MULTIPLIER / VIDEO_FPS))
    output_frame_count = total_frames // frame_step
    output_duration = output_frame_count / VIDEO_FPS

    print(f"  抽帧步长: 每 {frame_step} 帧取 1 帧")
    print(f"  输出帧数: {output_frame_count}")
    print(f"  输出时长: {output_duration:.0f} 秒 ({output_duration/60:.1f} 分钟)")

    # 创建视频写入器
    # cv2.VideoWriter_fourcc 指定编码格式
    # 'mp4v' 是 MP4 常用编码, 兼容性好
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H))

    if not writer.isOpened():
        print(f"  ERROR: 无法创建视频文件 {output_path}")
        return

    # 预先提取需要的列为 numpy 数组 (比逐行读 DataFrame 快很多)
    ball_x = df["ball_x"].values
    ball_y = df["ball_y"].values
    periods = df["period"].values
    timestamps = df["timestamp"].values

    player_arrays = {}
    for pid, (x_col, y_col, active_col, team) in player_cols.items():
        px = df[x_col].values
        py = df[y_col].values
        pa = df[active_col].values if active_col in df.columns else np.ones(len(df))
        player_arrays[pid] = (px, py, pa, team)

    # 逐帧绘制
    written = 0
    for idx in range(0, total_frames, frame_step):
        # 创建空白帧
        frame = np.full((VIDEO_H, VIDEO_W, 3), COLOR_BG, dtype=np.uint8)

        # 绘制球场
        draw_pitch(frame)

        # 绘制标题信息
        period = periods[idx]
        ts = timestamps[idx]
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        title = f"{game_id.upper()} | P{period} {minutes:02d}:{seconds:02d} | {SPEED_MULTIPLIER}x"
        cv2.putText(frame, title, (PITCH_X1, PITCH_Y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1)

        # 绘制球员
        for pid, (px, py, pa, team) in player_arrays.items():
            x, y, active = px[idx], py[idx], pa[idx]

            # 跳过 NaN 和不在场的球员
            if np.isnan(x) or np.isnan(y) or active == 0:
                continue

            pos = norm_to_pixel(x, y)
            color = COLOR_HOME if team == "home" else COLOR_AWAY
            # 绘制球员圆点 (半径 4 像素)
            cv2.circle(frame, pos, 4, color, -1)
            # 绘制边框让圆点更清晰
            cv2.circle(frame, pos, 4, (0, 0, 0), 1)

        # 绘制球
        bx, by = ball_x[idx], ball_y[idx]
        if not np.isnan(bx) and not np.isnan(by):
            ball_pos = norm_to_pixel(bx, by)
            cv2.circle(frame, ball_pos, 3, COLOR_BALL, -1)
            cv2.circle(frame, ball_pos, 3, (0, 0, 0), 1)

        # 绘制图例
        legend_y = VIDEO_H - 8
        cv2.circle(frame, (PITCH_X1, legend_y), 4, COLOR_HOME, -1)
        cv2.putText(frame, "Home", (PITCH_X1 + 8, legend_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1)
        cv2.circle(frame, (PITCH_X1 + 55, legend_y), 4, COLOR_AWAY, -1)
        cv2.putText(frame, "Away", (PITCH_X1 + 63, legend_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1)
        cv2.circle(frame, (PITCH_X1 + 110, legend_y), 3, COLOR_BALL, -1)
        cv2.putText(frame, "Ball", (PITCH_X1 + 117, legend_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1)

        writer.write(frame)
        written += 1

        # 进度报告
        if written % 500 == 0:
            pct = written / output_frame_count * 100
            print(f"  进度: {pct:.0f}% ({written}/{output_frame_count})")

    writer.release()
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  >> 完成: {output_path} ({size_mb:.1f} MB, {written} 帧)")


def main():
    """生成所有比赛的模拟视频"""
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    output_dir = os.path.join(PROJECT_ROOT, "data", "videos")
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有 parquet 文件
    parquet_files = sorted([
        f for f in os.listdir(processed_dir)
        if f.endswith(".parquet")
    ])

    print(f"找到 {len(parquet_files)} 个预处理文件")

    for pf in parquet_files:
        game_id = pf.replace(".parquet", "")
        parquet_path = os.path.join(processed_dir, pf)
        output_path = os.path.join(output_dir, f"{game_id}.mp4")
        generate_video(parquet_path, output_path, game_id)

    print(f"\n{'='*50}")
    print(f"全部视频生成完成! 输出目录: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
