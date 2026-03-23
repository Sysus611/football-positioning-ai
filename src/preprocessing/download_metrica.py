"""
下载 Metrica Sports 开源追踪数据
数据源: https://github.com/metrica-sports/sample-data

包含3场匿名比赛的追踪数据与事件数据:
- Game 1 & 2: CSV 格式 (标准Metrica格式)
- Game 3: EPTS FIFA 格式 + JSON 事件

追踪数据: 25Hz, 坐标归一化 [0,1], 场地 105x68m
"""

import os
import urllib.request
import sys

# Metrica Sports GitHub raw URL base
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"

# 要下载的文件列表
FILES = {
    # Game 1 - CSV 格式
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": "game1/tracking_away.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": "game1/tracking_home.csv",
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv": "game1/events.csv",

    # Game 2 - CSV 格式
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": "game2/tracking_away.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": "game2/tracking_home.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv": "game2/events.csv",

    # Game 3 - EPTS FIFA 格式 (tracking) + JSON (events)
    "Sample_Game_3/Sample_Game_3_tracking.txt": "game3/tracking.txt",
    "Sample_Game_3/Sample_Game_3_metadata.xml": "game3/metadata.xml",
    "Sample_Game_3/Sample_Game_3_events.json": "game3/events.json",
}


def download_file(url: str, dest_path: str) -> bool:
    """
    下载单个文件到指定路径。

    Args:
        url: 下载链接
        dest_path: 目标文件路径

    Returns:
        bool: 是否下载成功
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"  [跳过] 已存在: {dest_path}")
        return True

    try:
        print(f"  [下载中] {os.path.basename(dest_path)}...", end="", flush=True)
        urllib.request.urlretrieve(url, dest_path)
        # 获取文件大小 (MB)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f" OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f" ✗ 错误: {e}")
        return False


def main():
    """下载所有 Metrica Sports 样本数据"""
    # 数据目标目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data", "raw", "metrica")

    print("=" * 60)
    print("Metrica Sports 数据下载器")
    print(f"目标目录: {data_dir}")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for remote_path, local_path in FILES.items():
        url = f"{BASE_URL}/{remote_path}"
        dest = os.path.join(data_dir, local_path)
        if download_file(url, dest):
            success_count += 1
        else:
            fail_count += 1

    print("=" * 60)
    print(f"完成! 成功: {success_count}, 失败: {fail_count}")

    if fail_count > 0:
        print("\n[提示] 如果下载失败，请检查网络连接或使用代理。")
        print("也可以手动从 https://github.com/metrica-sports/sample-data 下载。")
        sys.exit(1)


if __name__ == "__main__":
    main()
