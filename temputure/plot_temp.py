#!/home/tl/ssd/docker_link/python/hex_device_test/.venv/bin/python3
"""绘制电机温度曲线，并使用 numpy.polyfit 进行多项式拟合

用法:
    python3 plot_temp.py <csv文件路径>
    python3 plot_temp.py temp_dev0_2026-06-18_22-46-14.csv --degree 5
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

parser = argparse.ArgumentParser(description="绘制电机温度曲线")
parser.add_argument("csv_file", nargs="?", default="temp_dev0_2026-06-18_22-46-14.csv",
                    help="CSV 文件路径 (默认: temp_dev0_2026-06-18_22-46-14.csv)")
parser.add_argument("--degree", "-d", type=int, default=5,
                    help="多项式拟合阶数 (默认: 5)")
args = parser.parse_args()

# CSV 文件路径（支持相对路径和绝对路径）
csv_path = Path(args.csv_file)
if not csv_path.is_absolute():
    csv_path = Path(__file__).parent / csv_path

# 读取数据
df = pd.read_csv(csv_path, parse_dates=["timestamp"])

motor_columns = ["motor_0", "motor_1", "motor_2", "motor_3", "motor_4", "motor_5"]
print("各电机最高温度:")
for col in motor_columns:
    max_val = df[col].max()
    print(f"  {col}: {max_val:.2f} °C")

# 设置中文字体（如果系统支持），否则用英文
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 创建图表
fig, ax = plt.subplots(figsize=(14, 7))

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# 将时间转为 matplotlib 可识别的数值格式
timestamps = df["timestamp"].values

# 将时间戳转为秒数（以第一个时间点为 0），用于 polyfit 的 x 坐标
t_seconds = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().values

for col, color in zip(motor_columns, colors):
    temps = df[col].values

    # 绘制原始曲线（细线）
    ax.plot(timestamps, temps, label=col, color=color, linewidth=1.2, alpha=0.85)

    # numpy.polyfit 多项式拟合，绘制拟合曲线（虚线）
    coeffs = np.polyfit(t_seconds, temps, args.degree)
    fitted = np.polyval(coeffs, t_seconds)
    ax.plot(timestamps, fitted, color=color, linewidth=2.0, linestyle="--", alpha=0.7)

# 最高温度上限红线
ax.axhline(y=110, color="red", linewidth=1.5, linestyle="-", alpha=0.8, label="Max limit 110°C")

# 设置标题和标签
ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"Motor Temperature Curves (polyfit degree={args.degree})\n{csv_path.name}")

# 格式化 x 轴时间显示
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate(rotation=30)

# 图例
ax.legend(loc="upper left", fontsize=9)

# 网格
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

# 保存图片
output_path = csv_path.with_suffix(".png")
plt.savefig(output_path, dpi=150)
print(f"图片已保存: {output_path}")

# 显示
plt.show()
