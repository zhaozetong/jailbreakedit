import time
import torch
from tqdm import tqdm

def get_gpu_memory_usage():
    """获取 GPU 显存使用情况"""
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # 转换为 GB
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 获取总显存大小（GB）
    return allocated_memory, total_memory

# 模拟持续监控 GPU 显存使用情况
monitor_duration = 10  # 监控 10 秒
refresh_interval = 0.5  # 每隔 0.5 秒刷新一次

# 创建进度条
with tqdm(total=monitor_duration, desc="GPU Memory Monitoring", ncols=100) as pbar:
    start_time = time.time()
    while True:
        # 获取显存使用情况
        allocated_memory, total_memory = get_gpu_memory_usage()
        used_ratio = allocated_memory / total_memory * 100  # 使用比例

        # 更新进度条描述
        pbar.set_description(f"GPU Memory: {allocated_memory:.2f}/{total_memory:.2f} GB ({used_ratio:.2f}%)")

        # 计算已经运行的时间
        elapsed_time = time.time() - start_time
        if elapsed_time >= monitor_duration:
            break

        # 更新进度条
        pbar.update(refresh_interval)  # 模拟时间增加
        time.sleep(refresh_interval)