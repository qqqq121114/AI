import platform
import psutil
import torch
import cpuinfo
import os
import sys
import json
from datetime import datetime

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_system_info():
    info = {}
    
    # 系统信息
    info["系统信息"] = {
        "操作系统": platform.system(),
        "操作系统版本": platform.version(),
        "操作系统架构": platform.machine(),
        "处理器": platform.processor(),
        "计算机名": platform.node(),
        "Python版本": sys.version
    }
    
    # CPU信息
    cpu_info = cpuinfo.get_cpu_info()
    info["CPU信息"] = {
        "物理核心数": psutil.cpu_count(logical=False),
        "逻辑核心数": psutil.cpu_count(logical=True),
        "CPU型号": cpu_info['brand_raw'],
        "基础频率": f"{cpu_info.get('hz_advertised_friendly', 'Unknown')}",
        "当前使用率": f"{psutil.cpu_percent()}%"
    }
    
    # 内存信息
    svmem = psutil.virtual_memory()
    info["内存信息"] = {
        "总物理内存": get_size(svmem.total),
        "可用内存": get_size(svmem.available),
        "内存使用率": f"{svmem.percent}%"
    }
    
    # GPU信息
    if torch.cuda.is_available():
        info["GPU信息"] = {
            "GPU数量": torch.cuda.device_count()
        }
        for i in range(torch.cuda.device_count()):
            gpu_properties = torch.cuda.get_device_properties(i)
            info["GPU信息"][f"GPU {i}"] = {
                "名称": gpu_properties.name,
                "总显存": f"{gpu_properties.total_memory / 1024**2:.2f} MB",
                "计算能力": f"{gpu_properties.major}.{gpu_properties.minor}"
            }
    else:
        info["GPU信息"] = "未检测到GPU"
    
    # 磁盘信息
    disk_info = {}
    for partition in psutil.disk_partitions():
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            disk_info[partition.device] = {
                "挂载点": partition.mountpoint,
                "文件系统": partition.fstype,
                "总大小": get_size(partition_usage.total),
                "已用": get_size(partition_usage.used),
                "可用": get_size(partition_usage.free),
                "使用率": f"{partition_usage.percent}%"
            }
        except Exception:
            continue
    info["磁盘信息"] = disk_info
    
    return info

def main():
    try:
        # 获取系统信息
        system_info = get_system_info()
        
        # 创建输出文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"system_info_{timestamp}.json"
        
        # 将信息保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(system_info, f, ensure_ascii=False, indent=4)
        
        print(f"系统信息已保存到文件: {output_file}")
        
        # 同时在控制台显示信息
        print("\n系统信息摘要:")
        print("="*50)
        for category, details in system_info.items():
            print(f"\n{category}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {details}")
        
    except Exception as e:
        print(f"获取系统信息时发生错误: {str(e)}")

if __name__ == "__main__":
    main() 