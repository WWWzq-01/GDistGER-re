#!/bin/bash

# 定义 hosts 文件路径
HOSTS_FILE="./hosts"

# 打印表头
printf "%-15s %-10s %-10s %-10s %-10s\n" "IP Address" "CPU Util" "Mem Util" "GPU Util" "GPU Mem Util"

# 从 hosts 文件中逐行读取 IP 地址
while  read  -r ip; do
    # 检查 IP 地址是否为空,或则是 # 开头的注释
    if ! [[ -n "$ip" && "$ip" != \#* ]]; then
        continue;
    fi
    # 获取 CPU 利用率
    cpu_util=$(ssh "$ip" "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d, -f1 | cut -d'%' -f1" < /dev/null)

    # 获取内存使用率（单位为百分比）
    mem_util=$(ssh "$ip" "free -m | awk 'NR==2{printf \"%.2f\", \$3/\$2 * 100}'" < /dev/null)

    # 获取 GPU 利用率和 GPU 显存利用率
    gpu_util=$(ssh "$ip" "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" < /dev/null)
    gpu_mem_util=$(ssh "$ip" "nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits" < /dev/null)

    # 如果没有 GPU，设置为 "N/A"
    if [[ -z "$gpu"_util ]]; then
        gpu_util="N/A"
        gpu_mem_util="N/A"
    fi

    # 打印结果
    printf "%-15s %-10s %-10s %-10s %-10s\n" "$ip" "$cpu_util%" "$mem_util%" "$gpu_util%" "$gpu_mem_util%"
done < hosts
