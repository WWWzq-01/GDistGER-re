#!/bin/bash

CPU_OUTPUT_FILE="cpu_usage.log"
GPU_OUTPUT_FILE="gpu_usage.log"

# 清空或者创建输出文件
> "$CPU_OUTPUT_FILE"
> "$GPU_OUTPUT_FILE"


while true; do 
    mpstat 1 1 | awk '/Average:/ {print 100 - $12}' >> "$CPU_OUTPUT_FILE"
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 >> "$GPU_OUTPUT_FILE"
    sleep 1
done
