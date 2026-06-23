#!/bin/bash
# ============================================================
# Baidu CTI 2026 - Generative Recommendation Inference Optim.
# Environment Build Script
# Must complete within 20 minutes
# ============================================================
set -e

echo "==== [Build Env] Installing Python dependencies ===="
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# Enable Paddle custom operator compilation if present
if [ -d "custom_ops" ]; then
    echo "==== [Build Env] Compiling custom CUDA operators ===="
    cd custom_ops
    python setup.py install 2>&1 | tail -5
    cd ..
    echo "==== [Build Env] Custom operators compiled ===="
fi

echo "==== [Build Env] Environment ready ===="
