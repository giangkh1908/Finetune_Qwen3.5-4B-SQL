#!/bin/bash
# ====================================================================
# 🚀 1-CLICK CLOUD GPU SETUP & TRAINING RUNNER
# Dành cho RunPod / Vast.ai / Lambda Labs / Lambda GPU Cloud
# ====================================================================

set -e

echo "===================================================================="
echo "⚡ [1/3] Cài đặt môi trường Unsloth tối ưu tốc độ..."
echo "===================================================================="
export UNSLOTH_SKIP_TORCHVISION_CHECK=1

pip install --upgrade pip
pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install unsloth_zoo
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets pyyaml huggingface_hub

echo ""
echo "===================================================================="
echo "🖥️  [2/3] Kiểm tra thông tin GPU..."
echo "===================================================================="
nvidia-smi

echo ""
echo "===================================================================="
echo "🔥 [3/3] Chạy script huấn luyện..."
echo "===================================================================="
python train_cloud_gpu.py "$@"
