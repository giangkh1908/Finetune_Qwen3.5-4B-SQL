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

# Tự động nhận diện hoặc cài đặt pip nếu môi trường container chưa có sẵn
if command -v pip &> /dev/null; then
    PIP_CMD="pip"
elif command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif python3 -m pip --version &> /dev/null; then
    PIP_CMD="python3 -m pip"
else
    echo "⚠️ Không tìm thấy pip. Đang tự động cài python3-pip qua apt..."
    apt-get update && apt-get install -y python3-pip python3-dev
    PIP_CMD="pip3"
fi

if command -v python &> /dev/null; then
    PY_CMD="python"
else
    PY_CMD="python3"
fi

echo "Sử dụng pip: $PIP_CMD | Python: $PY_CMD"

$PIP_CMD install --upgrade pip

# Kiểm tra xem PyTorch hiện tại có nhận CUDA không
echo "🔍 Kiểm tra tương thích CUDA của PyTorch..."
CUDA_OK=$($PY_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$CUDA_OK" != "True" ]; then
    echo "⚠️ PyTorch hiện tại không nhận GPU hoặc lệch phiên bản CUDA driver."
    echo "🔄 Đang cài đặt PyTorch tương thích CUDA 12.4 (hỗ trợ Driver 570+)..."
    $PIP_CMD install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

$PIP_CMD install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
$PIP_CMD install unsloth_zoo
$PIP_CMD install --no-deps trl peft accelerate bitsandbytes
$PIP_CMD install datasets pyyaml huggingface_hub

# Gỡ bỏ torchao xung đột phiên bản (Unsloth dùng bitsandbytes, không dùng torchao)
$PIP_CMD uninstall -y torchao 2>/dev/null || true

echo ""
echo "===================================================================="
echo "🖥️  [2/3] Kiểm tra thông tin GPU..."
echo "===================================================================="
nvidia-smi
$PY_CMD -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "===================================================================="
echo "🔥 [3/3] Chạy script huấn luyện..."
echo "===================================================================="
$PY_CMD train_cloud_gpu.py "$@"
