#!/bin/bash
# ====================================================================
# 🚀 1-CLICK ALL-IN-ONE CLOUD GPU SETUP & TRAINING RUNNER
# Tự động hóa 100%: Môi trường + Fix Driver CUDA + Fix Ổ đĩa + Train + Push Hub
# Dành cho Vast.ai / RunPod / Lambda Labs
# ====================================================================

set -e

echo "===================================================================="
echo "⚡ [1/4] Cài đặt và cấu hình môi trường Unsloth..."
echo "===================================================================="
export UNSLOTH_SKIP_TORCHVISION_CHECK=1

# 1. Nhận diện pip và python
if command -v pip &> /dev/null; then
    PIP_CMD="pip"
elif command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif python3 -m pip --version &> /dev/null; then
    PIP_CMD="python3 -m pip"
else
    echo "⚠️ Đang cài đặt python3-pip..."
    apt-get update && apt-get install -y python3-pip python3-dev
    PIP_CMD="pip3"
fi

if command -v python &> /dev/null; then
    PY_CMD="python"
else
    PY_CMD="python3"
fi

echo "Sử dụng: $PIP_CMD | $PY_CMD"
$PIP_CMD install --upgrade pip

# 2. Kiểm tra tương thích CUDA của PyTorch
echo "🔍 Kiểm tra CUDA..."
CUDA_OK=$($PY_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$CUDA_OK" != "True" ]; then
    echo "🔄 Đang cài đặt PyTorch tương thích CUDA 12.4 (hỗ trợ Driver 570+)..."
    $PIP_CMD install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# 3. Cài đặt Unsloth và thư viện training
$PIP_CMD install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
$PIP_CMD install unsloth_zoo
$PIP_CMD install --no-deps trl peft accelerate bitsandbytes
$PIP_CMD install datasets pyyaml huggingface_hub

# 4. Gỡ bỏ triệt để torchao (tránh lỗi xung đột utils._pytree)
$PIP_CMD uninstall -y torchao 2>/dev/null || true
rm -rf /usr/local/lib/python3*/dist-packages/torchao* /usr/local/lib/python3*/site-packages/torchao* ~/.local/lib/python3*/site-packages/torchao* 2>/dev/null || true

# 5. Dọn dẹp cache giải phóng dung lượng ổ đĩa
$PIP_CMD cache purge 2>/dev/null || true
rm -rf /root/.cache/pip /tmp/* 2>/dev/null || true

echo ""
echo "===================================================================="
echo "💾 [2/4] Kiểm tra và tối ưu dung lượng lưu trữ..."
echo "===================================================================="
FREE_ROOT_GB=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
echo "Dung lượng còn trống trên ổ /: ${FREE_ROOT_GB} GB"

if [ -d "/workspace" ] && [ $(df -BG /workspace 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G') -gt 20 2>/dev/null ]; then
    echo "✅ Sử dụng ổ đĩa dung lượng lớn /workspace"
    mkdir -p /workspace/hf_cache /workspace/outputs
    export HF_HOME=/workspace/hf_cache
    export TRANSFORMERS_CACHE=/workspace/hf_cache
    TARGET_OUT="/workspace/outputs"
elif [ "$FREE_ROOT_GB" -lt 10 ]; then
    echo "⚡ Ổ đĩa / < 10GB. Tự động kích hoạt ổ đĩa RAM siêu tốc /dev/shm (30GB)..."
    mount -o remount,size=30G /dev/shm 2>/dev/null || true
    mkdir -p /dev/shm/hf_cache /dev/shm/outputs
    export HF_HOME=/dev/shm/hf_cache
    export TRANSFORMERS_CACHE=/dev/shm/hf_cache
    TARGET_OUT="/dev/shm/outputs"
else
    TARGET_OUT="./outputs/qwen3_5_4b_financial_sql"
fi
echo "Thư mục Cache: $HF_HOME | Thư mục Output: $TARGET_OUT"

echo ""
echo "===================================================================="
echo "🖥️  [3/4] Thông tin GPU..."
echo "===================================================================="
nvidia-smi
$PY_CMD -c "import torch; print(f'✅ CUDA OK: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')"

echo ""
echo "===================================================================="
echo "🔥 [4/4] Khởi chạy huấn luyện..."
echo "===================================================================="

# Mặc định thêm --output_dir $TARGET_OUT nếu người dùng chưa chỉ định
EXTRA_ARGS=()
if [[ "$*" != *"--output_dir"* ]]; then
    EXTRA_ARGS+=(--output_dir "$TARGET_OUT")
fi

# Tự động thêm token và kích hoạt push hub nếu có biến môi trường HF_TOKEN
if [ -n "$HF_TOKEN" ] && [[ "$*" != *"--hf_token"* ]]; then
    echo "🔑 Phát hiện biến môi trường HF_TOKEN. Tự động bật push_to_hub và export_gguf."
    EXTRA_ARGS+=(--hf_token "$HF_TOKEN" --push_to_hub --export_gguf)
fi

$PY_CMD train_cloud_gpu.py "${EXTRA_ARGS[@]}" "$@"
