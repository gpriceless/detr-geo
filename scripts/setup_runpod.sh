#!/usr/bin/env bash
# setup_runpod.sh -- Bootstrap a fresh RunPod instance for xView fine-tuning.
#
# Sets up the detr-geo project with all dependencies for xView dataset
# processing and RF-DETR fine-tuning. Does NOT download anything
# automatically -- just sets up the environment and prints instructions.
#
# Dataset: xView Detection Challenge (Lam et al., 2018)
# License: CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)
#
# Usage:
#   bash scripts/setup_runpod.sh
#   # Or from a fresh RunPod instance:
#   git clone https://github.com/gpriceless/detr-geo.git
#   cd detr-geo
#   bash scripts/setup_runpod.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_URL="https://github.com/gpriceless/detr-geo.git"
PROJECT_NAME="detr-geo"
# RunPod persistent volume is at /runpod-volume/; /workspace is ephemeral
# Use /runpod-volume/ if it exists, otherwise fall back to /workspace
if [ -d "/runpod-volume" ]; then
    WORK_DIR="/runpod-volume"
else
    WORK_DIR="/workspace"
fi

# Colors for output (if terminal supports it)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

header "RunPod Setup: detr-geo xView Fine-Tuning"

echo ""
echo "  This script sets up the environment for xView fine-tuning."
echo "  It does NOT download the xView dataset (manual registration required)."
echo ""

# -------------------------------------------------------------------
# Step 1: Check GPU
# -------------------------------------------------------------------
header "Step 1/6: Checking GPU"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    info "GPU memory: ${GPU_MEM} MB"

    if [ "$GPU_MEM" -ge 70000 ]; then
        info "Recommended batch_size=8 --grad_accumulation_steps 2 (A100 80GB)"
    elif [ "$GPU_MEM" -ge 35000 ]; then
        info "Recommended batch_size=6 --grad_accumulation_steps 2 (A100 40GB)"
    elif [ "$GPU_MEM" -ge 20000 ]; then
        info "Recommended batch_size=4 --grad_accumulation_steps 4 (24GB+ VRAM)"
        warn "RTX 4090 ran at 84% VRAM with batch_size=4. Monitor closely."
    elif [ "$GPU_MEM" -ge 10000 ]; then
        info "Recommended batch_size=2 --grad_accumulation_steps 8 (10-20GB VRAM)"
    else
        warn "Low VRAM. Use batch_size=1 with --grad_accumulation_steps 16."
    fi
else
    error "nvidia-smi not found. No GPU detected."
    echo "  xView fine-tuning requires a CUDA-capable GPU."
    echo "  Recommended: RTX 4090 (24GB) or A100 (40/80GB)"
fi

# -------------------------------------------------------------------
# Step 2: Check if we're already in the project
# -------------------------------------------------------------------
header "Step 2/6: Setting Up Project"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    info "Already in project directory: ${PROJECT_ROOT}"
    cd "${PROJECT_ROOT}"
elif [ -d "${WORK_DIR}/${PROJECT_NAME}" ]; then
    info "Project already cloned at ${WORK_DIR}/${PROJECT_NAME}"
    cd "${WORK_DIR}/${PROJECT_NAME}"
else
    info "Cloning detr-geo repository..."
    cd "${WORK_DIR}"
    git clone "${REPO_URL}"
    cd "${PROJECT_NAME}"
fi

info "Working directory: $(pwd)"

# -------------------------------------------------------------------
# Step 3: Install Python dependencies
# -------------------------------------------------------------------
header "Step 3/6: Installing Dependencies"

# Upgrade pip first
info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install detr-geo with all extras
info "Installing detr-geo[all]..."
pip install -e ".[all]" --quiet 2>&1 | tail -5

# Install additional dependencies for xView processing
info "Installing additional dependencies..."
pip install pycocotools tqdm --quiet

# Verify key imports
info "Verifying imports..."
python -c "
import sys
checks = []

try:
    import detr_geo
    checks.append(('detr_geo', detr_geo.__version__))
except ImportError as e:
    checks.append(('detr_geo', f'FAILED: {e}'))

try:
    import rfdetr
    checks.append(('rfdetr', 'OK'))
except ImportError as e:
    checks.append(('rfdetr', f'FAILED: {e}'))

try:
    import torch
    cuda = torch.cuda.is_available()
    checks.append(('torch', f'{torch.__version__} (CUDA: {cuda})'))
except ImportError as e:
    checks.append(('torch', f'FAILED: {e}'))

try:
    import rasterio
    checks.append(('rasterio', rasterio.__version__))
except ImportError as e:
    checks.append(('rasterio', f'FAILED: {e}'))

try:
    import geopandas
    checks.append(('geopandas', geopandas.__version__))
except ImportError as e:
    checks.append(('geopandas', f'FAILED: {e}'))

try:
    from pycocotools.coco import COCO
    checks.append(('pycocotools', 'OK'))
except ImportError as e:
    checks.append(('pycocotools', f'FAILED: {e}'))

for name, status in checks:
    print(f'  {name}: {status}')

# Check for failures
failed = [name for name, status in checks if 'FAILED' in str(status)]
if failed:
    print(f'\n  WARNING: {len(failed)} package(s) failed to import: {failed}')
    sys.exit(1)
"

# -------------------------------------------------------------------
# Step 4: Create working directories
# -------------------------------------------------------------------
header "Step 4/6: Creating Working Directories"

XVIEW_RAW="${WORK_DIR}/xview_raw"
XVIEW_DATASET="${WORK_DIR}/xview_dataset"
XVIEW_COCO="${WORK_DIR}/xview_coco"
TRAINING_OUTPUT="${WORK_DIR}/training_output/xview"

mkdir -p "${XVIEW_RAW}"
mkdir -p "${XVIEW_DATASET}"
mkdir -p "${XVIEW_COCO}/train/images"
mkdir -p "${XVIEW_COCO}/valid/images"
mkdir -p "${XVIEW_COCO}/test/images"
mkdir -p "${TRAINING_OUTPUT}"

info "Created directories:"
echo "  ${XVIEW_RAW}         -- Place downloaded xView files here"
echo "  ${XVIEW_DATASET}     -- Preprocessed annotations"
echo "  ${XVIEW_COCO}        -- COCO-format dataset for training"
echo "  ${TRAINING_OUTPUT}   -- Training checkpoints and logs"

# -------------------------------------------------------------------
# Step 5: Check disk space
# -------------------------------------------------------------------
header "Step 5/6: Checking Disk Space"

AVAILABLE_GB=$(df -BG "${WORK_DIR}" | awk 'NR==2 {print $4}' | tr -d 'G')
info "Available disk space: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 35 ]; then
    warn "Less than 35 GB free. xView processing needs ~35 GB."
    warn "Consider using a larger volume."
elif [ "$AVAILABLE_GB" -lt 50 ]; then
    info "Sufficient for processing. Training will add ~2 GB checkpoints."
else
    info "Plenty of disk space available."
fi

# -------------------------------------------------------------------
# Step 6: Print xView download instructions
# -------------------------------------------------------------------
header "Step 6/6: xView Download Instructions"

echo ""
echo "  xView requires MANUAL registration. Cannot automate download."
echo ""
echo "  1. Register at: https://challenge.xviewdataset.org/data-download"
echo "     (any email works, typically approved within 24 hours)"
echo ""
echo "  2. Download to this instance:"
echo "     - train_images.tgz  (~20 GB)"
echo "     - train_labels.tgz  (~180 MB)"
echo ""
echo "  3. Extract files:"
echo "     cd ${XVIEW_RAW}"
echo "     tar -xzf train_images.tgz"
echo "     tar -xzf train_labels.tgz"
echo ""
echo "  4. Verify download:"
echo "     python scripts/download_xview.py --verify --data_dir ${XVIEW_RAW}"
echo ""
echo "  5. Preprocess (remap 60 classes to 5 vehicle classes):"
echo "     python scripts/preprocess_xview.py \\"
echo "       --input ${XVIEW_RAW}/xView_train.geojson \\"
echo "       --output ${XVIEW_DATASET}/xview_remapped.geojson"
echo ""
echo "  6. Process (tile GeoTIFFs into COCO-format training dataset):"
echo "     python scripts/process_xview.py \\"
echo "       --input_dir ${XVIEW_RAW}/train_images/ \\"
echo "       --annotations ${XVIEW_DATASET}/xview_remapped.geojson \\"
echo "       --output_dir ${XVIEW_COCO} \\"
echo "       --tile_size 576 --overlap 0.2 --workers 8"
echo ""
echo "  7. Train:"
echo "     python scripts/train_xview.py \\"
echo "       --dataset_dir ${XVIEW_COCO} \\"
echo "       --output_dir ${TRAINING_OUTPUT} \\"
echo "       --epochs 50 --batch_size 8 --save_interval 1"
echo ""

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
header "Setup Complete"

echo ""
echo "  Environment: $(python --version)"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  Project: $(pwd)"
echo "  Disk: ${AVAILABLE_GB} GB available"
echo ""
echo "  License: xView is CC BY-NC-SA 4.0 (non-commercial, share-alike)"
echo "  Citation: Lam et al., 'xView: Objects in Context' (2018)"
echo ""
echo "  IMPORTANT LESSONS FROM VME + xView TRAINING:"
echo "  1. Use custom_class_names to prevent COCO label mapping bug"
echo "  2. Save checkpoints EVERY EPOCH (--save_interval 1). Lost 6+ hrs in VME"
echo "  3. Best model is usually checkpoint_best_ema.pth, not regular"
echo "  4. If training crashes, pre-crash best may beat resumed training"
echo "  5. Clear GPU cache periodically to prevent OOM"
echo "  6. OOM protection: script auto-reduces batch_size on CUDA OOM"
echo "  7. NEVER kill 'duplicate' training processes -- they may be DataLoader workers"
echo "  8. RTX 4090 hit 84% VRAM at batch_size=4 -- A100 80GB can handle batch_size=8"
echo ""
info "Ready. Follow the download instructions above to begin."
