# Adapting DINO for SageMaker Training

## Overview

DINO is cloned locally at `room_detection_training/DINO/`. Here's what needs to be adapted for SageMaker training.

## Key Files to Understand

1. **`DINO/main.py`** - Main training script
   - Takes `--coco_path` argument (default: `/comp_robot/cv_public_dataset/COCO2017/`)
   - Takes `--config_file` argument (e.g., `config/DINO/DINO_4scale.py`)
   - Takes `--options` to override config values

2. **`DINO/config/DINO/DINO_4scale.py`** - Config file
   - Sets `num_classes=91` (for COCO)
   - Sets batch size, learning rates, etc.

3. **`DINO/datasets/coco.py`** - COCO dataset loader
   - Expects COCO format annotations
   - Uses `args.coco_path` to find data

## What Needs to Change

### 1. Create Custom Config File

Create `DINO/config/DINO/DINO_4scale_room_detection.py`:

```python
_base_ = ['coco_transformer.py']

num_classes = 1  # Single class: room (changed from 91)

# Keep other settings similar to DINO_4scale.py
lr = 0.0001
lr_backbone = 1e-05
batch_size = 2  # Small batch for A10G
epochs = 50
lr_drop = 40
# ... rest of config
```

### 2. Update `train_dino_sagemaker.py`

The current template needs to:

1. **Set SageMaker paths:**
   ```python
   args.coco_path = '/opt/ml/input/data/training'
   args.output_dir = '/opt/ml/model'
   ```

2. **Use custom config:**
   ```python
   args.config_file = '/opt/ml/code/dino/config/DINO/DINO_4scale_room_detection.py'
   ```

3. **Override num_classes:**
   ```python
   args.options = ['num_classes=1']
   ```

4. **Call DINO's main.py:**
   ```python
   from main import main as dino_main
   dino_main(args)
   ```

### 3. Handle Multi-GPU Training

DINO supports distributed training. For SageMaker `ml.g5.2xlarge` (2 GPUs):

```python
# SageMaker sets these automatically
args.world_size = 2
args.dist_url = 'env://'
args.rank = int(os.environ.get('SM_CURRENT_HOST', 0))
args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
```

## Step-by-Step Adaptation

### Step 1: Create Room Detection Config

```bash
cd room_detection_training/DINO
cp config/DINO/DINO_4scale.py config/DINO/DINO_4scale_room_detection.py
# Edit num_classes = 1
```

### Step 2: Test Locally First (Optional)

Before SageMaker, test with local COCO dataset:

```bash
python main.py \
  --config_file config/DINO/DINO_4scale_room_detection.py \
  --coco_path ../../room_detection_dataset_coco \
  --options num_classes=1 \
  --output_dir ./test_output
```

### Step 3: Update `train_dino_sagemaker.py`

Replace the placeholder code with actual DINO training:

```python
import sys
sys.path.insert(0, '/opt/ml/code/dino')

from main import main as dino_main
from main import get_args_parser

# Parse SageMaker hyperparameters
parser = get_args_parser()
args = parser.parse_args()

# Override for SageMaker
args.coco_path = '/opt/ml/input/data/training'
args.output_dir = '/opt/ml/model'
args.config_file = '/opt/ml/code/dino/config/DINO/DINO_4scale_room_detection.py'
args.options = ['num_classes=1']

# Handle distributed training
args.world_size = int(os.environ.get('SM_NUM_GPUS', 1))
args.dist_url = 'env://'

# Run training
dino_main(args)
```

### Step 4: Copy Config to SageMaker Source

Update `sagemaker_train_dino.py` to include the custom config:

```python
# In create_dino_training_script(), add:
# Copy custom config file to source directory
shutil.copy('DINO/config/DINO/DINO_4scale_room_detection.py', 
             source_dir / 'dino_config.py')
```

## Current Status

✅ **DONE:**
- DINO repo cloned locally
- SageMaker training script template created
- COCO dataset ready (no conversion needed!)

⏳ **TODO:**
1. Create `DINO_4scale_room_detection.py` config (num_classes=1)
2. Update `train_dino_sagemaker.py` with actual DINO training code
3. Test locally (optional but recommended)
4. Launch SageMaker training

## Next Steps

1. **Create the config file** - I can help with this
2. **Update the training script** - I can help adapt `train_dino_sagemaker.py`
3. **Test locally** (optional) - Verify it works before SageMaker
4. **Launch training** - Run `python sagemaker_train_dino.py`

Would you like me to:
- Create the custom config file now?
- Update `train_dino_sagemaker.py` with the actual DINO training code?
- Both?

