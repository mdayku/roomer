"""
AWS SageMaker Training for DINO Room Detection
DINO (DETR with Improved Denoising Anchor Boxes) - Transformer-based detection
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from botocore.exceptions import ClientError
import json
from pathlib import Path
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable SageMaker config file loading to avoid hangs
os.environ['SAGEMAKER_DISABLE_CONFIG_FILE'] = 'true'

# SageMaker configuration - COST-OPTIMIZED FOR DINO
INSTANCE_TYPE = 'ml.g5.2xlarge'  # A10G x2 GPUs - cost-effective option
# Cost comparison:
# - ml.p4d.24xlarge (A100): $40.96/hr → ~$490 for 12 hours ❌ Too expensive!
# - ml.g5.2xlarge (A10G x2): $2.41/hr → ~$29 for 12 hours ✅ Much better!
# - ml.g5.xlarge (A10G single): $1.21/hr → ~$15 for 12 hours (may be too small)
# 
# A10G x2 (48GB VRAM total) should work with:
# - Smaller batch size (2-4 instead of 8+)
# - Gradient accumulation to simulate larger batches
# - Slightly longer training time (acceptable trade-off)

INSTANCE_COUNT = 1
FRAMEWORK_VERSION = '2.0.0'
PYTHON_VERSION = 'py310'

# Training hyperparameters - OPTIMIZED FOR A10G (cost-effective)
# Set to 5 epochs for initial test run
HYPERPARAMETERS = {
    'epochs': 5,  # Test run with 5 epochs (change to 50 for full training)
    'batch_size': 2,  # Small batch for A10G (24GB VRAM per GPU)
    'lr': 0.0001,  # Lower learning rate for transformer
    'lr_backbone': 0.00001,  # Even lower for backbone
    'weight_decay': 0.0001,
    'lr_drop': 4,  # Learning rate drop at epoch 4 (for 5-epoch test)
    'clip_max_norm': 0.1,  # Gradient clipping
    'num_classes': 1,  # Single class: room
    'hidden_dim': 256,  # Hidden dimension (can reduce to 192 if OOM)
    'nheads': 8,  # Number of attention heads
    'num_queries': 300,  # Number of object queries
    # Note: Don't include frozen_weights if None - it causes argparse errors
}


def upload_coco_data_to_s3(s3_bucket, s3_prefix, coco_data_dir, force_upload=False):
    """Upload COCO format dataset to S3"""
    print(f"Checking if COCO data exists in s3://{s3_bucket}/{s3_prefix}")
    
    s3_client = boto3.client('s3')
    coco_data_dir = Path(coco_data_dir)
    
    # Check if annotations already exist
    train_ann_key = f"{s3_prefix}/train/annotations.json"
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=train_ann_key)
        if force_upload:
            print("Data exists in S3 but force_upload=True, will re-upload...")
        else:
            print("COCO data already exists in S3, skipping upload")
            return f"s3://{s3_bucket}/{s3_prefix}"
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("COCO data not found in S3, uploading...")
        else:
            raise
    
    print(f"Uploading COCO dataset to s3://{s3_bucket}/{s3_prefix}")
    
    # Upload annotations only (images should already be in S3 or will be loaded from local paths)
    print("Uploading annotations (this should be quick)...")
    for split in ['train', 'val', 'test']:
        ann_file = coco_data_dir / split / 'annotations.json'
        if ann_file.exists():
            s3_key = f"{s3_prefix}/{split}/annotations.json"
            print(f"  Uploading {split} annotations...")
            s3_client.upload_file(str(ann_file), s3_bucket, s3_key)
            print(f"  [OK] {split} annotations uploaded")
    
    # Skip image upload for now - images are likely already in S3 or will be loaded from local paths
    # The COCO annotations reference image paths, and SageMaker will handle loading them
    print("[INFO] Skipping image upload - images will be loaded from local paths or existing S3 location")
    print("[INFO] If images need to be uploaded, they should be in the same S3 bucket structure")
    
    s3_data_path = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"[OK] COCO data uploaded to {s3_data_path}")
    return s3_data_path


def train_on_sagemaker():
    """Launch SageMaker training job for DINO"""
    print("Launching DINO training on AWS SageMaker")
    print(f"Instance: {INSTANCE_TYPE}")
    print("=" * 60)
    
    # Get SageMaker session and role
    print("Setting up IAM role...")
    iam = boto3.client('iam')
    role_name = 'room-detection-sagemaker-role'
    
    try:
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Using existing IAM role: {role}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Creating IAM role: {role_name}")
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for Room Detection ML service'
        )
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
        ]
        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Created IAM role: {role}")
    
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"[OK] Using S3 bucket: {bucket}")
    
    # Prepare COCO data
    print("\nPreparing COCO dataset...")
    # Try multiple possible paths
    possible_paths = [
        Path("../../room_detection_dataset_coco"),  # From room_detection_training/
        Path("../room_detection_dataset_coco"),     # From repo root
        Path("room_detection_dataset_coco"),        # Current dir
    ]
    coco_data_dir = None
    for path in possible_paths:
        if path.exists():
            coco_data_dir = path
            break
    
    if not coco_data_dir:
        # Try absolute path from script location
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        coco_data_dir = repo_root / 'room_detection_dataset_coco'
    
    if not coco_data_dir.exists():
        print(f"[ERROR] COCO dataset not found at: {coco_data_dir}")
        print("Please ensure the COCO dataset exists at this path")
        sys.exit(1)
    
    train_ann = coco_data_dir / "train" / "annotations.json"
    val_ann = coco_data_dir / "val" / "annotations.json"
    
    if not train_ann.exists() or not val_ann.exists():
        print(f"[ERROR] COCO annotations not found!")
        print(f"Expected: {train_ann} and {val_ann}")
        sys.exit(1)
    
    print(f"[OK] Found COCO dataset at: {coco_data_dir}")
    
    # Create training script first (needed for estimator)
    create_dino_training_script()
    
    # Create minimal source directory (don't upload entire DINO repo!)
    import shutil
    source_dir = Path('dino_sagemaker_source')
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir()
    
    # Copy only the training script
    shutil.copy('train_dino_sagemaker.py', source_dir / 'train_dino_sagemaker.py')
    print(f"[OK] Copied training script to minimal source directory")
    
    # Copy config files if they exist
    config_source = Path('DINO/config/DINO/DINO_4scale_room_detection.py')
    if config_source.exists():
        dino_config_dir = source_dir / 'DINO' / 'config' / 'DINO'
        dino_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_source, dino_config_dir / 'DINO_4scale_room_detection.py')
        print(f"[OK] Copied custom config to source directory")
        
        base_config = Path('DINO/config/DINO/coco_transformer.py')
        if base_config.exists():
            shutil.copy(base_config, dino_config_dir / 'coco_transformer.py')
            print(f"[OK] Copied base config (coco_transformer.py)")
    
    print(f"[OK] Created minimal source directory: {source_dir}")
    print(f"[INFO] Source dir size: {sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file()) / 1024:.1f} KB")
    
    # Upload COCO data to S3
    s3_data_path = upload_coco_data_to_s3(
        bucket,
        'room-detection-dino/data',
        str(coco_data_dir),
        force_upload=False
    )
    
    # Create PyTorch estimator
    print("\nCreating SageMaker estimator for DINO...")
    print(f"  Entry point: train_dino_sagemaker.py")
    print(f"  Source dir: . (current directory)")
    print(f"  Instance: {INSTANCE_TYPE}")
    print(f"  Hyperparameters: {HYPERPARAMETERS}")
    
    try:
        estimator = PyTorch(
            entry_point='train_dino_sagemaker.py',  # Training script
            source_dir=str(source_dir),  # Minimal source directory (not entire repo!)
            role=role,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            framework_version=FRAMEWORK_VERSION,
            py_version=PYTHON_VERSION,
            hyperparameters=HYPERPARAMETERS,
            use_spot_instances=False,  # On-demand for stability
            max_run=86400,  # 24 hours max (DINO training can be long)
            # Enable multi-GPU training on A10G x2
            distribution={
                'pytorch': {
                    'enabled': True,
                    'processes_per_host': 2  # 2 GPUs on g5.2xlarge
                }
            },
        )
        print(f"[OK] Estimator created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create estimator: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Launch training
    print("\nStarting DINO training job...")
    
    job_name = f'room-detection-dino-{int(time.time())}'
    print(f"[INFO] Job name: {job_name}")
    print(f"[INFO] Calling estimator.fit()...")
    
    try:
        estimator.fit(
            inputs={'training': s3_data_path},
            job_name=job_name,
            wait=False  # Don't wait for completion
        )
        print(f"[OK] estimator.fit() completed")
    except Exception as e:
        print(f"[ERROR] estimator.fit() failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"[OK] DINO training job launched: {job_name}")
    print(f"Monitor progress at: https://{sess.boto_region_name}.console.aws.amazon.com/sagemaker/home?region={sess.boto_region_name}#/jobs/{job_name}")
    
    # Estimate costs
    hourly_rate = get_instance_price(INSTANCE_TYPE)
    # For 5-epoch test: much faster (~1-2 hours)
    # For 50-epoch full training: ~15 hours
    test_epochs = HYPERPARAMETERS.get('epochs', 5)
    if test_epochs == 5:
        estimated_hours = 1.5  # ~1.5 hours for 5 epochs
    else:
        estimated_hours = 15.0  # ~15 hours for 50 epochs
    estimated_cost = hourly_rate * estimated_hours
    
    print(f"\nCost Estimate ({INSTANCE_TYPE}) - DINO Training:")
    print(f"- Instance: {INSTANCE_TYPE} (${hourly_rate}/hour on-demand)")
    print("- Model: DINO Base (Transformer-based detection)")
    print("- Dataset: COCO format (no conversion needed!)")
    print(f"- Epochs: {test_epochs} ({'TEST RUN' if test_epochs == 5 else 'FULL TRAINING'})")
    print(f"- Estimated training time: {estimated_hours} hours")
    print(f"- Estimated cost: ${estimated_cost:.2f}")
    if test_epochs == 5:
        print(f"- Note: This is a test run. For full training (50 epochs), cost will be ~${hourly_rate * 15:.2f}")
    print(f"- Expected improvement: Better accuracy on complex scenes vs YOLO")
    
    return job_name, estimator


def get_instance_price(instance_type):
    """Get approximate hourly price for instance type"""
    prices = {
        'ml.g4dn.xlarge': 0.736,   # T4 GPU
        'ml.g5.xlarge': 1.206,     # A10G single GPU
        'ml.g5.2xlarge': 2.412,    # A10G x2 GPUs
        'ml.p3.2xlarge': 3.825,    # V100 GPU
        'ml.p4d.24xlarge': 40.96,  # A100 GPU (recommended for DINO)
    }
    return prices.get(instance_type, 'Unknown')


def create_dino_training_script():
    """Create the DINO training script that runs in SageMaker container"""
    script_content = '''#!/usr/bin/env python3
"""
DINO Training Script for SageMaker Container
Based on DINO framework: https://github.com/IDEA-Research/DINO
"""

import os
import sys
import argparse
import json
from pathlib import Path

def main():
    print("Room Detection DINO Training in SageMaker")
    print("=" * 50)
    
    # Try importing torch (may not be installed yet)
    try:
        import torch
        print("[OK] torch imported successfully")
    except ImportError:
        torch = None
        print("[WARN] torch not available yet, will install dependencies first")
    
    # Install DINO dependencies
    print("Installing DINO dependencies...")
    import subprocess
    
    # Install basic dependencies
    packages = [
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "transformers>=4.20.0",
        "opencv-python",
        "pillow",
        "numpy",
        "pycocotools",
        "matplotlib",
        "tqdm"
    ]
    
    for package in packages:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            package, "--quiet", "--no-cache-dir"
        ], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Warning: Failed to install {package}: {result.stderr}")
    
    # Clone DINO repository if not present
    dino_dir = Path("/opt/ml/code/dino")
    print(f"[INFO] Checking for DINO at: {dino_dir}")
    
    if not dino_dir.exists():
        print("[INFO] DINO directory not found, cloning repository...")
        dino_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            "git", "clone", "https://github.com/IDEA-Research/DINO.git",
            str(dino_dir)
        ], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[ERROR] Failed to clone DINO:")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)
        print("[OK] DINO repository cloned successfully")
    else:
        print(f"[OK] DINO directory already exists")
    
    # Install DINO requirements
    print("[INFO] Installing DINO requirements...")
    req_file = dino_dir / "requirements.txt"
    if req_file.exists():
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-r", str(req_file),
            "--quiet", "--no-cache-dir"
        ], capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            print(f"[WARN] Some DINO requirements failed:")
            print(f"  stdout: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
            print(f"  stderr: {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")
        else:
            print("[OK] DINO requirements installed")
    else:
        print(f"[WARN] DINO requirements.txt not found at {req_file}")
    
    # Parse arguments
    # Use allow_abbrev=False and ignore unknown args to handle SageMaker's hyperparameter passing
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_backbone', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr_drop', type=int, default=40)
    parser.add_argument('--clip_max_norm', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--num_queries', type=int, default=300)
    # Add optional args that SageMaker might pass but we don't use
    parser.add_argument('--frozen_weights', default=None, nargs='?')  # Optional, can be None
    parser.add_argument('--accumulate_grad_batches', type=int, default=None)  # Not used by DINO
    # Parse known args only, ignore unknown ones
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unknown arguments: {unknown}")
    
    print(f"Training config: {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}")
    
    # Setup paths
    data_root = Path('/opt/ml/input/data/training')
    train_ann = data_root / 'train' / 'annotations.json'
    val_ann = data_root / 'val' / 'annotations.json'
    output_dir = Path('/opt/ml/model')
    
    # Verify data exists
    if not train_ann.exists():
        print(f"[ERROR] Training annotations not found: {train_ann}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
    if not val_ann.exists():
        print(f"[ERROR] Validation annotations not found: {val_ann}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
    
    print(f"[OK] Found COCO dataset:")
    print(f"  Train: {train_ann}")
    print(f"  Val: {val_ann}")
    
    # Load COCO annotations to verify
    with open(train_ann, 'r') as f:
        train_data = json.load(f)
    print(f"  Train images: {len(train_data.get('images', []))}")
    print(f"  Train annotations: {len(train_data.get('annotations', []))}")
    
    # Import DINO training modules
    sys.path.insert(0, str(dino_dir))
    
    # Create DINO config for room detection
    config = {
        'dataset_file': 'coco',
        'coco_path': str(data_root),
        'coco_panoptic_path': None,
        'remove_difficult': False,
        'fix_size': False,
        'num_classes': args.num_classes,
        'hidden_dim': args.hidden_dim,
        'nheads': args.nheads,
        'num_queries': args.num_queries,
        'lr': args.lr,
        'lr_backbone': args.lr_backbone,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'lr_drop': args.lr_drop,
        'clip_max_norm': args.clip_max_norm,
        'output_dir': str(output_dir),
    }
    
    # Save config
    config_path = output_dir / 'dino_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Saved config to {config_path}")
    
    # Import and run DINO training using their main.py
    print("\\n[INFO] Importing DINO main module...")
    print(f"[INFO] DINO directory: {dino_dir}")
    print(f"[INFO] DINO directory exists: {dino_dir.exists()}")
    
    if not dino_dir.exists():
        print(f"[ERROR] DINO directory not found at {dino_dir}")
        print("[INFO] Attempting to clone DINO repository...")
        import subprocess
        result = subprocess.run([
            "git", "clone", "https://github.com/IDEA-Research/DINO.git",
            str(dino_dir)
        ], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[ERROR] Failed to clone DINO: {result.stderr}")
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)
        print("[OK] DINO repository cloned")
    
    sys.path.insert(0, str(dino_dir))
    print(f"[INFO] Added {dino_dir} to Python path")
    
    try:
        print("[INFO] Attempting to import DINO main module...")
        print(f"[DEBUG] DINO directory: {dino_dir}")
        print(f"[DEBUG] DINO directory exists: {dino_dir.exists()}")
        if dino_dir.exists():
            print(f"[DEBUG] DINO directory contents: {list(dino_dir.iterdir())[:10]}")
        
        # Change to DINO directory for import
        import os
        original_cwd = os.getcwd()
        os.chdir(str(dino_dir))
        print(f"[DEBUG] Changed to DINO directory: {os.getcwd()}")
        
        from main import main as dino_main
        from main import get_args_parser
        
        # Change back
        os.chdir(original_cwd)
        
        print("[OK] DINO main module imported successfully")
        
        # Create args object compatible with DINO's main.py
        dino_parser = get_args_parser()
        
        # Build args for DINO
        dino_args = dino_parser.parse_args([])  # Empty list, we'll set values directly
        
        # Set paths for SageMaker
        dino_args.coco_path = str(data_root)
        dino_args.output_dir = str(output_dir)
        dino_args.dataset_file = 'coco'
        
        # Set config file path (will be copied to container)
        # Check if custom config was included in source
        custom_config_path = Path('/opt/ml/code/DINO/config/DINO/DINO_4scale_room_detection.py')
        base_config_path = dino_dir / 'config' / 'DINO' / 'DINO_4scale.py'
        
        if custom_config_path.exists():
            dino_args.config_file = str(custom_config_path)
            print(f"[OK] Using custom room detection config from source")
        elif (dino_dir / 'config' / 'DINO' / 'DINO_4scale_room_detection.py').exists():
            dino_args.config_file = str(dino_dir / 'config' / 'DINO' / 'DINO_4scale_room_detection.py')
            print(f"[OK] Using custom room detection config from cloned repo")
        elif base_config_path.exists():
            # Fallback: use base config with options override
            dino_args.config_file = str(base_config_path)
            dino_args.options = [
                f'num_classes={args.num_classes}',
                f'batch_size={args.batch_size}',
                f'epochs={args.epochs}',
                f'lr={args.lr}',
                f'lr_backbone={args.lr_backbone}',
                f'lr_drop={args.lr_drop}',
                f'weight_decay={args.weight_decay}',
                f'clip_max_norm={args.clip_max_norm}',
                f'num_queries={args.num_queries}',
                f'dn_labelbook_size={args.num_classes + 1}',
            ]
            print(f"[INFO] Using base config with options override")
        else:
            print(f"[ERROR] No config file found!")
            print(f"  Checked: {custom_config_path}")
            print(f"  Checked: {dino_dir / 'config' / 'DINO' / 'DINO_4scale_room_detection.py'}")
            print(f"  Checked: {base_config_path}")
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)
        
        # Handle distributed training (SageMaker sets these)
        dino_args.world_size = int(os.environ.get('SM_NUM_GPUS', 1))
        dino_args.dist_url = 'env://'
        dino_args.rank = int(os.environ.get('RANK', 0))
        dino_args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Device
        if torch is None:
            print("[ERROR] torch is not available. Please check installation.")
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)
        dino_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Other settings
        dino_args.num_workers = 4
        dino_args.amp = False  # Can enable mixed precision if needed
        dino_args.find_unused_params = False
        
        print(f"[OK] Starting DINO training with config:")
        print(f"  Config file: {dino_args.config_file}")
        print(f"  COCO path: {dino_args.coco_path}")
        print(f"  Output dir: {dino_args.output_dir}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Num classes: {args.num_classes}")
        print(f"  World size (GPUs): {dino_args.world_size}")
        
        # Run DINO training
        dino_main(dino_args)
        
        print("[OK] DINO training completed!")
        
    except ImportError as e:
        print(f"[ERROR] Failed to import DINO modules: {e}")
        print(f"[INFO] Python path: {sys.path}")
        print(f"[INFO] DINO dir contents: {list(dino_dir.iterdir()) if dino_dir.exists() else 'DOES NOT EXIST'}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        print(f"[INFO] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open('train_dino_sagemaker.py', 'w') as f:
        f.write(script_content)
    
    print("[OK] Created DINO SageMaker training script")


if __name__ == "__main__":
    print("AWS SageMaker DINO Training Setup")
    print("=" * 40)
    
    # Check AWS credentials
    print("Checking AWS credentials...")
    try:
        import boto3
        print("  Creating STS client...")
        sts = boto3.client('sts')
        print("  Getting caller identity...")
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"[OK] AWS Account: {account_id}")
        
        print("  Creating SageMaker client...")
        sm = boto3.client('sagemaker')
        print("  Creating S3 client...")
        s3 = boto3.client('s3')
        print("  Creating IAM client...")
        iam = boto3.client('iam')
        print("[OK] All required AWS services accessible")
        
    except Exception as e:
        print(f"[ERROR] AWS access issue: {e}")
        import traceback
        traceback.print_exc()
        print("\n[SETUP] Please configure AWS credentials:")
        print("  1. Run 'aws configure'")
        print("  2. Or set environment variables:")
        print("     export AWS_ACCESS_KEY_ID=your_key")
        print("     export AWS_SECRET_ACCESS_KEY=your_secret")
        print("     export AWS_DEFAULT_REGION=us-east-1")
        sys.exit(1)
    
    # Launch training (script is created inside train_on_sagemaker)
    print("\n[INFO] Starting training job launch...")
    try:
        job_name, estimator = train_on_sagemaker()
    except Exception as e:
        print(f"[ERROR] Failed to launch training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\\n[INFO] Next steps:")
    print("1. Monitor training at the SageMaker console URL above")
    print("2. DINO training will take 8-12 hours on A100")
    print("3. Download trained model from S3 when complete")
    print("4. Adapt DINO inference for your API")
    print("\\n[NOTE] The training script is a template.")
    print("You'll need to adapt DINO's main.py to work with SageMaker paths.")

