"""
AWS SageMaker Training for YOLO Room Detection
Supports both DETECTION (bbox) and SEGMENTATION (polygon) models
Cloud training with enterprise GPUs for maximum speed
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import json
from pathlib import Path
import os
import sys
import time
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# Disable SageMaker config file loading to avoid hangs
os.environ['SAGEMAKER_DISABLE_CONFIG_FILE'] = 'true'

# SageMaker configuration - OPTIMIZED FOR YOLO LARGE + 200 EPOCHS
INSTANCE_TYPE = 'ml.g5.2xlarge'  # A10G x2 GPUs - best cost/performance for large models
# Why this choice for YOLO Large + 200 epochs:
# - 48GB VRAM total (24GB per GPU) → handles yolov8l/x models easily
# - Multi-GPU training → faster convergence, shorter training time
# - Cost-effective: ~$2.41/hr vs $9.60/hr for single A100
# Alternatives:
# 'ml.g5.xlarge' - A10G single GPU ($1.21/hr, 24GB VRAM)
# 'ml.p4d.24xlarge' - A100 single GPU ($40.96/hr, 80GB VRAM) - overkill
# 'ml.g4dn.xlarge' - T4 GPU ($0.74/hr) - too slow for 200 epochs

INSTANCE_COUNT = 1
FRAMEWORK_VERSION = '2.0.0'
PYTHON_VERSION = 'py310'

def get_hyperparameters(task='detect', model_size='l', imgsz=800):
    """
    Get hyperparameters based on task type
    
    Args:
        task: 'detect' (bbox) or 'segment' (polygons)
        model_size: 'n', 's', 'm', 'l', 'x' (default: 'l' for large)
        imgsz: Training image size (default: 800)
               - CubiCasa5K analysis: mean=972x866px, median=775x698px
               - 640: Fast, good for 44% of images
               - 800: Balanced, good for 27% of images (RECOMMENDED)
               - 1024: High detail, only 13% are this large (may upscale)
    
    Models use COCO pretrained weights for transfer learning:
    - yolov8l.pt = YOLOv8 Large pretrained on COCO (80 classes)
    - yolov8l-seg.pt = YOLOv8 Large Segmentation pretrained on COCO
    """
    base_params = {
        'epochs': 200,
        'batch_size': 4,  # Will be adjusted based on model size and imgsz
        'imgsz': imgsz,
        'data': 'data.yaml',
        'workers': 4,
        'patience': 20,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Lower LR for transfer learning
        'save_period': 25,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'task': task,
    }
    
    # Adjust batch size based on model size AND image size
    # Larger images = more VRAM = smaller batch
    imgsz_factor = (imgsz / 640) ** 2  # Quadratic scaling for memory
    
    # Set model weights and base batch size
    if task == 'detect':
        base_params['weights'] = f'yolov8{model_size}.pt'  # COCO pretrained detection
        if model_size in ['x', 'l']:
            base_batch = 4
        elif model_size == 'm':
            base_batch = 8
        else:
            base_batch = 16
            
    elif task == 'segment':
        base_params['weights'] = f'yolov8{model_size}-seg.pt'  # COCO pretrained segmentation
        # Segmentation is more memory intensive
        if model_size in ['x', 'l']:
            base_batch = 3
        elif model_size == 'm':
            base_batch = 6
        else:
            base_batch = 12
        # Note: 'seg' loss weight is handled internally by ultralytics, don't pass as hyperparam
    
    # Adjust batch size for image resolution
    adjusted_batch = max(1, int(base_batch / imgsz_factor))
    base_params['batch_size'] = adjusted_batch
    
    return base_params


def upload_data_to_s3(s3_bucket, s3_prefix, local_data_dir, force_upload=False, reuse_images_from_s3=None):
    """Upload training data to S3, optionally reusing images from another dataset"""
    print(f"Checking if data exists in s3://{s3_bucket}/{s3_prefix}")

    s3_client = boto3.client('s3')
    from botocore.exceptions import ClientError

    print(f"Checking and uploading data to s3://{s3_bucket}/{s3_prefix}")
    print("Using parallel uploads (10 threads) for faster transfer...")
    
    # Collect all files to process
    files_to_process = []
    for root, dirs, files in os.walk(local_data_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_data_dir)
            # CRITICAL: Normalize path separators to forward slashes for S3/Linux
            relative_path = relative_path.replace('\\', '/')
            s3_key = f"{s3_prefix}/{relative_path}"
            files_to_process.append((local_path, relative_path, s3_key))
    
    total_files = len(files_to_process)
    print(f"Total local files: {total_files}")
    
    # Thread-safe counters
    uploaded_count = 0
    skipped_existing = 0
    copied_from_other = 0
    lock = Lock()
    
    def upload_file_task(task_data):
        """Upload a single file (or skip if exists, or copy from other dataset)"""
        local_path, relative_path, s3_key = task_data
        nonlocal uploaded_count, skipped_existing, copied_from_other
        
        # Each thread needs its own S3 client
        thread_s3_client = boto3.client('s3')
        
        try:
            # Check if file already exists in S3
            try:
                thread_s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
                # File exists, skip unless force_upload
                if not force_upload:
                    with lock:
                        skipped_existing += 1
                    return 'skipped'
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise  # Some other error
            
            # Check if we should reuse images from another dataset
            if reuse_images_from_s3 and '/images/' in relative_path:
                try:
                    source_key = f"{reuse_images_from_s3}/{relative_path}"
                    thread_s3_client.head_object(Bucket=s3_bucket, Key=source_key)
                    thread_s3_client.copy_object(
                        CopySource={'Bucket': s3_bucket, 'Key': source_key},
                        Bucket=s3_bucket,
                        Key=s3_key
                    )
                    with lock:
                        copied_from_other += 1
                    return 'copied'
                except ClientError:
                    pass  # Source doesn't exist, upload normally
            
            # Upload the file
            thread_s3_client.upload_file(local_path, s3_bucket, s3_key)
            with lock:
                uploaded_count += 1
            return 'uploaded'
        
        except Exception as e:
            print(f"  [ERROR] Failed to process {relative_path}: {e}")
            return 'error'
    
    # Process files in parallel
    print("Starting parallel upload...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(upload_file_task, task): task for task in files_to_process}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 200 == 0 or completed == total_files:
                print(f"  Progress: {completed}/{total_files} files processed ({completed/total_files*100:.1f}%)")
            future.result()  # Will raise exception if task failed

    s3_data_path = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"\n[OK] Data sync complete: {s3_data_path}")
    print(f"  - Uploaded: {uploaded_count} new files")
    print(f"  - Skipped: {skipped_existing} existing files")
    if copied_from_other > 0:
        print(f"  - Copied: {copied_from_other} images from other dataset (optimization)")
    print(f"  - Total: {total_files} files in dataset")
    return s3_data_path


def create_data_yaml_for_sagemaker(task='detect'):
    """Create data.yaml for SageMaker container"""
    yaml_content = f"""# YOLO Data Configuration for SageMaker
path: /opt/ml/input/data/training  # SageMaker data channel
train: train/images
val: val/images
test: test/images

# Classes
nc: 1
names: ['room']

# Task type
task: {task}
"""

    # Save locally for upload
    with open('data.yaml', 'w') as f:
        f.write(yaml_content.strip())

    print(f"[OK] Created data.yaml for SageMaker (task: {task})")
    return 'data.yaml'


def train_on_sagemaker(task='detect', local_data_dir=None, model_size='l', imgsz=800, s3_prefix=None):
    """
    Launch SageMaker training job
    
    Args:
        task: 'detect' for detection (bbox) or 'segment' for segmentation (polygons)
        local_data_dir: Path to local YOLO dataset directory (auto-detected if None)
        model_size: Model size: 'n', 's', 'm', 'l', 'x' (default: 'l' = large)
        imgsz: Training resolution (default: 800, based on CubiCasa5K analysis)
    """
    print("="*60)
    print("AWS SageMaker YOLO Training - Transfer Learning")
    print("="*60)
    print(f"Task: {task.upper()}")
    print(f"Model: YOLOv8-{model_size.upper()} (COCO pretrained)")
    print(f"Image size: {imgsz}x{imgsz}px")
    print(f"Instance: {INSTANCE_TYPE} ({INSTANCE_COUNT}x)")
    print("="*60)

    # Get SageMaker session and role
    print("Setting up IAM role...")
    iam = boto3.client('iam')
    role_name = 'room-detection-sagemaker-role'

    try:
        # Try to get existing role
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Using existing IAM role: {role}")
    except iam.exceptions.NoSuchEntityException:
        # Create new role
        print(f"Creating IAM role: {role_name}")

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for Room Detection ML service'
        )

        # Attach policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
        ]

        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

        # Get the role ARN
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Created IAM role: {role}")

    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"[OK] Using S3 bucket: {bucket}")

    # Prepare data
    print("\nPreparing training data...")
    
    # Auto-detect or use specified data directory based on TASK and S3_PREFIX
    if local_data_dir is None:
        # First try to infer from S3 prefix if provided
        if s3_prefix:
            if '2class' in s3_prefix or '2-class' in s3_prefix:
                local_data_dir = Path("./yolo_room_wall_2class")
                print("Auto-detected: 2-CLASS dataset (room + wall) from S3 prefix")
            elif 'segment' in s3_prefix or 'seg' in s3_prefix:
                local_data_dir = Path("./yolo_room_seg")
                print("Auto-detected: SEGMENTATION dataset (polygons) from S3 prefix")
            elif '1class' in s3_prefix or 'detect' in s3_prefix:
                local_data_dir = Path("./yolo_room_only")
                print("Auto-detected: 1-CLASS dataset (room only) from S3 prefix")
        
        # Fallback to task-based detection if no S3 prefix or couldn't infer
        if local_data_dir is None:
            if task == 'detect':
                local_data_dir = Path("./yolo_room_only")  # BBOX dataset (default to 1-class)
                print("Auto-detected: DETECTION dataset (bounding boxes, 1-class)")
            elif task == 'segment':
                local_data_dir = Path("./yolo_room_seg")  # POLYGON dataset
                print("Auto-detected: SEGMENTATION dataset (polygons)")
            else:
                raise ValueError(f"Unknown task: {task}")
    else:
        local_data_dir = Path(local_data_dir)
    
    if not local_data_dir.exists():
        print(f"\n[ERROR] Dataset not found: {local_data_dir.absolute()}")
        print(f"\nDataset Requirements by Task:")
        print("  - DETECT task → needs yolo_room_only/ (bbox labels)")
        print("  - SEGMENT task → needs yolo_room_seg/ (polygon labels)")
        print(f"\nPlease build the {task} dataset:")
        if task == 'detect':
            print("  python rebuild_yolo_dataset.py")
        elif task == 'segment':
            print("  python rebuild_yolo_seg_dataset.py")
        sys.exit(1)
    
    print(f"Dataset path: {local_data_dir.absolute()}")
    
    # Verify data format matches task
    sample_label = None
    labels_dir = local_data_dir / 'train' / 'labels'
    if labels_dir.exists():
        label_files = list(labels_dir.glob('*.txt'))
        if label_files:
            with open(label_files[0], 'r') as f:
                first_line = f.readline().strip()
                coords = first_line.split()[1:]  # Skip class ID
                if task == 'detect' and len(coords) == 4:
                    print("[OK] Detected BBOX format (4 coords) - matches DETECT task")
                elif task == 'segment' and len(coords) > 4:
                    print(f"[OK] Detected POLYGON format ({len(coords)} coords) - matches SEGMENT task")
                elif task == 'detect' and len(coords) > 4:
                    print(f"[ERROR] Dataset has POLYGON labels but task is DETECT!")
                    print("  Use --task segment or rebuild with rebuild_yolo_dataset.py")
                    sys.exit(1)
                elif task == 'segment' and len(coords) == 4:
                    print(f"[ERROR] Dataset has BBOX labels but task is SEGMENT!")
                    print("  Use --task detect or rebuild with rebuild_yolo_seg_dataset.py")
                    sys.exit(1)
    
    # Verify dataset structure
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for req_dir in required_dirs:
        dir_path = local_data_dir / req_dir
        if not dir_path.exists():
            print(f"[ERROR] Required directory missing: {dir_path}")
            sys.exit(1)
    
    print("[OK] Dataset structure verified")

    # Create data.yaml for SageMaker
    data_yaml = create_data_yaml_for_sagemaker(task=task)

    # Copy the SageMaker data.yaml to overwrite the local one
    import shutil
    shutil.copy('data.yaml', str(local_data_dir / 'data.yaml'))
    print("[OK] Updated data.yaml in dataset directory")

    # Determine S3 prefix based on task and dataset
    if args.s3_prefix:
        s3_prefix = args.s3_prefix
        print(f"[INFO] Using custom S3 prefix: {s3_prefix}")
    else:
        # Auto-detect based on task and dataset directory
        if task == 'detect':
            if 'wall' in str(local_data_dir).lower() or '2class' in str(local_data_dir).lower():
                s3_prefix = 'room-detection-training/data-detect-2class'
                print("[INFO] Detected 2-class dataset (wall + room)")
            else:
                s3_prefix = 'room-detection-training/data-detect-1class'
                print("[INFO] Detected 1-class dataset (room only)")
        elif task == 'segment':
            s3_prefix = 'room-detection-training/data-segment'
            print("[INFO] Using segmentation dataset")
        else:
            s3_prefix = f'room-detection-training/data-{task}'
    
    print(f"S3 location: s3://{bucket}/{s3_prefix}")
    
    # Check if images are already in S3 from 1-class detect dataset (for reuse optimization)
    reuse_images_s3 = False
    detect_1class_prefix = 'room-detection-training/data-detect-1class'
    if task in ['segment'] or '2class' in s3_prefix:
        try:
            s3_client = boto3.client('s3')
            from botocore.exceptions import ClientError
            # Check if 1-class detect images exist
            s3_client.head_object(Bucket=bucket, Key=f"{detect_1class_prefix}/train/images/000001.png")
            reuse_images_s3 = True
            print(f"[OPTIMIZATION] Images already in S3 from 1-class detect dataset")
            print(f"Will only upload labels for {task} task (much faster!)")
        except:
            pass
    
    s3_data_path = upload_data_to_s3(
        bucket,
        s3_prefix,
        str(local_data_dir),
        reuse_images_from_s3=detect_1class_prefix if reuse_images_s3 else None
    )

    # Get hyperparameters for task
    hyperparameters = get_hyperparameters(task=task, model_size=model_size, imgsz=imgsz)
    
    # Override epochs if specified
    if args.epochs is not None:
        hyperparameters['epochs'] = args.epochs
        print(f"[INFO] Overriding epochs: {args.epochs} (default was 200)")
    
    # Create PyTorch estimator
    print("\nCreating SageMaker estimator...")
    print(f"Model: {hyperparameters['weights']} (COCO pretrained -> transfer learning)")
    print(f"Task: {task}")
    print(f"Image size: {imgsz}x{imgsz}px")
    print(f"Batch size: {hyperparameters['batch_size']} (auto-adjusted for {imgsz}px)")
    print(f"Epochs: {hyperparameters['epochs']}")

    estimator = PyTorch(
        entry_point='train_yolo_sagemaker.py',
        source_dir=None,
        role=role,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        framework_version=FRAMEWORK_VERSION,
        py_version=PYTHON_VERSION,
        hyperparameters=hyperparameters,
        use_spot_instances=False,
        max_run=43200,  # 12 hours
        distribution={
            'pytorch': {
                'enabled': True,
                'processes_per_host': 2
            }
        },
    )

    # Launch training
    print("\nStarting training job...")

    # Create descriptive job name with metadata
    # Format: room-{task}-{dataset_type}-yolo{model_size}-{imgsz}px-{epochs}ep-{timestamp}
    # Examples:
    #   room-detect-1class-yolol-800px-20ep-1234567890
    #   room-segment-1class-yolol-1024px-200ep-1234567890
    #   room-detect-2class-yolol-800px-200ep-1234567890
    
    # Determine dataset type from S3 prefix
    if '1class' in s3_prefix or 'room_only' in str(local_data_dir).lower():
        dataset_type = '1class'
    elif '2class' in s3_prefix or 'wall' in str(local_data_dir).lower():
        dataset_type = '2class'
    else:
        dataset_type = '1class'  # default
    
    job_name = f'room-{task}-{dataset_type}-yolo{model_size}-{imgsz}px-{hyperparameters["epochs"]}ep-{int(time.time())}'
    
    print(f"Job name: {job_name}")
    print(f"  Task: {task}")
    print(f"  Dataset: {dataset_type}")
    print(f"  Model: yolo{model_size}")
    print(f"  Resolution: {imgsz}px")
    print(f"  Epochs: {hyperparameters['epochs']}")

    estimator.fit(
        inputs={'training': s3_data_path},
        job_name=job_name,
        wait=False
    )

    print(f"[OK] Training job launched: {job_name}")
    print(f"Monitor progress at: https://{sess.boto_region_name}.console.aws.amazon.com/sagemaker/home?region={sess.boto_region_name}#/jobs/{job_name}")

    # Estimate costs and time based on epochs
    hourly_rate = get_instance_price(INSTANCE_TYPE)
    epochs = hyperparameters['epochs']
    
    # Base estimates for 200 epochs
    if task == 'detect':
        base_hours = 8.0  # Detection: ~8 hours for 200 epochs
    else:
        base_hours = 10.0  # Segmentation: ~10 hours for 200 epochs
    
    # Scale based on actual epochs (assuming linear scaling, conservative estimate)
    estimated_hours = (epochs / 200) * base_hours
    estimated_cost = hourly_rate * estimated_hours

    print(f"\nCost Estimate ({INSTANCE_TYPE}) - {task.upper()}:")
    print(f"- Instance: {INSTANCE_TYPE} (${hourly_rate}/hour on-demand)")
    print("- Multi-GPU training: 2x A10G GPUs")
    print(f"- Model: {hyperparameters['weights']}")
    print(f"- Task: {task}")
    print(f"- Epochs: {epochs} {'(QUICK DEMO)' if epochs < 50 else '(FULL TRAINING)' if epochs >= 150 else ''}")
    print(f"- Estimated training time: {estimated_hours:.1f} hours")
    print(f"- Estimated cost: ${estimated_cost:.2f}")
    if epochs < 50:
        full_cost = hourly_rate * base_hours
        print(f"- Note: This is a quick demo run. Full training (200 epochs) would cost ~${full_cost:.2f}")

    return job_name, estimator


def get_instance_price(instance_type):
    """Get approximate hourly price for instance type"""
    prices = {
        'ml.g4dn.xlarge': 0.736,   # T4 GPU (what we used for small model)
        'ml.g5.xlarge': 1.206,     # A10G single GPU
        'ml.g5.2xlarge': 2.412,    # A10G x2 GPUs (current choice for large model!)
        'ml.p3.2xlarge': 3.825,    # V100 GPU
        'ml.p4d.24xlarge': 40.96,  # A100 GPU
    }
    return prices.get(instance_type, 'Unknown')


def create_sagemaker_training_script():
    """Create the training script that runs in SageMaker container"""
    script_content = '''
#!/usr/bin/env python3
"""
YOLO Training Script for SageMaker Container
"""

import os
import sys
import argparse

def main():
    print("Room Detection YOLO Training in SageMaker")
    print("=" * 50)

    # Install ultralytics if not available
    try:
        import ultralytics
        print("Ultralytics already installed")
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "ultralytics>=8.1.0", "torch", "torchvision", "torchaudio",
            "--quiet", "--no-cache-dir"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Installation failed: {result.stderr}")
            return

        print("Ultralytics installed successfully")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)  # Alternative format
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--weights', type=str, default='yolov8s.pt')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--save_period', type=int, default=10)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=float, default=3.0)
    parser.add_argument('--box', type=float, default=7.5)
    parser.add_argument('--cls', type=float, default=0.5)
    parser.add_argument('--dfl', type=float, default=1.5)
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment'])
    args = parser.parse_args()

    print(f"Training config: task={args.task}, {args.epochs} epochs, batch {args.batch_size}, size {args.imgsz}")

    # Import after installation
    from ultralytics import YOLO, settings
    import torch

    # Configure Ultralytics to not download datasets and use our paths
    settings.update({'datasets_dir': '/opt/ml/input/data/training'})
    print(f"Ultralytics settings updated: datasets_dir = {settings['datasets_dir']}")

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print(f"\\nLoading YOLO {args.task} model: {args.weights}...")
    model = YOLO(args.weights)

    # Update data path for SageMaker
    data_path = '/opt/ml/input/data/training/data.yaml'
    if os.path.exists(data_path):
        args.data = data_path
        print(f"Using SageMaker data: {data_path}")
        # Verify the data.yaml can be loaded
        try:
            import yaml
            with open(data_path, 'r') as f:
                data_config = yaml.safe_load(f)
            print(f"Data config loaded: path={data_config.get('path')}, train={data_config.get('train')}, val={data_config.get('val')}")
        except Exception as e:
            print(f"Warning: Could not verify data.yaml: {e}")
    else:
        print("Warning: SageMaker data not found, using local data.yaml")

    # Training
    print("\\nStarting training...")
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': device,
        'workers': args.workers,
        'project': '/opt/ml/model',
        'name': f'room_{args.task}',
        'save': True,
        'save_period': args.save_period,
        'patience': args.patience,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'box': args.box,
        'cls': args.cls,
        'dfl': args.dfl,
    }
    
    # Note: Segmentation loss weights are handled internally by ultralytics
    print(f"\\nTraining parameters:")
    for k, v in train_params.items():
        print(f"  {k}: {v}")
    
    results = model.train(**train_params)

    # Save final model
    model_path = '/opt/ml/model/final_model.pt'
    model.save(model_path)
    print(f"\\n[OK] Model saved to {model_path}")

    # Export for inference
    onnx_path = '/opt/ml/model/model.onnx'
    model.export(format='onnx', imgsz=args.imgsz)
    print(f"[OK] ONNX model exported to {onnx_path}")

    # Run validation
    print("\\nRunning final validation...")
    metrics = model.val()
    print(f"Validation mAP: {metrics.box.map:.4f}")

    print("\\nSageMaker training completed!")

if __name__ == "__main__":
    main()
'''

    with open('train_yolo_sagemaker.py', 'w') as f:
        f.write(script_content)

    print("Created SageMaker training script")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='AWS SageMaker YOLO Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Image Size Recommendations (based on CubiCasa5K analysis):
  640:  Fast, good baseline (YOLO default)
  800:  RECOMMENDED - balanced speed/quality for CubiCasa5K
  1024: High detail, slower training (only 13% of images are this large)
  1280: Maximum detail, very slow (only 9% of images are this large)

CubiCasa5K Stats: Mean 972x866px, Median 775x698px
        """
    )
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment'],
                       help='Task type: detect (bbox) or segment (polygons)')
    parser.add_argument('--model-size', type=str, default='l', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: l)')
    parser.add_argument('--imgsz', type=int, default=800,
                       help='Training image size in pixels (default: 800, based on dataset analysis)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override auto-detected data directory')
    parser.add_argument('--s3-prefix', type=str, default=None,
                       help='Override S3 prefix (default: auto-detect based on task and dataset)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override training epochs (default: 200 for full training, use 20 for quick demo)')
    args = parser.parse_args()
    
    print("AWS SageMaker YOLO Training Setup")
    print("=" * 60)
    print(f"Task: {args.task.upper()}")
    print(f"Model: YOLOv8-{args.model_size.upper()} (COCO pretrained)")
    print(f"Image size: {args.imgsz}x{args.imgsz}px")
    print("=" * 60)

    # Check AWS credentials and permissions
    print("\nChecking AWS credentials...")
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"[OK] AWS Account: {account_id}")

        # Test required services
        sm = boto3.client('sagemaker')
        s3 = boto3.client('s3')
        iam = boto3.client('iam')
        print("[OK] All required AWS services accessible")

    except Exception as e:
        print(f"[ERROR] AWS access issue: {e}")
        print("\n[SETUP] Setup Options:")
        print("1. AWS CLI: Run 'aws configure' with your access keys")
        print("2. Environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret")
        print("   export AWS_DEFAULT_REGION=us-east-1")
        print("3. IAM role (if on EC2/SageMaker)")
        print("4. ~/.aws/credentials file")
        sys.exit(1)

    # Create training script
    create_sagemaker_training_script()

    # Launch training
    job_name, estimator = train_on_sagemaker(
        task=args.task,
        local_data_dir=args.data_dir,
        model_size=args.model_size,
        imgsz=args.imgsz,
        s3_prefix=args.s3_prefix
    )

    print("\nNext steps:")
    print("1. Monitor training at the SageMaker console URL above")
    if args.task == 'detect':
        time_est = 8.0 * (args.imgsz / 640) ** 2  # Scale with resolution
        print(f"2. Detection training: ~{time_est:.1f} hours at {args.imgsz}px")
    else:
        time_est = 10.0 * (args.imgsz / 640) ** 2
        print(f"2. Segmentation training: ~{time_est:.1f} hours at {args.imgsz}px")
    print("3. Download trained model from S3 when complete")
    print("4. Test model with comprehensive_test.py")
    print("\nUsage examples:")
    print("  python sagemaker_train.py --task detect --model-size l --imgsz 800")
    print("  python sagemaker_train.py --task segment --model-size l --imgsz 1024")
