#!/usr/bin/env python3
"""
DINO Training Script for SageMaker Container
Based on DINO framework: https://github.com/IDEA-Research/DINO
"""

# CRITICAL: Print immediately to verify script is running
print("[SCRIPT START] DINO training script starting...")
import sys
sys.stdout.flush()

import os
import argparse
import json
from pathlib import Path

# Try importing torch (may not be installed yet)
try:
    import torch
    print("[OK] torch imported successfully")
except ImportError:
    torch = None
    print("[WARN] torch not available yet, will install dependencies first")

sys.stdout.flush()

def main():
    import sys
    import traceback
    import builtins
    
    # Redirect stderr to stdout so errors show up in CloudWatch
    sys.stderr = sys.stdout
    
    # Flush output immediately
    import functools
    original_print = builtins.print
    def flush_print(*args, **kwargs):
        original_print(*args, **kwargs, flush=True)
    print = flush_print
    
    print("Room Detection DINO Training in SageMaker")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # First 3 entries
    
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
    print("\n[INFO] Importing DINO main module...")
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
        from main import main as dino_main
        from main import get_args_parser
        
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
    import traceback
    import builtins
    
    # Ensure we can print even if main() fails early
    try:
        print("[SCRIPT] Entering main execution block")
        sys.stdout.flush()
    except:
        # If print fails, use builtin print directly
        builtins.print("[SCRIPT] Entering main execution block (using builtin print)")
        sys.stdout.flush()
    
    # Wrap everything in try-except to catch any early failures
    exit_code = 1
    try:
        main()
        exit_code = 0
    except KeyboardInterrupt:
        try:
            print("[INFO] Training interrupted by user")
            sys.stdout.flush()
        except:
            pass
        exit_code = 130
    except SystemExit as e:
        # Re-raise SystemExit but ensure output is flushed
        try:
            print(f"[INFO] SystemExit with code: {e.code}")
            sys.stdout.flush()
        except:
            pass
        raise
    except Exception as e:
        try:
            print(f"[FATAL ERROR] Script failed with exception: {e}")
            print(f"[FATAL ERROR] Exception type: {type(e).__name__}")
            print("[FATAL ERROR] Full traceback:")
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            # Last resort - write to stderr directly
            import sys as sys_module
            sys_module.stderr.write(f"FATAL ERROR: {e}\n")
            sys_module.stderr.write(f"Exception type: {type(e).__name__}\n")
            traceback.print_exc(file=sys_module.stderr)
            sys_module.stderr.flush()
        exit_code = 1
    
    # Ensure all output is flushed before exiting
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except:
        pass
    
    sys.exit(exit_code)
