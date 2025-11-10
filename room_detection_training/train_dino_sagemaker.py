#!/usr/bin/env python3
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
        "numpy<1.24",  # CRITICAL: Pin numpy to avoid np.float deprecation error in pycocotools
        "pycocotools",
        "matplotlib",
        "tqdm",
        "yapf<0.40"  # Pin yapf to avoid "verify" parameter error in DINO's slconfig.py
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
    
    # Compile CUDA operators BEFORE importing DINO
    print("\n[INFO] Compiling DINO's custom CUDA operators...")
    ops_dir = dino_dir / "models" / "dino" / "ops"
    if ops_dir.exists():
        original_cwd = os.getcwd()
        try:
            os.chdir(str(ops_dir))
            print(f"[INFO] Changed to ops directory: {ops_dir}")
            
            # Run the compilation script
            compile_script = ops_dir / "make.sh"
            if compile_script.exists():
                result = subprocess.run(
                    ["bash", str(compile_script)],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    print("[OK] CUDA ops compiled successfully via make.sh")
                else:
                    print(f"[WARN] make.sh failed, trying setup.py...")
                    print(f"  stderr: {result.stderr[-300:]}")
            
            # Fallback: try setup.py directly
            setup_script = ops_dir / "setup.py"
            if setup_script.exists():
                result = subprocess.run([
                    sys.executable, str(setup_script), "build_ext", "--inplace"
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    print("[OK] CUDA ops compiled successfully via setup.py")
                else:
                    print(f"[ERROR] Failed to compile CUDA ops:")
                    print(f"  stdout: {result.stdout[-500:]}")
                    print(f"  stderr: {result.stderr[-500:]}")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    sys.exit(1)
            else:
                print(f"[ERROR] Neither make.sh nor setup.py found in {ops_dir}")
                sys.stdout.flush()
                sys.stderr.flush()
                sys.exit(1)
        finally:
            os.chdir(original_cwd)
            print(f"[INFO] Changed back to: {original_cwd}")
        
        # Add ops directory to Python path so compiled CUDA ops can be imported
        sys.path.insert(0, str(ops_dir))
        print(f"[INFO] Added ops directory to Python path: {ops_dir}")
    else:
        print(f"[ERROR] CUDA ops directory not found: {ops_dir}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
    
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
        print(f"[DEBUG] DINO directory: {dino_dir}")
        print(f"[DEBUG] DINO directory exists: {dino_dir.exists()}")
        if dino_dir.exists():
            print(f"[DEBUG] DINO directory contents: {list(dino_dir.iterdir())[:10]}")
        
        # Change to DINO directory for import
        original_cwd = os.getcwd()
        os.chdir(str(dino_dir))
        print(f"[DEBUG] Changed to DINO directory: {os.getcwd()}")
        
        from main import main as dino_main
        from main import get_args_parser
        
        # Change back
        os.chdir(original_cwd)
        
        print("[OK] DINO main module imported successfully")
        
        # Check which config file exists FIRST (before parsing args)
        custom_config_path = Path('/opt/ml/code/DINO/config/DINO/DINO_4scale_room_detection.py')
        base_config_path = dino_dir / 'config' / 'DINO' / 'DINO_4scale.py'
        
        config_file = None
        config_options = []
        
        if custom_config_path.exists():
            config_file = str(custom_config_path)
            print(f"[OK] Using custom room detection config from source")
        elif (dino_dir / 'config' / 'DINO' / 'DINO_4scale_room_detection.py').exists():
            config_file = str(dino_dir / 'config' / 'DINO' / 'DINO_4scale_room_detection.py')
            print(f"[OK] Using custom room detection config from cloned repo")
        elif base_config_path.exists():
            # Fallback: use base config with options override
            config_file = str(base_config_path)
            config_options = [
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
        
        # Build command line args for DINO's parser (MUST include --config_file!)
        dino_cmd_args = [
            '--config_file', config_file,
            '--coco_path', str(data_root),
            '--output_dir', str(output_dir),
            '--dataset_file', 'coco',
        ]
        
        # CRITICAL: Always pass num_classes via --options to override config defaults
        # DINO doesn't accept --num_classes directly, must use --options
        override_options = [
            f'num_classes={args.num_classes}',
            f'dn_labelbook_size={args.num_classes}',  # Must match num_classes
        ]
        
        # Add base config options if using base config
        if config_options:
            override_options.extend(config_options)
        
        # Add all options to command
        if override_options:
            dino_cmd_args.extend(['--options'] + override_options)
        
        # Create args object compatible with DINO's main.py
        dino_parser = get_args_parser()
        
        # Parse with actual arguments (not empty list!)
        dino_args = dino_parser.parse_args(dino_cmd_args)
        
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
        print(f"  Config file: {config_file}")
        print(f"  COCO path: {dino_args.coco_path}")
        print(f"  Output dir: {dino_args.output_dir}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Num classes: {args.num_classes}")
        print(f"  World size (GPUs): {dino_args.world_size}")
        if config_options:
            print(f"  Config options: {', '.join(config_options[:3])}...")
        
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
