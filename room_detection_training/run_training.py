"""
Launch script for Room Detection Training
Handles setup, training, and basic validation
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run command with proper error handling"""
    print(f"\n📋 {description}")
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=False,  # Show output live
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    print("🚀 Room Detection Training Launcher")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("config.py").exists():
        print("❌ Please run this script from the room_detection_training directory")
        sys.exit(1)

    # Step 1: Test environment
    print("\n🔍 Step 1: Testing Environment Setup")
    if not run_command("python test_environment.py", "Testing training environment"):
        print("❌ Environment test failed. Please fix issues and try again.")
        sys.exit(1)

    # Step 2: Confirm data exists
    print("\n📊 Step 2: Checking Training Data")
    data_dir = Path("../../room_detection_dataset_coco")
    if not data_dir.exists():
        print(f"❌ Training data not found: {data_dir}")
        print("Please run these commands first:")
        print("  pnpm --filter @app/server convert:coco")
        print("  pnpm --filter @app/server prepare-dataset")
        print("  pnpm --filter @app/server convert-to-coco")
        sys.exit(1)

    train_data = data_dir / "train" / "annotations.json"
    if not train_data.exists():
        print(f"❌ Training annotations not found: {train_data}")
        sys.exit(1)

    print(f"✅ Training data found: {train_data}")

    # Step 3: Ask for confirmation
    print("\n⚠️ Step 3: Training Confirmation")
    print("This will start Mask R-CNN training with the following settings:")
    print("- Model: Mask R-CNN with ResNet-50 backbone")
    print("- Dataset: COCO format with polygon segmentation")
    print("- Epochs: 20 (can be interrupted with Ctrl+C)")
    print("- Output: ./outputs/models/")

    response = input("\nDo you want to start training? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return

    # Step 4: Start training
    print("\n🎯 Step 4: Starting Training")
    print("Training will show progress. You can interrupt with Ctrl+C at any time.")
    print("Models will be saved to ./outputs/models/")
    print("-" * 60)

    try:
        run_command("python train.py", "Running Mask R-CNN training")
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("✅ Partial results saved to ./outputs/")

    # Step 5: Post-training validation
    print("\n🔍 Step 5: Post-Training Validation")

    model_path = Path("outputs/models/best_model.pth")
    if model_path.exists():
        print("✅ Best model found. Testing inference...")
        run_command("python inference.py", "Testing trained model inference")
    else:
        print("⚠️ No trained model found. Training may have been interrupted.")

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("📊 Training Session Summary")
    print("=" * 60)

    # Check outputs
    model_dir = Path("outputs/models")
    if model_dir.exists():
        models = list(model_dir.glob("*.pth"))
        print(f"📁 Models saved: {len(models)}")
        for model in sorted(models):
            print(f"   - {model.name}")

    log_file = Path("outputs/logs/training_log.json")
    if log_file.exists():
        print(f"📋 Training log: {log_file}")

    vis_dir = Path("outputs/visualizations")
    if vis_dir.exists():
        vis_files = list(vis_dir.glob("*.png"))
        print(f"🖼️ Visualizations: {len(vis_files)} files")

    print("\n🎉 Training session completed!")
    print("\nNext steps:")
    print("1. Review training logs: outputs/logs/training_log.json")
    print("2. Test model: python inference.py")
    print("3. Visualize results: check outputs/visualizations/")
    print("4. Deploy model for production use")

if __name__ == "__main__":
    main()
