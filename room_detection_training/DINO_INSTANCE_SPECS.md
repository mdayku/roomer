# DINO SageMaker Instance Specifications

## Selected Instance: `ml.g5.2xlarge` ✅

### Hardware Specs
- **GPUs:** 2x NVIDIA A10G (24GB VRAM each)
- **Total VRAM:** 48GB
- **vCPUs:** 8
- **RAM:** 32GB
- **Network:** Up to 10 Gbps
- **CUDA:** Supported (for custom ops compilation)

### Why This Instance?

✅ **Cost-Effective:** $2.41/hour vs $40.96/hour for A100
✅ **Sufficient VRAM:** 48GB total handles DINO with batch size 2
✅ **CUDA Support:** Deep Learning Container (DLC) has nvcc + gcc for compiling DINO custom ops
✅ **Multi-GPU:** 2 GPUs = faster training with PyTorch DDP
✅ **Proven:** Same GPU family (Ampere) as A100, just consumer-grade

### Instance Comparison

| Instance | GPU | VRAM | Cost/hr | 5-epoch | 50-epoch | Verdict |
|----------|-----|------|---------|---------|----------|---------|
| **ml.g5.2xlarge** | 2x A10G | 48GB | $2.41 | **$3.62** | **$36.15** | ✅ **Best choice** |
| ml.g5.xlarge | 1x A10G | 24GB | $1.21 | $1.82 | $18.15 | ⚠️ May be tight |
| ml.p3.2xlarge | 1x V100 | 16GB | $3.83 | $5.75 | $57.45 | ❌ Less VRAM, more $ |
| ml.p4d.24xlarge | 8x A100 | 320GB | $40.96 | $61.44 | $614.40 | ❌ Overkill + expensive |

### Training Configuration (Optimized for A10G)

```python
HYPERPARAMETERS = {
    'epochs': 5,           # Test run (change to 50 for full)
    'batch_size': 2,       # Per GPU (total effective = 4)
    'lr': 0.0001,          # Transformer learning rate
    'lr_backbone': 0.00001,# Backbone learning rate
    'num_classes': 2,      # Wall + room
    'hidden_dim': 256,     # Can reduce to 192 if OOM
    'num_queries': 300,    # Object queries
}
```

### Multi-GPU Configuration

```python
distribution={
    'pytorch': {
        'enabled': True,
        'processes_per_host': 2  # 2 GPUs on g5.2xlarge
    }
}
```

### Deep Learning Container (DLC)

**Framework:** PyTorch 2.0.0 with Python 3.10

**Pre-installed:**
- ✅ PyTorch + torchvision
- ✅ CUDA Toolkit (nvcc compiler)
- ✅ GCC/G++ (for C++ compilation)
- ✅ Common ML libraries (numpy, opencv, etc.)

**Will be installed in container:**
- DINO custom CUDA operators (MultiScaleDeformableAttention)
- transformers
- pycocotools
- Other DINO requirements

### Expected Performance

**5-epoch test run:**
- Duration: ~1.5 hours
- Cost: ~$3.62
- Purpose: Verify setup works, catch errors early

**50-epoch full training:**
- Duration: ~12-15 hours
- Cost: ~$29-36
- Purpose: Production model

### Memory Usage Estimates

| Component | VRAM Usage |
|-----------|------------|
| Model (DINO-4scale) | ~8GB per GPU |
| Batch (size=2) | ~4GB per GPU |
| Optimizer states | ~2GB per GPU |
| Gradient buffers | ~2GB per GPU |
| **Total per GPU** | **~16GB / 24GB** |
| **Safety margin** | **~33% free** ✅ |

### Fallback Options

**If OOM (Out of Memory) occurs:**

1. **Reduce batch size** (2 → 1)
   ```python
   'batch_size': 1,  # Half memory usage
   ```

2. **Reduce hidden_dim** (256 → 192)
   ```python
   'hidden_dim': 192,  # Smaller model
   ```

3. **Reduce num_queries** (300 → 200)
   ```python
   'num_queries': 200,  # Fewer object queries
   ```

4. **Gradient accumulation** (simulate larger batch)
   ```python
   'accumulate_grad_batches': 4,  # Effective batch = 4
   ```

### Dependencies Check

**The training script will:**
1. ✅ Clone DINO repo from GitHub
2. ✅ Install requirements (torch, transformers, etc.)
3. ✅ Compile custom CUDA operators
4. ✅ Verify COCO dataset structure
5. ✅ Launch training with multi-GPU

**All automated - no manual setup needed!**

### Monitoring

**CloudWatch Logs will show:**
```
[OK] torch imported successfully
[OK] DINO repository cloned successfully
[OK] DINO requirements installed
[OK] DINO ops compiled successfully
[OK] Found COCO dataset: train/val/test
[OK] Starting DINO training with config
Epoch [1/5] ...
```

### Summary

✅ **Instance:** ml.g5.2xlarge (2x A10G, 48GB VRAM)
✅ **Cost:** $2.41/hr (~$3.62 for test, ~$36 for full)
✅ **Specs:** Sufficient for DINO with batch size 2
✅ **CUDA:** DLC has all tools for custom ops compilation
✅ **Multi-GPU:** Configured for 2-GPU training
✅ **Ready:** Dataset prepared, scripts updated

**You're good to launch once segmentation training finishes!** 🚀

