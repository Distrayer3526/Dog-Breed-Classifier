# 🐶 Dog Breed Classifier (PyTorch)

This project is a deep-learning pipeline for classifying **dog breeds** using transfer learning in PyTorch.  
We fine-tuned multiple pretrained models, experimented with data pipelines, optimizers, schedulers, and training loops, and analyzed failure cases when accuracy plateaued.

---

## 📂 Dataset

We used the **Stanford Dogs Dataset**, which contains high-resolution images (not CIFAR–style).  
The dataset consists of:
- **120 dog breeds**
- **20,580 total images**
- Cropped, ImageNet-style natural images

Each image has significant variation in lighting, background, and pose, making the task harder than small datasets like CIFAR-10.

---

## 🧠 Model Architecture

We experimented with transfer learning using:

### **ResNet-50 (primary model)**
- Pretrained on ImageNet
- Replaced final fully-connected layer with output size = number of dog breeds
- Used **CrossEntropyLoss**

### Other architectural modifications tested:
- Added dropout to reduce overfitting
- Tried different learning rates and schedulers
- Attempted training deeper models (e.g., ResNet-101)  

---

## 🔧 Training Pipeline

### **Transforms**
```python
transforms = {
    "train": T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]),
    "val": T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]),
}
```
## 🔧 Optimization

We experimented with several optimizers, learning rates, and schedulers to improve convergence and stability.

### **Optimizers Tested**
- **Adam** — stable and easy to tune.  
- **AdamW** — decouples weight decay for better generalization.  
- **SGD + Momentum** — sometimes yields stronger final accuracy for large-image tasks, but requires careful LR scheduling.

### **Learning Rates Explored**
- `1e-3` — often too high; can cause divergence or noisy training.  
- `3e-4` — a good middle ground for Adam/AdamW in many runs.  
- `1e-4` — more stable, slower but useful during fine-tuning.

### **Schedulers**
- **StepLR** — reduces LR by a factor at predefined epochs.  
- **CosineAnnealingLR** — smooth decay; produced stable learning curves in our experiments.  
- **CosineAnnealingWarmRestarts** — useful to escape local plateaus by periodically resetting LR.  
- **ReduceLROnPlateau** — adapts to validation metric, but can react slowly for volatile val metrics.

### **Loss & Regularization**
- **Loss:** `CrossEntropyLoss` (with optional `label_smoothing=0.1`).  
- **Regularization:** weight decay (`5e-4` — `1e-4` range), dropout in classifier head (0.3–0.5), and mixup/cutmix augmentation.

### **Practical Tips**
- Log the current LR each epoch (`optimizer.param_groups[0]['lr']`) to verify scheduler behavior.  
- Use `CosineAnnealingLR` or `CosineAnnealingWarmRestarts` with AdamW for stable convergence if training from scratch.  
- If using SGD+Momentum, start with a higher LR (e.g., `0.1`) and strong LR decay milestones (e.g., `[60, 120, 160]`) or cosine decay over many epochs.

---

## 🐕 Future Improvement Ideas

To move toward the ~80% target, consider the following:

- **Unfreeze More Layers:** progressively unfreeze deeper encoder layers to adapt pretrained features.  
- **Stronger / Structured Augmentations:** RandomResizedCrop, RandAugment/AutoAugment, RandomErasing, Mixup, CutMix.  
- **Model Exploration:** try ConvNeXt, EfficientNet, or ViT variants (these often outperform older ResNets on fine-grained, high-res datasets).  
- **Longer Training + Proper Schedule:** training for more epochs (100–200) with a robust scheduler often helps when training from scratch.  
- **Class-Balanced Sampling:** if class imbalance exists, use weighted sampling or re-weighted loss.  
- **Debug on a Small Subset:** verify pipeline and label mapping on 5–10 classes first; this isolates data/label issues quickly.

---

## ✨ Credits
- **Stanford Dogs Dataset** — high-resolution dog-breed images (derived from ImageNet).
- **PyTorch & Torchvision** — core deep learning framework and model utilities.
- **ImageNet pretrained models / papers** — architectural inspirations and training recipes.
- Thanks to all open-source authors whose implementations and ideas informed the experiments.
