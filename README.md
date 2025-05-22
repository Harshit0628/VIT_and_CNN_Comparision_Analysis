# CNN vs Vision Transformer Comparison

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive comparison between Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for binary image classification using the Human vs Non-Human dataset from Kaggle.

## 🎯 Project Overview

This project implements and compares two state-of-the-art computer vision architectures:
- **ResNet-50** (Convolutional Neural Network)
- **ViT-B/16** (Vision Transformer)

Both models are pretrained on ImageNet and fine-tuned for binary classification to distinguish between humans and non-humans in images.

## 📊 Key Results

| Model | Test Accuracy | F1 Score | Precision | Recall | Training Time | ROC-AUC |
|-------|---------------|----------|-----------|--------|---------------|---------|
| ResNet-50 | 99.96% | 99.96% | 99.92% | 100.00% | 292.14s | 1.0000 |
| ViT-B/16 | 99.96% | 99.96% | 99.92% | 100.00% | 354.02s | 1.0000 |

## 🚀 Features

- **Multi-GPU Support**: Automatic detection and utilization of multiple GPUs using DataParallel
- **Mixed Precision Training**: Accelerated training with Automatic Mixed Precision (AMP)
- **Advanced Data Augmentation**: Comprehensive augmentation pipeline for improved generalization
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC, and Average Precision
- **Rich Visualizations**: Training curves, confusion matrices, ROC curves, and feature space visualization
- **Model Comparison**: Side-by-side performance analysis with detailed metrics

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cnn-vs-vit-comparison.git
   cd cnn-vs-vit-comparison
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   ```bash
   # Using Kaggle API (requires kaggle.json in ~/.kaggle/)
   kaggle datasets download -d aliasgartaksali/human-and-non-human
   unzip human-and-non-human.zip -d data/
   ```

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
Pillow>=8.3.0
```

## 🏃‍♂️ Usage

### Quick Start

```python
from main import main

# Run the complete comparison
main()
```

### Custom Training

```python
from models.resnet_model import ResNetModel
from models.vit_model import ViTModel
from utils.training import train_model
from utils.data_loader import load_data

# Load data
train_loader, val_loader, test_loader, class_names = load_data()

# Initialize models
cnn_model = ResNetModel(num_classes=2)
vit_model = ViTModel(num_classes=2)

# Train models
cnn_model, cnn_history = train_model(cnn_model, train_loader, val_loader, "ResNet-50")
vit_model, vit_history = train_model(vit_model, train_loader, val_loader, "ViT-B/16")
```

### Configuration

Key hyperparameters can be modified in the main script:

```python
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = 224
```

## 🎨 Visualizations

The project generates comprehensive visualizations including:

1. **Training History**: Loss and accuracy curves over epochs
2. **Confusion Matrices**: Detailed classification performance
3. **ROC Curves**: Receiver Operating Characteristic analysis
4. **Precision-Recall Curves**: Precision vs Recall trade-offs
5. **Sample Predictions**: Visual inspection of model predictions
6. **Feature Space Visualization**: t-SNE plots of learned representations
7. **Performance Comparison**: Side-by-side metric comparisons

## 🧪 Experimental Setup

### Data Augmentation
- Random cropping and resizing
- Horizontal flipping (50% probability)
- Random rotation (±15 degrees)
- Color jittering (brightness and contrast)
- ImageNet normalization

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Mixed Precision**: Enabled for faster training
- **Multi-GPU**: Automatic DataParallel scaling

### Early Stopping
- Patience: 5 epochs
- Metric: Validation loss
- Mode: Minimize

## 📈 Performance Analysis

### Model Comparison

Both models achieved exceptional performance on this dataset:

- **Accuracy**: 99.96% for both models
- **Training Speed**: ResNet-50 was ~21% faster (292s vs 354s)
- **Memory Efficiency**: ResNet-50 used less GPU memory
- **Convergence**: Both models converged within 8 epochs

### Key Insights

1. **Dataset Suitability**: The human vs non-human classification task is well-suited for both architectures
2. **Transfer Learning**: Both pretrained models adapted excellently to the specific task
3. **Efficiency**: ResNet-50 demonstrated superior computational efficiency
4. **Robustness**: Both models showed excellent generalization capabilities

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   BATCH_SIZE = 16  # or 8
   ```

2. **Slow Training**:
   ```python
   # Ensure CUDA is available
   print(torch.cuda.is_available())
   # Use DataLoader with num_workers
   num_workers = 4
   ```

3. **Import Errors**:
   ```bash
   # Reinstall packages
   pip install --upgrade torch torchvision timm
   ```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 📚 References

1. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
3. Wightman, R. (2019). PyTorch Image Models. GitHub repository.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Human and Non-Human Dataset](https://www.kaggle.com/datasets/aliasgartaksali/human-and-non-human) by Alias Gartaksali
- **PyTorch Team**: For the excellent deep learning framework
- **Timm Library**: For the pretrained Vision Transformer models
- **Kaggle**: For hosting the dataset and providing compute resources

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🎯 Future Work

- [ ] Implement additional architectures (EfficientNet, ConvNeXt)
- [ ] Add support for multi-class classification
- [ ] Implement gradient-based visualization techniques
- [ ] Add model interpretability analysis
- [ ] Optimize for mobile deployment
- [ ] Add real-time inference capabilities

---

⭐ **Star this repository if you found it helpful!**

![Visitors](https://visitor-badge.glitch.me/badge?page_id=yourusername.cnn-vs-vit-comparison)
