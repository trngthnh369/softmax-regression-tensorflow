# TensorFlow Softmax Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive implementation of Softmax Regression using TensorFlow 2.x for multi-class classification. This project demonstrates both standard and custom training approaches, with a focus on the MNIST handwritten digit classification task.

## Features

- ✨ Clean, modular implementation of Softmax Regression
- 🔄 Custom training loop using GradientTape for fine-grained control
- 🎯 Automated data preprocessing and augmentation
- 📊 Extensive evaluation metrics and visualizations
- 🧪 Comprehensive test suite
- 📝 Detailed documentation and examples

## Requirements

- Python 3.8+
- TensorFlow 2.19+
- NumPy
- Matplotlib
- scikit-learn
- PyYAML

## Installation

1. Clone the repository:
```bash
git clone https://github.com/trngthnh369/tensorflow-softmax-regression.git
cd tensorflow-softmax-regression
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install in development mode:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage
```python
from src.model.softmax_regression import SoftmaxRegression
from src.data.data_preprocessing import load_and_preprocess_mnist
from src.training.trainer import ModelTrainer

# Load and preprocess data
(X_train, Y_train), (X_val, Y_val) = load_and_preprocess_mnist()

# Create and compile model
model = SoftmaxRegression(input_size=784, num_classes=10)
compiled_model = model.create_and_compile(
    optimizer='adam',
    learning_rate=0.01
)

# Train model
trainer = ModelTrainer(compiled_model)
history = trainer.train(
    X_train, Y_train,
    X_val, Y_val,
    epochs=10,
    batch_size=32
)

# Evaluate
metrics = trainer.evaluate(X_val, Y_val)
print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
```

### Custom Training Loop
```python
from src.training.custom_trainer import CustomTrainer

trainer = CustomTrainer(
    model,
    learning_rate=0.01,
    momentum=0.9
)

history = trainer.train(
    X_train, Y_train,
    X_val, Y_val,
    epochs=10,
    batch_size=32
)
```

## Project Structure

```
├── src/                    # Source code
│   ├── model/             # Model definitions
│   │   ├── softmax_regression.py
│   │   └── layers.py
│   ├── data/             # Data handling
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── training/         # Training logic
│   │   ├── trainer.py
│   │   └── custom_trainer.py
│   └── utils/           # Utilities
│       ├── metrics.py
│       └── visualization.py
├── tests/               # Unit tests
├── notebooks/          # Jupyter notebooks
├── examples/           # Usage examples
├── configs/            # Configuration files
└── docs/              # Documentation
```

## Performance

| Model Configuration | Accuracy | Training Time | Parameters |
|--------------------|----------|---------------|------------|
| Default (Adam) | 92.1% | 18s | 7,850 |
| Custom SGD | 91.8% | 22s | 7,850 |
| With Momentum | 92.3% | 20s | 7,850 |

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_model.py
```

### Running examples

```bash
python examples/basic_usage.py
python examples/custom_training_example.py
```

### Jupyter notebook

```bash
jupyter notebook notebooks/softmax_regression_demo.ipynb
```

### Documentation
```bash
# Generate documentation
sphinx-build -b html docs/source docs/build
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Versioning

We use [SemVer](http://semver.org/) for versioning. For available versions, see the [tags on this repository](https://github.com/yourusername/tensorflow-softmax-regression/tags).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for the excellent framework
- MNIST dataset creators
- All contributors and maintainers

## Contact

- **TruongThinh** [Email](truongthinhnguyen30303@gmail.com)

Project Link: [https://github.com/trngthnh369/tensorflow-softmax-regression](https://github.com/trngthnh369/tensorflow-softmax-regression)