# Softmax Regression with TensorFlow

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án triển khai thuật toán **Softmax Regression** sử dụng TensorFlow 2.x để phân loại chữ số viết tay trên dataset MNIST.

## 🌟 Tính năng

- ✅ Triển khai Softmax Regression với TensorFlow/Keras
- ✅ Custom training loop sử dụng GradientTape
- ✅ Tiền xử lý dữ liệu tự động
- ✅ Đánh giá và visualization kết quả
- ✅ Cấu trúc code rõ ràng, dễ mở rộng
- ✅ Unit tests đầy đủ

## 📋 Yêu cầu

- Python 3.8+
- TensorFlow 2.19+
- NumPy
- Matplotlib
- PyYAML

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/trngthnh369/softmax-regression-tensorflow.git
cd softmax-regression-tensorflow
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cài đặt package (optional)

```bash
pip install -e .
```

## 📖 Sử dụng

### Basic Usage

```python
from src.model.softmax_regression import SoftmaxRegression
from src.data.data_preprocessing import load_and_preprocess_mnist
from src.training.trainer import ModelTrainer

# Load và preprocess dữ liệu
(X_train, Y_train), (X_val, Y_val) = load_and_preprocess_mnist()

# Tạo model
model = SoftmaxRegression(input_size=784, num_classes=10)
compiled_model = model.create_and_compile(learning_rate=0.01)

# Training
trainer = ModelTrainer(compiled_model)
history = trainer.train(X_train, Y_train, X_val, Y_val, epochs=10)

# Đánh giá
loss, accuracy = trainer.evaluate(X_val, Y_val)
print(f"Validation Accuracy: {accuracy:.4f}")
```

### Custom Training Loop

```python
from src.training.custom_trainer import CustomTrainer

# Sử dụng custom training loop
custom_trainer = CustomTrainer(model)
custom_trainer.train(X_train, Y_train, X_val, Y_val, epochs=10)
```

## 📊 Kết quả

Model đạt được độ chính xác **~90%** trên MNIST validation set sau 10 epochs:

- **Training Accuracy**: 89.4%
- **Validation Accuracy**: 90.0%
- **Training Time**: ~20 giây

## 📁 Cấu trúc dự án

```
├── src/                    # Mã nguồn chính
│   ├── model/             # Định nghĩa model
│   ├── data/              # Xử lý dữ liệu
│   ├── training/          # Logic training
│   └── utils/             # Tiện ích
├── notebooks/             # Jupyter notebooks
├── examples/              # Ví dụ sử dụng
├── tests/                 # Unit tests
└── configs/               # Cấu hình
```

## 🔧 Development

### Chạy tests

```bash
python -m pytest tests/ -v
```

### Chạy ví dụ

```bash
python examples/basic_usage.py
python examples/custom_training_example.py
```

### Jupyter notebook

```bash
jupyter notebook notebooks/softmax_regression_demo.ipynb
```

## 📈 Benchmark

| Method | Accuracy | Training Time | Parameters |
|--------|----------|---------------|------------|
| Standard Training | 90.0% | 20s | 7,850 |
| Custom Training | 89.8% | 25s | 7,850 |

## 🤝 Đóng góp

Chúng tôi chào đón mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

Dự án này được phân phối dưới MIT License. Xem [LICENSE](LICENSE) để biết thêm thông tin.

## 👨‍💻 Tác giả

- **Your Name** - *Initial work* - [Truong Thinh](https://github.com/trngthnh369)

## 🙏 Acknowledgments

- TensorFlow team cho framework tuyệt vời
- MNIST dataset creators
- Cộng đồng Machine Learning Việt Nam