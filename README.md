# GlassBoxDL

<p align="center">
  <strong>A Transparent Deep Learning Framework Built from Scratch</strong>
</p>

<p align="center">
  Build, understand, and visualize deep learning models using only <strong>NumPy</strong>, <strong>SciPy</strong>, and <strong>Matplotlib</strong>.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Folder Guide](#folder-guide)
- [Development Roadmap](#development-roadmap)
- [Examples](#examples)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [Branch Strategy](#branch-strategy)
- [Coding Standards](#coding-standards)
- [License](#license)
- [Team](#team)

---

## Overview

GlassBoxDL is an open-source deep learning framework being built entirely from scratch in Python using **NumPy**, **SciPy**, and **Matplotlib**.

Unlike traditional deep learning frameworks that often abstract away internal computations, GlassBoxDL is designed with **transparency**, **education**, and **extensibility** as its core principles. Every component will be implemented from first principles, making it easier to understand how modern deep learning frameworks work under the hood.

Whether you're a student learning neural networks, an educator teaching deep learning, or a developer curious about framework internals, GlassBoxDL aims to provide a clean and approachable implementation.

> **Status:** Early stage. The repository scaffolding, testing infrastructure, and development workflow are in place (see [Features](#features)); the core framework itself (tensors, autograd, layers, etc.) is under active development per the [roadmap](#development-roadmap) below.

---

## Project Goals

* Build a deep learning framework completely from scratch.
* Provide transparent implementations of deep learning algorithms.
* Encourage learning through readable and modular code.
* Maintain production-quality repository standards.
* Serve as both an educational resource and a reusable Python library.
* Publish the framework on PyPI.

---

## Features

### Current

* Clean repository architecture
* Modular project structure
* Production-style development workflow
* Unit testing support
* Documentation framework
* GitHub Actions ready

### Planned

* Tensor implementation
* Automatic differentiation (Autograd)
* Neural network modules
* Sequential API
* Dense (Linear) layers
* Activation functions
* Loss functions
* Optimizers
* CNN layers
* U-Net
* Attention U-Net
* Training utilities
* Metrics
* Visualization tools
* Model serialization
* Benchmark suite

---

## Technology Stack

* Python
* NumPy
* SciPy
* Matplotlib

Dependencies are intentionally minimal — no TensorFlow, no PyTorch. Every major framework component (tensors, autograd, layers, optimizers, etc.) is being implemented from scratch as it's added, rather than wrapping an existing library.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Hogwarts-coder10/GlassBoxDL.git

cd GlassBoxDL
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
GlassBoxDL/
│
├── glassboxdl/                      # Core Deep Learning Framework
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── tensor.py
│   │   ├── parameter.py
│   │   ├── module.py
│   │   ├── sequential.py
│   │   ├── initialization.py
│   │   └── serialization.py
│   │
│   ├── layers/
│   │   ├── linear.py
│   │   ├── convolution.py
│   │   ├── pooling.py
│   │   ├── batchnorm.py
│   │   ├── dropout.py
│   │   ├── flatten.py
│   │   └── upsampling.py
│   │
│   ├── activations/
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   ├── tanh.py
│   │   ├── softmax.py
│   │   ├── leakyrelu.py
│   │   └── gelu.py
│   │
│   ├── losses/
│   │   ├── mse.py
│   │   ├── binary_crossentropy.py
│   │   ├── categorical_crossentropy.py
│   │   ├── dice.py
│   │   ├── focal.py
│   │   └── dice_bce.py
│   │
│   ├── optimizers/
│   │   ├── optimizer.py
│   │   ├── sgd.py
│   │   ├── momentum.py
│   │   ├── rmsprop.py
│   │   └── adam.py
│   │
│   ├── metrics/
│   │   ├── accuracy.py
│   │   ├── precision.py
│   │   ├── recall.py
│   │   ├── f1score.py
│   │   ├── iou.py
│   │   ├── dice.py
│   │   └── confusion_matrix.py
│   │
│   ├── models/
│   │   ├── ann.py
│   │   ├── cnn.py
│   │   ├── unet.py
│   │   ├── attention_unet.py
│   │   ├── unetplusplus.py
│   │   └── resunet.py
│   │
│   ├── preprocessing/
│   │   ├── normalize.py
│   │   ├── resize.py
│   │   ├── augmentation.py
│   │   └── masks.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── checkpoint.py
│   │   └── callbacks.py
│   │
│   ├── visualization/
│   │   ├── loss_plot.py
│   │   ├── metrics_plot.py
│   │   ├── prediction_plot.py
│   │   └── segmentation_plot.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── random.py
│       ├── helpers.py
│       └── image.py
│
├── examples/
│   ├── xor/
│   ├── mnist/
│   ├── cifar10/
│   ├── image_classification/
│   └── image_segmentation/          # Semester project (see below)
│       ├── data/
│       │   ├── raw/
│       │   ├── processed/
│       │   ├── train/
│       │   ├── validation/
│       │   └── test/
│       ├── configs/
│       ├── experiments/
│       │   ├── unet/
│       │   ├── attention_unet/
│       │   ├── unetplusplus/
│       │   └── resunet/
│       ├── results/
│       │   ├── graphs/
│       │   ├── predictions/
│       │   ├── confusion_matrices/
│       │   └── comparison/
│       ├── report/
│       ├── reviews/
│       │   ├── review0/
│       │   ├── review1/
│       │   ├── review2/
│       │   └── review3/
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── README.md
│
├── benchmarks/
├── docs/
├── tests/
│
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

Each package under `glassboxdl/` is designed with modularity and maintainability in mind, allowing individual components to evolve independently.

---

## Folder Guide

### `glassboxdl/`
The framework itself. Everything inside is reusable and independent of any single project.

| Subfolder | Purpose |
|---|---|
| `core/` | Foundation of the framework — Tensor, Parameters, base Module, Sequential container, weight initialization, model serialization |
| `layers/` | Neural network layers — Linear, Convolution, Pooling, BatchNorm, Dropout, Upsampling |
| `activations/` | Activation functions — ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU |
| `losses/` | Loss functions — MSE, BCE, Categorical Cross-Entropy, Dice, Focal, Dice+BCE |
| `optimizers/` | Optimization algorithms — SGD, Momentum, RMSProp, Adam |
| `metrics/` | Evaluation metrics — Accuracy, Precision, Recall, F1, IoU, Dice Score, Confusion Matrix |
| `models/` | Complete architectures — ANN, CNN, U-Net, Attention U-Net, U-Net++, ResUNet |
| `preprocessing/` | Data prep — resizing, normalization, augmentation, mask generation |
| `training/` | Training loop, evaluation, checkpointing, callbacks |
| `visualization/` | Matplotlib-based plotting — loss/accuracy curves, prediction and segmentation overlays |
| `utils/` | Shared helpers — config loader, logger, random seed control, image helpers |

### `examples/`
Demonstrates how to use the framework: XOR, MNIST, CIFAR-10, image classification, and image segmentation. Applied project work lives here, not inside the framework itself.

### `examples/image_segmentation/`
The complete semester project — dataset, experiments (U-Net, Attention U-Net, U-Net++, ResUNet), results, reports, and review material for course evaluations.

### `benchmarks/`
Framework performance benchmarks — Conv2D speed, matrix multiplication, optimizer comparison, memory usage.

### `docs/`
Framework documentation — API reference, UML/architecture diagrams, developer guide.

### `tests/`
Unit tests for every framework component — layers, optimizers, losses, metrics, models.

---

## Development Roadmap

### Phase 1 — Foundation *(current)*

* Repository bootstrap
* Documentation
* Development workflow
* Testing infrastructure

### Phase 2

* Tensor
* Automatic differentiation
* Module system
* Sequential container

### Phase 3

* Linear layers
* Activation functions
* Loss functions
* Optimizers

### Phase 4

* Convolution
* Pooling
* CNN

### Phase 5

* U-Net
* Attention U-Net
* Visualization
* Performance improvements

---

## Examples

The `examples/` directory will contain practical implementations demonstrating how to use the framework, including:

* XOR Classification
* Image Segmentation *(the team's full semester project — see [Folder Guide](#folder-guide) for its internal structure)*
* CNN Image Classification
* Custom Training Loops

---

## Documentation

Project documentation lives inside the `docs/` directory and will cover:

* API Reference
* Architecture
* Tutorials
* Framework Design
* Diagrams

---

## Testing

Run the complete test suite using:

```bash
pytest
```

Tests are located inside the `tests/` directory.

---

## Contributing

Contributions are always welcome.

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before opening an issue or submitting a pull request. All framework changes are reviewed before being merged.

---

## Branch Strategy

```text
main       → Stable releases
develop    → Integration branch
feature/*  → Individual feature development
```

Workflow:

```text
feature/*  →  Pull Request  →  develop  →  main
```

Direct commits to the stable branch are not accepted.

---

## Coding Standards

GlassBoxDL follows:

* PEP 8
* Type hints where applicable
* Modular architecture
* Conventional Commits
* Unit tests for new features
* Code review before merging

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

GlassBoxDL is inspired by the educational value of understanding how deep learning frameworks operate internally, combined with modern software engineering and open-source development practices.

---

## Team

| Member | Responsibilities |
|---|---|
| **V SS Karthik** *(Maintainer)* | Core framework, architecture, code review, Git maintenance, final merges |
| **Sumera** | Dataset, preprocessing, augmentation, configs |
| **Riddhi** | Training, metrics, visualization, experiments, results |
| **Archit** | Documentation, examples, tests, benchmarks, reports, presentations |

As maintainer, Karthik is responsible for framework architecture, core implementation, API design, code reviews, and pull request approvals.

---

## Star the Repository

If you find GlassBoxDL useful, consider giving the repository a ⭐ — it helps others discover the project and supports future development.
