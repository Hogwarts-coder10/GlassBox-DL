# Contributing to GlassBoxDL

First of all, thank you for your interest in contributing to **GlassBoxDL**!

Whether you're fixing a bug, improving documentation, writing tests, or implementing a new feature, every contribution helps make GlassBoxDL a better framework for the community.

Please take a few minutes to read this guide before contributing.

> **Status:** GlassBoxDL is in early development — the core framework (tensors, autograd, layers, etc.) is still being built out. See the [README](./README.md) for the current roadmap before planning larger contributions.

---

# Our Philosophy

GlassBoxDL is built around three core principles:

* **Transparency** – Every component should be understandable and easy to inspect.
* **Maintainability** – Clean, modular, and well-documented code is preferred over clever code.
* **Learning** – The framework is designed to help others understand how deep learning frameworks work internally.

When contributing, keep these principles in mind.

---

# Ways to Contribute

You can contribute in many ways, including:

* Fixing bugs
* Improving documentation
* Writing unit tests
* Optimizing existing implementations
* Adding visualization utilities
* Improving benchmarks
* Reporting issues
* Suggesting new features

If you're unsure where to begin, look for issues labeled **good first issue** or **help wanted**.

---

# Development Workflow

All development follows a Pull Request-based workflow.

```
feature/*
      │
      ▼
Pull Request
      │
      ▼
develop
      │
      ▼
main
```

**Do not commit directly to `main`.**

All framework changes should be reviewed before merging.

---

# Branch Strategy

The repository uses the following branches:

| Branch      | Purpose                        |
| ----------- | ------------------------------ |
| `main`      | Stable releases                |
| `develop`   | Integration branch             |
| `feature/*` | Individual feature development |

Examples:

```
feature/tensor
feature/cnn
feature/optimizer-adam
feature/documentation
```

---

# Getting Started

## 1. Fork the repository

Click the **Fork** button on GitHub.

---

## 2. Clone your fork

```bash
git clone https://github.com/<your-username>/GlassBoxDL.git

cd GlassBoxDL
```

---

## 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it before installing dependencies.

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Create a feature branch

```bash
git checkout develop

git pull origin develop

git checkout -b feature/your-feature-name
```

---

# Coding Standards

Please follow these guidelines:

* Follow **PEP 8**
* Write readable and modular code
* Use meaningful variable and function names
* Keep functions focused on a single responsibility
* Add comments only where necessary
* Avoid duplicated code
* Prefer clarity over cleverness

---

# Project Structure

Please place files in their appropriate directories.

Examples:

| Component             | Directory                    |
| ---------------------- | ----------------------------- |
| Core (Tensor, Module, Sequential) | `glassboxdl/core/`      |
| Activation Functions   | `glassboxdl/activations/`     |
| Layers                 | `glassboxdl/layers/`          |
| Loss Functions         | `glassboxdl/losses/`          |
| Metrics                | `glassboxdl/metrics/`         |
| Optimizers             | `glassboxdl/optimizers/`      |
| Models                 | `glassboxdl/models/`          |
| Preprocessing          | `glassboxdl/preprocessing/`   |
| Training Utilities     | `glassboxdl/training/`        |
| Visualization          | `glassboxdl/visualization/`   |
| Shared Helpers         | `glassboxdl/utils/`           |

Avoid introducing new top-level directories unless discussed first.

---

# Documentation

Every significant feature should include appropriate documentation.

Documentation may include:

* Docstrings
* Usage examples
* Tutorials
* API documentation
* Architecture notes

If a new public API is introduced, update the documentation accordingly.

---

# Testing

All new features should include corresponding unit tests whenever possible.

Run the test suite before opening a Pull Request:

```bash
pytest
```

Pull Requests that introduce failing tests may be requested to fix them before merging.

---

# Commit Messages

GlassBoxDL follows the **Conventional Commits** specification.

Examples:

```text
feat(core): implement Tensor class

feat(layers): add Conv2D layer

fix(training): correct gradient update

docs: improve README

refactor(metrics): simplify accuracy calculation

test(losses): add cross entropy tests

chore: update dependencies
```

Please write clear and descriptive commit messages.

---

# Pull Requests

Before submitting a Pull Request, ensure that:

* Your branch is up to date with `develop`
* The project builds successfully
* Tests pass
* Documentation has been updated (if required)
* Code follows the project's coding standards

Your Pull Request should include:

* A clear description of the changes
* The motivation behind the changes
* Any relevant issue references (if applicable)

---

# Reporting Issues

When reporting a bug, please include:

* Operating System
* Python Version
* Steps to reproduce
* Expected behavior
* Actual behavior
* Relevant logs or screenshots (if available)

Clear issue reports help us resolve problems more efficiently.

---

# Feature Requests

Feature requests are welcome.

When proposing a feature, please describe:

* The problem it solves
* The proposed solution
* Possible alternatives
* Any implementation ideas (optional)

Constructive discussion is encouraged before large features are implemented.

---

# Code Review

Every framework-related contribution is reviewed before merging.

Reviewers may request:

* Code improvements
* Additional tests
* Documentation updates
* Refactoring

These requests are intended to maintain the quality and consistency of the project.

---

# Community Guidelines

Please be respectful and constructive when interacting with other contributors.

Healthy discussions, questions, and feedback are encouraged.

---

# Thank You

Thank you for taking the time to contribute to GlassBoxDL.

Your contributions—whether big or small—help improve the project and make deep learning more transparent, accessible, and educational for everyone.
