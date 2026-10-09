.PHONY: setup lint format test data-check train train-fast eval demo clean security-check

PYTHON ?= python
VENV_PYTHON ?= .venv/bin/python

ifeq ($(OS),Windows_NT)
    VENV_PYTHON = .venv/Scripts/python.exe
endif

setup:
	@echo "Setting up development environment..."
	uv pip install --native-tls -e ".[dev]"

lint:
	@echo "Running ruff and mypy..."
	$(VENV_PYTHON) -m ruff check src tests scripts
	$(VENV_PYTHON) -m ruff format --check src tests scripts
	$(VENV_PYTHON) -m mypy src

format:
	@echo "Formatting code with ruff..."
	$(VENV_PYTHON) -m ruff check --fix src tests scripts
	$(VENV_PYTHON) -m ruff format src tests scripts

test:
	@echo "Running test suite..."
	$(VENV_PYTHON) -m pytest tests

data-check:
	@echo "Validating dataset integrity and splits..."
	$(VENV_PYTHON) scripts/validate_data.py
	$(VENV_PYTHON) -m pytest tests/unit/test_dataset.py

train:
	@echo "Training avatar_cnn with ArcFace loss..."
	$(VENV_PYTHON) -m fras.training.train --config config/train.yaml

train-fast:
	@echo "Running fast smoke training for CI..."
	$(VENV_PYTHON) -m fras.training.train --config config/train.yaml --fast

eval:
	@echo "Evaluating open-set performance, robustness, and replay..."
	$(VENV_PYTHON) scripts/evaluate_all.py
	$(VENV_PYTHON) scripts/render_readme_results.py

security-check:
	@echo "Running bandit security audit..."
	$(VENV_PYTHON) -m bandit -r src/ -ll

demo:
	@echo "Seeding database and launching demo..."
	$(VENV_PYTHON) scripts/seed_database.py
	$(VENV_PYTHON) -m streamlit run src/fras/app/Home.py

clean:
	@echo "Cleaning up caches and build artifacts..."
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist src/*.egg-info
