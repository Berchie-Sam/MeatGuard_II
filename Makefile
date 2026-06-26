
# MeatGuard II Automation Makefile
# Note: This uses a simple shell wrapper for Python calls

SHELL := powershell
PYTHON := python

# Default target
all: setup train distill compress export

setup:
	@echo "Setting up environment..."
	# In a real scenario, we would run pip install -r requirements.txt
	mkdir -p artifacts/models artifacts/plots

train:
	@echo "Running Supervised Training..."
	$(PYTHON) src/main.py --config configs/config.yaml --mode training

distill: train
	@echo "Running Knowledge Distillation..."
	$(PYTHON) src/main.py --config configs/config.yaml --mode distill

compress: distill
	@echo "Running Pruning and Quantization..."
	$(PYTHON) src/main.py --config configsLconfig.yaml --mode compress

export: compress
	@echo "Exporting to TFLite/ONNX..."
	$(PYTHON) src/main.py --config configs/config.yaml --mode export

test:
	@echo "Running parity and unit tests..."
	pytest tests/

clean:
	Remove-Item -Recurse -Force artifacts/

