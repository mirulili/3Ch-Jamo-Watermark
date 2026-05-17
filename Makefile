# Makefile for the Jamo Watermark project

PYTHON = python
CONDA = conda
ENV_FILE = environment.yml

.PHONY: all setup install run test_robustness corpus_analysis length_sweep clean

all: install

setup:
	@echo "Creating conda environment..."
	$(CONDA) env create --file $(ENV_FILE)

install:
	@echo "Updating conda environment..."
	$(CONDA) env update --file $(ENV_FILE) --prune

run:
	@echo "Running the main application..."
	$(PYTHON) -m src.main

test_robustness:
	@echo "Running robustness evaluation..."
	$(PYTHON) -m src.evaluation.eval_robustness

corpus_analysis:
	@echo "Running vocabulary corpus analysis..."
	$(PYTHON) -m src.evaluation.corpus_analysis

length_sweep:
	@echo "Running message-length sweep..."
	$(PYTHON) -m src.evaluation.length_sweep

clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
