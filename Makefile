# Makefile for the Jamo Watermark project

# Define the python interpreter
PYTHON = python3

.PHONY: all install run clean

all: install

# Install dependencies from requirements.txt
install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -r requirements.txt

# Run the main application
run:
	@echo "Running the main application..."
	$(PYTHON) -m src.main

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +