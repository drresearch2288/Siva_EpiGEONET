.PHONY: setup lint test eda train evaluate figures reproduce

setup:
	@echo "Setting up environment..."

lint:
	@echo "Running linter..."

test:
	@echo "Running tests..."

eda:
	@echo "Running EDA..."

train:
	@echo "Training model..."

evaluate:
	@echo "Evaluating model..."

figures:
	@echo "Generating figures..."

reproduce:
	@echo "Reproducing results..."
	python scripts/reproduce_all.py
