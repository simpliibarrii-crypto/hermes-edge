# Hermes Edge — Build and Dev Commands

.PHONY: install lint test clean build convert-270m convert-500m convert-1b help

install:  ## Install dev dependencies
	pip install -e . ai-edge-torch litert-lm torch sentencepiece pytest ruff
	pre-commit install 2>/dev/null || true

lint:  ## Run linter
	ruff check hermes/ scripts/ --ignore=E501

test:  ## Run tests
	pytest tests/ -v

clean:  ## Clean build artifacts
	rm -rf dist/ build/ checkpoints/ tokenizer/ *.litertlm *.tflite __pycache__/
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

convert-270m:  ## Convert 270M preset for iPhone 16 (ANE)
	python scripts/convert_to_litertlm.py \
		--checkpoint checkpoints/hermes-270m.pt \
		--tokenizer tokenizer/hermes.model \
		--preset hermes-270m \
		--backend apple \
		--multi-sig \
		--output dist/hermes-mobile-270m-int4.litertlm

convert-500m:  ## Convert 500M preset for iPhone 16 (ANE)
	python scripts/convert_to_litertlm.py \
		--checkpoint checkpoints/hermes-500m.pt \
		--tokenizer tokenizer/hermes.model \
		--preset hermes-500m \
		--backend apple \
		--multi-sig \
		--output dist/hermes-mobile-500m-int4.litertlm

convert-1b:  ## Convert 1B preset for iPhone 16 Pro (ANE)
	python scripts/convert_to_litertlm.py \
		--checkpoint checkpoints/hermes-1b.pt \
		--tokenizer tokenizer/hermes.model \
		--preset hermes-1b \
		--backend apple \
		--multi-sig \
		--output dist/hermes-mobile-1b-int4.litertlm

distill:  ## Distill from Gemma 3 1B (DeepSeek-style)
	python scripts/distill_from_gemma.py \
		--teacher google/gemma-3-1b \
		--student-preset hermes-distilled-1b \
		--data data/agentic_sft.jsonl \
		--output checkpoints/hermes-distilled-1b.pt \
		--temperature 3.0 --alpha 0.7

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
