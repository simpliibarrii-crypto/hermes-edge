.PHONY: install lint test clean convert-270m convert-500m convert-1b distill help

VENV ?= .venv
PYTHON ?= python3

help:
	@echo "Hermes Edge Makefile"
	@echo "  install       - Install lightweight core + dev tools"
	@echo "  install-all   - Install full model/runtime/conversion stack"
	@echo "  lint          - Run ruff linter"
	@echo "  test          - Run pytest"
	@echo "  clean         - Remove dist/, checkpoints/, tokenizer/, *.litertlm"
	@echo "  convert-270m  - Convert Qwen2.5-0.5B to INT4 .litertlm (270M eq.)"
	@echo "  convert-500m  - Convert Qwen2.5-1.5B to INT4 .litertlm (500M eq.)"
	@echo "  convert-1b    - Convert Qwen3-0.6B to INT4 .litertlm (1B eq.)"
	@echo "  run           - Start HF Space demo locally"
	@echo "  upload        - Upload model to HuggingFace"
	@echo ""

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip setuptools wheel
	$(VENV)/bin/pip install -e ".[dev]"
	@echo "Done. Activate: source $(VENV)/bin/activate"

install-all:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip setuptools wheel
	$(VENV)/bin/pip install -e ".[all]"
	@echo "Done. Activate: source $(VENV)/bin/activate"

lint:
	$(VENV)/bin/ruff check hermes/ scripts/ tests/ space_app.py

test:
	$(VENV)/bin/pytest tests/ -v --tb=short

clean:
	rm -rf dist/ build/ checkpoints/ tokenizer/ *.litertlm .venv/ __pycache__/
	rm -rf hermes/__pycache__ tests/__pycache__ scripts/__pycache__
	find . -name "*.pyc" -delete

convert-270m:
	$(PYTHON) scripts/convert_hf_to_litertlm.py \
		--model_id Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir ./dist \
		--quantization dynamic_wi4_afp32 \
		--cache_length 2048 \
		--prefill_lengths 32 \
		--force
	@echo "270M model ready in dist/"

convert-500m:
	$(PYTHON) scripts/convert_hf_to_litertlm.py \
		--model_id Qwen/Qwen2.5-1.5B-Instruct \
		--output_dir ./dist \
		--quantization dynamic_wi4_afp32 \
		--cache_length 2048 \
		--prefill_lengths 32 \
		--force
	@echo "500M model ready in dist/"

convert-1b:
	$(PYTHON) scripts/convert_hf_to_litertlm.py \
		--model_id litert-community/Qwen3-0.6B \
		--output_dir ./dist \
		--quantization dynamic_wi4_afp32 \
		--cache_length 4096 \
		--prefill_lengths 32 \
		--force
	@echo "1B model ready in dist/"

distill:
	@echo "Distillation requires GPU. Run on cloud instance:"
	@echo "  python scripts/distill_from_gemma.py --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
	@echo ""

run:
	$(PYTHON) space_app.py

upload:
	@echo "Upload to HuggingFace:"
	@echo "  hf upload bclermo/hermes-edge dist/hermes-mobile-270m-int4.litertlm --repo-type model"
	@echo ""
