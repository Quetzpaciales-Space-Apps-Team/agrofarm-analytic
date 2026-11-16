.PHONY: test lint format typecheck audit build check
IMAGE ?= $(notdir $(CURDIR))-service:local

test:
	uv run pytest --cov --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

audit:
	uv export --frozen --no-dev --no-emit-project -o .requirements.audit.txt -q
	uv run pip-audit -r .requirements.audit.txt --require-hashes --disable-pip
	rm -f .requirements.audit.txt

build:
	docker build -t $(IMAGE) .

check: lint typecheck test
