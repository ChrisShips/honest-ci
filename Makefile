PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
# The acceptance target has to run in CI too, and CI installs the dependencies
# into the system Python, with no .venv.
PYTEST := $(if $(wildcard $(BIN)/pytest),$(BIN)/pytest,pytest)

.DEFAULT_GOAL := help

.PHONY: help venv install lint type test cov check acceptance clean

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	@test -d $(VENV) || $(PY) -m venv $(VENV)

install: venv ## Install the development dependencies (an editable install)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"

lint: ## Static checks with ruff
	$(BIN)/ruff check src tests bench

type: ## Type checks with mypy
	$(BIN)/mypy

test: ## Run the tests
	$(BIN)/pytest -q

# No --cov=honest_ci: which files are measured is decided by source in
# pyproject.toml's [tool.coverage.run], and a value on the command line would
# override it.
cov: ## Run the tests and check the coverage bar
	$(BIN)/pytest -q --cov --cov-report=term-missing --cov-fail-under=75

check: lint type cov ## The local gate: the same checks as CI

acceptance: ## Acceptance on a real machine: Docker + Maven run the three terminal states of examples/java-junit5
	@# Pull the image onto this machine before the run. Pulling during the run would
	@# hit the tool probe's 60-second timeout, which is truthfully reported as no mvn
	@# in the image: correct, but not the question this command means to answer.
	docker pull $${HONEST_CI_JAVA_DOCKER_IMAGE:-maven:3.9-eclipse-temurin-21}
	$(PYTEST) -m docker -v -p no:cacheprovider tests/integration

clean: ## Remove the caches and runtime output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
