SHELL := /bin/bash

VENV := .venv

ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
	PYTHON := $(VENV_BIN)/python.exe
	RM := del /s /q
else
	VENV_BIN := $(VENV)/bin
	PYTHON := $(VENV_BIN)/python
	RM := rm -rf
endif

.PHONY: venv install lint test format run clean

venv:
	uv venv $(VENV)

install: venv
	uv sync --python $(PYTHON)

lint: 
	$(VENV_BIN)/pylint --disable=R,C app.py chatlib

test:
	PYTHONPATH=. $(VENV_BIN)/pytest -vv

format:
	$(VENV_BIN)/black app.py chatlib

run:
	$(PYTHON) app.py

clean:
	$(RM) $(VENV)
	$(RM) .pytest_cache
	$(RM) __pycache__
	$(RM) .mypy_cache