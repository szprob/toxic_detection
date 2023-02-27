SHELL=/bin/bash
PROJECT_NAME=sprite

setup-dev:
	pip install -r requirements-lint.txt
	pip install -r requirements.txt
	python setup.py install

check:
	black --check .
	isort --check .
	flake8 .

test:
	pytest tests/*
