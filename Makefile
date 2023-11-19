PY = python3

py = $(shell find -type f -name "*.py")
main = main.py
run:
	$(PY) $(py)