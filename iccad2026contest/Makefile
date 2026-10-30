# ICCAD 2026 FloorSet Challenge — Build & Run
#
# Override Python version:  make PYTHON=python3.10 <target>
#

PYTHON ?= python3

all: compile

compile:
	$(PYTHON) setup.py build_ext --inplace

evaluate: compile
	$(PYTHON) iccad2026_evaluate.py --evaluate my_optimizer.py

test: compile
	$(PYTHON) iccad2026_evaluate.py --evaluate my_optimizer.py --test-id $(N)

eval-exe:
	$(PYTHON) iccad2026_evaluate.py --evaluate op_wrapper.py --test-id $(N)

dist: compile
	pyinstaller --onedir --name my_optimizer \
		--add-data "bstar_core*.so:." \
		--hidden-import torch --hidden-import numpy \
		--hidden-import Cython \
		--exclude-module iccad2026_evaluate \
		--exclude-module litetestLoader \
		--exclude-module liteLoader \
		--exclude-module lite_dataset \
		--exclude-module cost \
		--exclude-module utils \
		my_optimizer.py

clean:
	rm -rf build/ dist/ *.spec __pycache__/ *.pyc
	rm -f bstar_core.c bstar_core*.so
