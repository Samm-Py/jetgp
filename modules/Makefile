
all: cython

cython: setup.py
	# This comand runs python to execute the cython file.
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -rf *.so
	rm -rf *.c
	rm -rf __pycache__
