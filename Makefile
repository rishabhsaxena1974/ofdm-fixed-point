CC     = gcc
CFLAGS = -O2 -std=c99 -Wall -Wextra
COMMON = c/ofdm.c c/fft.c c/tables_q15.c
DEPS   = $(COMMON) c/ofdm.h

all: build/ofdm_sim build/bitexact

c/tables_q15.c: python/tables.py python/gen_tables.py
	python python/gen_tables.py

build/ofdm_sim: c/sim.c $(DEPS)
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ c/sim.c $(COMMON) -lm

build/bitexact: c/bitexact.c $(DEPS)
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ c/bitexact.c $(COMMON) -lm

test: all
	python python/make_vectors.py --est dft && ./build/bitexact vectors
	python python/make_vectors.py --est linear && ./build/bitexact vectors

results: all
	python python/experiments.py

clean:
	rm -rf build vectors

.PHONY: all test results clean
