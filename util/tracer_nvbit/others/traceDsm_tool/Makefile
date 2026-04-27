all: traceDsm

traceDsm: traceDsm.cpp
	g++ -std=c++17 -O3 -g -I../../tracer_tool -o $@ $< -lzstd

clean:
	rm -f traceDsm
