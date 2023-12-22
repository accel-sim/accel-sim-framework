TARGET := post-traces-processing

$(TARGET): post-traces-processing.cpp
	g++ -march=native -std=c++14 -O3 -g -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
