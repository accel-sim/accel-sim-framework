TARGET := post-traces-processing

$(TARGET): post-traces-processing.cpp
	g++ -std=c++11 -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean: 
	rm -f $(TARGET) *.o

