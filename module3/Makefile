# Makefile for assignment.cu

TARGET = assignment
SRC = assignment.cu
NVCC = nvcc
NVCC_FLAGS = -O2

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
