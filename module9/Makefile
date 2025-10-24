# Project settings
CUDA_PATH      ?= /usr/local/cuda
NVCC           := $(CUDA_PATH)/bin/nvcc
CUDALIBS       := -lcudart -lcurand -lcufft
INCLUDES       := -I$(CUDA_PATH)/include -I./src
SRC_DIR        := src
SRCS           := $(SRC_DIR)/main.cu $(SRC_DIR)/kernels.cu $(SRC_DIR)/utils.cu
OBJS           := $(SRCS:.cu=.o)
TARGET         := noise_fft

# Compiler flags
NVCCFLAGS      := -O2 -arch=sm_60 $(INCLUDES)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
    $(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDALIBS)

%.o: %.cu
    $(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
    rm -f $(SRC_DIR)/*.o $(TARGET)