CC := nvcc
CFLAGS := -O3 -I/usr/include/opencv4/
LDFLAGS := -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

all: CUDA_GameOfLife

CUDA_GameOfLife: CUDA_GameOfLife.cu

	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f CUDA_GameOfLife
