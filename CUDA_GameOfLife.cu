#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <limits.h>
#include <fstream>
#include <cassert>
#include <math.h>
#include <sys/time.h>
// OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// CUDA
#include <cuda.h>

#define MAX_SIZE 1024

using namespace std;

void saveImage(const cv::Mat &image, const std::string &filename)
{
	cv::imwrite(filename, image);
	cout << "Saved image: " << filename << endl;
}

void saveImages(const cv::Mat &population, int iter, int maxiter, const std::string &prefix)
{
	int displayQuarter = maxiter / 3; // Save 4~5 images in the iterations
	if (iter % displayQuarter == 0 || iter == maxiter - 1)
	{
		cv::Mat image_for_viewing;
		cv::resize(population, image_for_viewing, cv::Size(MAX_SIZE, MAX_SIZE), cv::INTER_LINEAR);
		string filename = prefix + "_iteration_" + to_string(iter) + ".png";
		saveImage(image_for_viewing, filename);
	}
}

#define cudaErrorCheck(result)                                  \
	{                                                           \
		cudaAssert((result), __FILE__, __FUNCTION__, __LINE__); \
	}

inline void cudaAssert(cudaError_t err, const char *file, const char *function, int line, bool quit = true)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "cudaAssert failed with error \'%s\', in File: %s, Function: %s, at Line: %d\n", cudaGetErrorString(err), file, function, line);
		if (quit)
			exit(err);
	}
}

// CUDA Kernel to update Game of Life using the provided cpp code
// Input parameters: initial population data, resulting population after applying rules, image row size, and image column size
// Applies a wrap around the image using boundaries for calculating neighbors
__global__ void updateGameOfLife(uchar *population, uchar *newpopulation, int ny, int nx)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check boundaries
	if (ix < nx && iy < ny)
	{
		int occupied_neighbours = 0;
		// Check living neighboring grids
		for (int jy = iy - 1; jy <= iy + 1; jy++)
		{
			for (int jx = ix - 1; jx <= ix + 1; jx++)
			{
				if (jx == ix && jy == iy)
					continue;
				// Wrap around rows using ny
				int row = jy;
				if (row < 0)
					row = ny - 1;
				else if (row >= ny)
					row = 0;
				// Wrap around columns using nx
				int col = jx;
				if (col < 0)
					col = nx - 1;
				else if (col >= nx)
					col = 0;
				// If current neighbor cell is alive, increment
				if (population[row * nx + col] == 0)
					occupied_neighbours++;
			}
		}
		// Apply rules
		if (population[iy * nx + ix] == 0)
		{ // Alive cell
			if (occupied_neighbours <= 1 || occupied_neighbours >= 4)
			{
				newpopulation[iy * nx + ix] = 255; // Dies
			}
			else if (occupied_neighbours == 2 || occupied_neighbours == 3)
			{
				newpopulation[iy * nx + ix] = 0; // Survives
			}
		}
		else
		{ // Dead cell
			if (occupied_neighbours == 3)
			{
				newpopulation[iy * nx + ix] = 0; // Reproduction
			}
			else
			{
				newpopulation[iy * nx + ix] = 255; // Stays dead
			}
		}
	}
}

//---------------------------------------
// CUDA C++ display CUDA device properties
//---------------------------------------
void printCUDADevice(cudaDeviceProp properties)
{
	cout << "CUDA Device: " << std::endl;
	cout << "\tDevice name              : " << properties.name << std::endl;
	cout << "\tMajor revision           : " << properties.major << std::endl;
	cout << "\tMinor revision           : " << properties.minor << std::endl;
	cout << "\tGlobal memory            : " << properties.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << " Gb" << std::endl;
	cout << "\tShared memory per block  : " << properties.sharedMemPerBlock / 1024.0 << " Kb" << std::endl;
	cout << "\tRegisters per block      : " << properties.regsPerBlock << std::endl;
	cout << "\tWarp size                : " << properties.warpSize << std::endl;
	cout << "\tMax threads per block    : " << properties.maxThreadsPerBlock << std::endl;
	cout << "\tMaximum x dim of block   : " << properties.maxThreadsDim[0] << std::endl;
	cout << "\tMaximum y dim of block   : " << properties.maxThreadsDim[1] << std::endl;
	cout << "\tMaximum z dim of block   : " << properties.maxThreadsDim[2] << std::endl;
	cout << "\tMaximum x dim of grid    : " << properties.maxGridSize[0] << std::endl;
	cout << "\tMaximum y dim of grid    : " << properties.maxGridSize[1] << std::endl;
	cout << "\tMaximum z dim of grid    : " << properties.maxGridSize[2] << std::endl;
	cout << "\tClock frequency          : " << properties.clockRate / 1000.0 << " MHz" << std::endl;
	cout << "\tConstant memory          : " << properties.totalConstMem << std::endl;
	cout << "\tNumber of multiprocs     : " << properties.multiProcessorCount << std::endl;
}

int main(int argc, char **argv)
{
	if (argc != 8)
	{
		cerr << "Usage: " << argv[0];
		cerr << " <imageRows> <imageColumns> ";
		cerr << "<Iterations> <X_BLOCK_SIZE> <Y_BLOCK_SIZE> ";
		cerr << "<Frame multiples to display, '0' for none> <save_images: 'yes' or 'no'>" << std::endl;
		return EXIT_FAILURE;
	}

	cudaError_t err;

	//---------------------------------------
	// Display Device Info
	//---------------------------------------
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, 0);

	printCUDADevice(device_properties);

	//-----------------------
	// Convert Command Line
	//-----------------------
	int ny = atoi(argv[1]);						// # of rows
	int nx = atoi(argv[2]);						// # of Columns
	int maxiter = atoi(argv[3]);				// Number of generations to simulate
	int block_size_x = atoi(argv[4]);			// block.x size
	int block_size_y = atoi(argv[5]);			// block.y size
	int displayIteration = atoi(argv[6]);		// Iteration multiple to display
	string saveQuarterlyImagesOption = argv[7]; // User option for saving quarterly images

	assert(ny <= MAX_SIZE); // Ensure within max size
	assert(nx <= MAX_SIZE); // Ensure within max size

	//---------------------------------
	// Generate the initial image
	//---------------------------------
	srand(clock());
	cv::Mat h1_population(ny, nx, CV_8UC1); // For GPU profiling
	// Initialize all grid cells as either alive or dead
	for (unsigned int iy = 0; iy < ny; iy++)
	{
		for (unsigned int ix = 0; ix < nx; ix++)
		{
			// seed a 1/2 density of alive (just arbitrary really)
			int state = rand() % 2;
			if (state == 0)
				h1_population.at<uchar>(iy, ix) = 255; // dead
			else
				h1_population.at<uchar>(iy, ix) = 0; // alive
		}
	}
	cv::Mat h2_population = h1_population.clone();	  // For CPU profiling
	cv::Mat h1_newpopulation = h1_population.clone(); // copy of initial population for GPU profiling
	cv::Mat h2_newpopulation = h1_population.clone(); // copy of initial population for CPU profiling

	cv::Mat image_for_viewing(MAX_SIZE, MAX_SIZE, CV_8UC1);

	//----------------------------------------------------
	// CPU COMPUTATION
	//----------------------------------------------------
	double cpu_t_start = (double)clock() / (double)CLOCKS_PER_SEC;
	// Determine grid cell states for each generation
	for (int iter = 0; iter < maxiter; iter++)
	{
		// Display the newpopulation using OpenCV in multiples of the iteration argument
		if (displayIteration != 0)
		{
			if (iter % displayIteration == 0)
			{
				cv::resize(h2_population, image_for_viewing, image_for_viewing.size(), cv::INTER_LINEAR); // resize image to MAX_SIZE x MAX_SIZE
				cv::imshow("CPU Population", image_for_viewing);
				cv::waitKey(10); // wait 10 seconds before closing image (or a keypress to close)
			}
		}
		for (int iy = 0; iy < ny; iy++) // Row index
		{
			for (int ix = 0; ix < nx; ix++) // Column index
			{
				int occupied_neighbours = 0; // Count alive neighbors
				for (int jy = iy - 1; jy <= iy + 1; jy++)
				{
					for (int jx = ix - 1; jx <= ix + 1; jx++)
					{
						if (jx == ix && jy == iy)
							continue;
						// Wrap around rows
						int row = jy;
						if (row == ny)
							row = 0;
						if (row == -1)
							row = ny - 1;
						// Wrap around columns
						int col = jx;
						if (col == nx)
							col = 0;
						if (col == -1)
							col = nx - 1;
						// If current cell is alive, increment
						if (h2_population.at<uchar>(row, col) == 0)
							occupied_neighbours++;
					}
				}
				// Apply game of life rules
				if (h2_population.at<uchar>(iy, ix) == 0) // alive
				{
					if (occupied_neighbours <= 1 || occupied_neighbours >= 4)
						h2_newpopulation.at<uchar>(iy, ix) = 255; // dies
					if (occupied_neighbours == 2 || occupied_neighbours == 3)
						h2_newpopulation.at<uchar>(iy, ix) = 0; // same as population
				}
				else if (h2_population.at<uchar>(iy, ix) == 255) // dead
				{
					if (occupied_neighbours == 3)
					{
						h2_newpopulation.at<uchar>(iy, ix) = 0; // reproduction
					}
				}
			}
		}
		h2_population = h2_newpopulation.clone(); // Update population with next population
		if (saveQuarterlyImagesOption == "yes")
		{
			// Save images quarterly for CPU
			saveImages(h2_population, iter, maxiter, "cpu");
		}
	}
	double cpu_time = (double)clock() / (double)CLOCKS_PER_SEC - cpu_t_start;

	//---------------------------------------
	// Setup GPU Profiling
	//---------------------------------------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//---------------------------------------
	// Create GPU (Device) Buffers
	//---------------------------------------
	// Allocate GPU memory for population and newpopulation
	uchar *d_population, *d_newpopulation;
	err = cudaMalloc((void **)&d_population, ny * nx * sizeof(uchar));
	cudaErrorCheck(err);
	err = cudaMalloc((void **)&d_newpopulation, ny * nx * sizeof(uchar));
	cudaErrorCheck(err);

	//---------------------------------------
	// Copy Memory To Device
	//---------------------------------------
	// Copy initial population data from Host to Device
	err = cudaMemcpy(d_population, h1_population.data, ny * nx * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaErrorCheck(err);

	//---------------------------------------
	// Setup Execution Configuration
	//---------------------------------------
	// Specify grid and block dimensions for CUDA kernel
	dim3 block_size(block_size_x, block_size_y);

	int gridy = (int)ceil((double)ny / (double)block_size_y);
	int gridx = (int)ceil((double)nx / (double)block_size_x);

	dim3 grid_size(gridx, gridy);

	if (block_size.x * block_size.y > device_properties.maxThreadsPerBlock)
	{
		cerr << "Block Size of " << block_size.x << " x " << block_size.y << " (" << block_size_x * block_size_y << " threads) is too big. " << endl;
		cerr << "Maximum threads per block = " << device_properties.maxThreadsPerBlock << endl;
		return -1;
	}
	else if (block_size.x > device_properties.maxThreadsDim[0] || block_size.y > device_properties.maxThreadsDim[1])
	{
		cerr << "Block Size of " << block_size.x << " x " << block_size.y << " is too big. " << endl;
		cerr << "Maximum threads for dimension 0 = " << device_properties.maxThreadsDim[0] << endl;
		cerr << "Maximum threads for dimension 1 = " << device_properties.maxThreadsDim[1] << endl;
		return -1;
	}
	else if (grid_size.x > device_properties.maxGridSize[0] || grid_size.y > device_properties.maxGridSize[1])
	{
		cerr << "Grid Size of " << grid_size.x << " x " << grid_size.y << " is too big. " << endl;
		cerr << "Maximum grid dimension 0 = " << device_properties.maxGridSize[0] << endl;
		cerr << "Maximum grid dimension 1 = " << device_properties.maxGridSize[1] << endl;
		return -1;
	}

	//---------------------------------------
	// Call Kernel
	//---------------------------------------
	// Iterate over populations
	for (int iter = 0; iter < maxiter; iter++)
	{
		// Launch CUDA kernel
		updateGameOfLife<<<grid_size, block_size>>>(d_population, d_newpopulation, ny, nx);

		//---------------------------------------
		// Obtain Result
		//---------------------------------------
		// Copy newpopulation data from Device to Host
		err = cudaMemcpy(h1_newpopulation.data, d_newpopulation, ny * nx * sizeof(uchar), cudaMemcpyDeviceToHost);
		cudaErrorCheck(err);

		// Swap pointers for the next iteration, keep temporary copy for displaying
		uchar *temp = d_population;
		d_population = d_newpopulation;
		d_newpopulation = temp;

		if (saveQuarterlyImagesOption == "yes")
		{
			// Save images quarterly for GPU
			saveImages(h1_newpopulation, iter, maxiter, "gpu");
		}
		// Display the newpopulation using OpenCV in multiples of the iteration argument
		if (displayIteration != 0)
		{
			if (iter % displayIteration == 0)
			{
				cv::resize(h1_newpopulation, image_for_viewing, image_for_viewing.size(), cv::INTER_LINEAR);
				cv::imshow("GPU Population", image_for_viewing);
				cv::waitKey(1);
			}
		}
	}

	//---------------------------------------
	// Stop GPU Profiling
	//---------------------------------------
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpu_time;
	cudaEventElapsedTime(&gpu_time, start, stop); // time in milliseconds

	gpu_time /= 1000.0;
	cout << "\n\nDone GPU Computations in " << gpu_time << " seconds" << endl;
	cout.flush();

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Free GPU memory
	cudaFree(d_population);
	cudaFree(d_newpopulation);

	//----------------------------------------------
	// Display Timing Results and Images
	//----------------------------------------------
	cout << "Results for image row size = " << nx << ", image column size = " << ny << ", iterations = " << maxiter;
	cout << ", X block size = " << block_size_x << ", Y block size = " << block_size_y << endl;
	cout << "\tBlock Size = " << block_size.x << " x " << block_size.y << "\n";
	cout << "\tGrid Size  = " << grid_size.x << " x " << grid_size.y << "\n";
	cout << "GPU Time = " << gpu_time << endl;
	cout << "CPU Time = " << cpu_time << endl;
	cout << "Speedup = " << cpu_time / gpu_time << endl;

	return 0;
}
