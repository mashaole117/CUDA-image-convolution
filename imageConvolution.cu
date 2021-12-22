#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "helperFiles/inc/helper_cuda.h"
#include "helperFiles/inc/helper_functions.h"
#define MAX_EPSILON_ERROR 5e-3f
#define AVG_SIZE 7
#define KER_SIZE 3
#define TILE_SIZE 26
#define BLOCK_SIZE (TILE_SIZE + KER_SIZE - 1)
#define BLOCK_SIZE_AVG (TILE_SIZE + AVG_SIZE - 1)



__constant__ float dSharp[KER_SIZE*KER_SIZE];
__constant__ float dEdge[KER_SIZE*KER_SIZE];
__constant__ float dAvg[AVG_SIZE*AVG_SIZE];

texture<float, 2, cudaReadModeElementType> tex;

bool testResult = true;

void convSerial(float *input, float *kernel, float *output, int xDim, int yDim, int dim)
{
  float sum = 0;
  int middle = (dim-1)/2; // middle of kernal matrix
  int x, y;

  for (int i = middle; i < yDim - middle; i++) // rows covered by kernel
    for (int j = middle; j < xDim - middle; j++){ // cols covered by kernel
      sum = 0;
      for (int l = 0; l < dim; l++)
	      for (int p = 0; p < dim; p++){
	        y = l + i - middle; // calculate index for input image
          x = p + j - middle;
	        sum += input[y*xDim+x]*kernel[l*dim + p];
	      }
        output[i*xDim + j] = sum;
    }
}

__global__ void convGlobal(float *input, float *output, int xDim, int yDim, int dim)
{
  
  int row = blockIdx.x + (dim-1)/2;  
  int rowPixel = threadIdx.x + (dim-1)/2; 
  int globalIdx = row*xDim + rowPixel;                      
       
  int middle = (dim-1)/2;		

  int x, y;
  float sum = 0.0;			  
  
  if (globalIdx < xDim*yDim){
    for (int l = 0; l < dim; l++)
      for (int p = 0; p < dim; p++)
      {
	      x = p + rowPixel - middle;
	      y = l + row - middle;
	      sum+=input[y*xDim+x]*dSharp[l*dim + p]; // sharpening
        // sum+=input[y*xDim+x]*dEdge[l*dim + p]; // edge detect
        // sum+=input[y*xDim+x]*dAvg[l*dim + p]; // averaging
      }

    output[globalIdx]  = sum;
  }
  
}

__global__ void convShared(float* input,float *output, int xDim, int yDim, int dim)
{
  // __shared__ float sIn[BLOCK_SIZE_AVG][BLOCK_SIZE_AVG]; // tile array for averaging kernel
  __shared__ float sIn[BLOCK_SIZE][BLOCK_SIZE]; // tile array for all other kernels

    int h_out = threadIdx.y + blockIdx.y * TILE_SIZE; // height index for tiled outputs 
    int w_out = threadIdx.x + blockIdx.x * TILE_SIZE; // width index for tiled outputs

    int h_in = h_out - dim/2; // height index to create tiled input
    int w_in = w_out - dim/2; // width index to create tiled input

    if(h_in >= 0 && h_in < yDim && w_in >= 0 && w_in < xDim)
        sIn[threadIdx.y][threadIdx.x] = input[h_in*xDim + w_in]; // create tiled input for threads
    else
        sIn[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE){ // only threads within tile boundary calculate output
        
        float sum = 0.0;
        
        for(int i=0; i<dim; i++)
        {
          for(int j=0; j<dim; j++)
          {
            sum += dSharp[i*dim + j]*sIn[i+threadIdx.y][j+threadIdx.x]; // sharpening
            // sum += dEdge[i*dim + j] * sIn[i+threadIdx.y][j+threadIdx.x]; // edge detect
            // sum += dAvg[i*dim + j] * sIn[i+threadIdx.y][j+threadIdx.x]; // averaging
          } 
        }
        
        if(h_out < yDim && w_out < xDim) // restrict threads within image dimensions
            output[h_out*xDim + w_out] = sum;
    }
}

__global__ void convTexture(float *input, float *output, int xDim, int yDim, int dim)
{
  int h_out = threadIdx.y + blockIdx.y * yDim;
  int w_out = threadIdx.x + blockIdx.x * xDim;		  
  
        float sum = 0.0;
        
        for(int i=0; i<dim; i++)
        {
          for(int j=0; j<dim; j++)
          {
            sum += dSharp[i*dim + j]*tex2D(tex,i+threadIdx.y,j+threadIdx.x); // sharpening
            // sum += dEdge[i*dim + j] * sIn[i+threadIdx.y][j+threadIdx.x]; // edge detect
            // sum += dAvg[i*dim + j] * sIn[i+threadIdx.y][j+threadIdx.x]; // averaging
          } 
        }
        
        if(h_out < yDim && w_out < xDim) // restrict threads within image dimensions
            output[h_out*xDim + w_out] = sum;
}

int main(int argc, char **argv)
{
  
  int devID = findCudaDevice(argc, (const char **) argv);
  unsigned int width, height;
  float *input = NULL;
  float sharp[KER_SIZE*KER_SIZE] = {-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0};
  float edge[KER_SIZE*KER_SIZE] = {-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0};
  float avg[AVG_SIZE*AVG_SIZE];

  for(int i=0;i<AVG_SIZE*AVG_SIZE;i++)
  {
      avg[i] = 1.0/(AVG_SIZE*AVG_SIZE);
  }

  // char *imagePath = sdkFindFilePath("helperFiles/data/image21.pgm", argv[0]); // first image
  char *imagePath = sdkFindFilePath("helperFiles/data/lena_bw.pgm", argv[0]); // second image

  if (imagePath == NULL)
  {
      printf("Unable to source image file\n");
      exit(EXIT_FAILURE);
  }

  sdkLoadPGM(imagePath, &input, &width, &height);

  unsigned int size = width * height * sizeof(float);
  float *output = (float *) malloc(size); // Serial output
  float *hOutput = (float *) malloc(size); // CUDA output

  printf("Loaded image, %d x %d pixels\n", width, height);
  
  /*Serial execution*/

  printf("Serial convolution starting...\n");
  
  /*Sharpen*/
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  printf("Sharpening execution...\n");
  sdkStartTimer(&timer);
  
  convSerial(input, sharp, output, width, height, KER_SIZE);
  
  sdkStopTimer(&timer);
  
  printf("Sharpen processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  printf("%.2f Mpixels/sec\n",(width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  
  sdkDeleteTimer(&timer);

  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out_sharp.pgm");
  sdkSavePGM(outputFilename, output, width, height);

  //--------------------------------------------------------

  /*Edge detect*/
  // StopWatchInterface *timer = NULL;
  // sdkCreateTimer(&timer);

  // printf("Edge detection execution...\n");
  // sdkStartTimer(&timer);
  
  // convSerial(input, edge, output, width, height, KER_SIZE);
  
  // sdkStopTimer(&timer);
  
  // printf("Edge detection processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  // printf("%.2f Mpixels/sec\n",(width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  
  // sdkDeleteTimer(&timer);

  // char outputFilename1[1024];
  // strcpy(outputFilename1, imagePath);
  // strcpy(outputFilename1 + strlen(imagePath) - 4, "_out_edge.pgm");
  // sdkSavePGM(outputFilename1, output, width, height);

  //--------------------------------------------------------

  /*Average*/
  // StopWatchInterface *timer = NULL;
  // sdkCreateTimer(&timer);

  // printf("Averaging execution...\n");
  // sdkStartTimer(&timer);
  
  // convSerial(input, avg, output, width, height, AVG_SIZE);
  
  // sdkStopTimer(&timer);
  
  // printf("Averaging processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  // printf("%.2f Mpixels/sec\n",(width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  
  // sdkDeleteTimer(&timer);

  // char outputFilename2[1024];
  // strcpy(outputFilename2, imagePath);
  // strcpy(outputFilename2 + strlen(imagePath) - 4, "_out_avg.pgm");
  // sdkSavePGM(outputFilename2, output, width, height);

  /*-------------------------------------------------------------------*/


  /*CUDA execution*/  

  printf("Allocating device memory...\n");
  float *dInput,*dOutput,*dKernel,*dAverage;
  checkCudaErrors(cudaMalloc((void**)&dInput,size));
  checkCudaErrors(cudaMalloc((void**)&dOutput,size));
  
  printf("Copying data from host data to device...\n");
  checkCudaErrors(cudaMemcpy(dInput, input, size,cudaMemcpyHostToDevice));
  printf("Input done\n");

  /*Kernels to be applied by GPU*/
  checkCudaErrors(cudaMemcpyToSymbol(dSharp,sharp,KER_SIZE*KER_SIZE*sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(dEdge,edge,KER_SIZE*KER_SIZE*sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(dAvg,avg,AVG_SIZE*AVG_SIZE*sizeof(float))); 

  /*Allocate cuda array and assign input image to it*/
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc,width,height));
  checkCudaErrors(cudaMemcpyToArray(cuArray,0, 0,input,size,cudaMemcpyHostToDevice));

  tex.addressMode[0] = cudaAddressModeWrap;
  tex.addressMode[1] = cudaAddressModeWrap;
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = true;

  checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));


  // Resources for global execution
  int blocksAvg = height - (AVG_SIZE-1), threadsAvg = width - (AVG_SIZE-1); // for averaging kernel
  int blocks = height - (KER_SIZE-1), threads = width - (KER_SIZE-1); // for everything else

  // Resources for shared execution
  dim3 blockSizeAvg(BLOCK_SIZE_AVG,BLOCK_SIZE_AVG), gridSizeAvg(ceil((float)width/TILE_SIZE),ceil((float)height/TILE_SIZE)); // for averaging execution
  dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE), gridSize(ceil((float)width/TILE_SIZE),ceil((float)height/TILE_SIZE)); // for everything else

  checkCudaErrors(cudaDeviceSynchronize());
  sdkCreateTimer(&timer);

  printf("Staring kernel execution...\n");
  sdkStartTimer(&timer);
  
  /*EVERYTHING ELSE*/
  // uncomment desired execution type, global shared or texture

  // convGlobal<<<blocks, threads>>>(dInput, dOutput, width, height, KER_SIZE);
  // convShared<<<gridSize, blockSize>>>(dInput, dOutput, width, height, KER_SIZE);
  convTexture<<<gridSize, blockSize>>>(dInput, dOutput, width, height, KER_SIZE);

  
  /*AVERAGING CALLS*/
  // uncomment desired execution type, global, shared or texture.
   
  // convGlobal<<<blocksAvg, threadsAvg>>>(dInput, dOutput, width, height, AVG_SIZE);
  // convShared<<<gridSizeAvg, blockSizeAvg>>>(dInput, dOutput, width, height,AVG_SIZE);
  // convTexture<<<blocksAvg, threadsAvg>>>(dInput, dOutput, width, height, AVG_SIZE);

  getLastCudaError("Kernel execution failed");
  checkCudaErrors(cudaDeviceSynchronize());
  
  sdkStopTimer(&timer);
  
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  printf("%.2f Mpixels/sec\n",(width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
  
  sdkDeleteTimer(&timer);

  checkCudaErrors(cudaMemcpy(hOutput,dOutput,size,cudaMemcpyDeviceToHost));
  
  char outputFileNameGPU[1024];
  strcpy(outputFileNameGPU, imagePath);
  strcpy(outputFileNameGPU + strlen(imagePath) - 4, "_out_gpu.pgm");
  sdkSavePGM(outputFileNameGPU, hOutput, width, height);

  printf("Wrote '%s'\n", outputFileNameGPU);
  
  testResult = compareData(output,hOutput,width*height,MAX_EPSILON_ERROR,0.15f);
  checkCudaErrors(cudaFree(dOutput));
  free(imagePath);
  checkCudaErrors(cudaFreeArray(cuArray));
  
  cudaDeviceReset();
  printf("Kernel and serial completed, returned %s\n",testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}