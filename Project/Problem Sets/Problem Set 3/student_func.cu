/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <float.h>
#include <limits.h>

__global__ void minmax_kernel(const float* const d_input,float*  d_output, size_t size, bool minmax){
   extern __shared__ float shard[];

   int index = threadIdx.x + blockDim.x * blockIdx.x;
   int tid = threadIdx.x;

   if(index < size){
      shard[tid] = d_input[index];
   }else{
      if(minmax==0){
         shard[tid] = FLT_MAX;
      }else{
         shard[tid] = -FLT_MAX;
      }
   }
   // wait for data copy
   __syncthreads();
   if(index > size){
      // printf("index > size \n");
      return;
   }

   for(int s = blockDim.x / 2; s > 0; s/=2){
      if(tid < s){
         if(minmax == 0){
            shard[tid] = min(shard[tid],shard[tid+s]);
         }else{
            shard[tid] = max(shard[tid],shard[tid+s]);
         }
      }
      __syncthreads();
   }

   if(tid == 0){
      d_output[blockIdx.x] = shard[0];
   }
}
__global__ void histogram_kernel(unsigned int* d_bins,const float* const d_logLuminance, int numBins,float min_logLum, float max_logLum, size_t size){
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if(index >= size){
      return;
   }
   int bin_index;
   bin_index = (d_logLuminance[index] - min_logLum) / (max_logLum - min_logLum) * numBins;
   atomicAdd(&d_bins[bin_index],1);
}

__global__ void scan_kernel(unsigned int *d_bins, int size){
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if(index >= size){
      return;
   }
   int val = 0;
   int self_val = d_bins[index];
   for(int s = 1; s<=size; s *=2){
      int pre = index - s;
      if(pre >= 0)
         val = d_bins[pre];
      __syncthreads();
      if(pre >= 0)
         d_bins[index] += val;
      __syncthreads();
      
   } 
   d_bins[index] -= self_val;  
}
float cal_size(int a, int b){
   return (int) ceil((float)a/(float)b);
}
float minmax(const float* const d_input,const size_t size, bool minmax){
   const int BLOCK_WIDTH = 32;
   size_t cur_size = size;
   float *d_cur_input;

   checkCudaErrors(cudaMalloc(&d_cur_input,sizeof(float)*size));
   checkCudaErrors(cudaMemcpy(d_cur_input, d_input, sizeof(float)*size, cudaMemcpyDeviceToDevice));

   float * d_cur_output;
   const int shared_mem_size = BLOCK_WIDTH * sizeof(float);
   while(1){
      checkCudaErrors(cudaMalloc(&d_cur_output,sizeof(float)*cal_size(cur_size,BLOCK_WIDTH)));
      minmax_kernel<<<cal_size(cur_size,BLOCK_WIDTH), BLOCK_WIDTH, shared_mem_size>>>(
         d_cur_input,
         d_cur_output,
         cur_size,
         minmax
      );
      cudaDeviceSynchronize();
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaFree(d_cur_input));
      d_cur_input = d_cur_output;
      if(cur_size < BLOCK_WIDTH)
         break;
      cur_size = cal_size(cur_size,BLOCK_WIDTH);
   }
   float h_output;
   checkCudaErrors(cudaMemcpy(&h_output, d_cur_output, sizeof(float), cudaMemcpyDeviceToHost));
   return h_output;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
   */
   const size_t size = numRows * numCols;
   min_logLum = minmax(d_logLuminance, size, 0);
   max_logLum = minmax(d_logLuminance, size, 1);
   
   printf("got min of %f\n", min_logLum);
   printf("got max of %f\n", max_logLum);
   printf("numBins %d\n", numBins);

   /*
    2) subtract them to find the range
   */
    /*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins 
   */
   unsigned int* d_bins;
   size_t histo_size = sizeof(unsigned int) * numBins;

   checkCudaErrors(cudaMalloc(&d_bins,histo_size));
   checkCudaErrors(cudaMemset(d_bins,0,histo_size));
   dim3 thread_dim(1024);
   dim3 hist_block_dim(cal_size(size, thread_dim.x));
   histogram_kernel<<<hist_block_dim, thread_dim>>>(d_bins, d_logLuminance, numBins, min_logLum, max_logLum, size);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   unsigned int h_out[100];
   cudaMemcpy(&h_out, d_bins, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost);
   for(int i = 0; i < 10; i++)
      printf("hist out %d\n", h_out[i]);

     /*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)
        */
   dim3 scan_block_dim(cal_size(numBins, thread_dim.x));

   scan_kernel<<<scan_block_dim, thread_dim>>>(d_bins, numBins);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
   cudaMemcpy(&h_out, d_bins, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost);
   for(int i = 0; i < 10; i++)
      printf("cdf out %d\n", h_out[i]);
   

   cudaMemcpy(d_cdf, d_bins, histo_size, cudaMemcpyDeviceToDevice);

   
   checkCudaErrors(cudaFree(d_bins));
//  
}
