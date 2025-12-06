#include <climits>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include "gemini/core/mpi.hpp"
#include "type.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <unistd.h>
#include "edge_container.hpp"
#include "lr_scheduler.hpp"
#include "util.hpp"
#include <algorithm>
#include <mpi.h>

extern int my_rank;

using std::vector;
using std::string;
using std::cout;
using std::endl;

namespace {

void log_host_allocation(const char* tag, size_t bytes)
{
    if (my_rank != 0) {
        return;
    }
    printf("[MEM][HOST] %s -> %.2f MB\n", tag, bytes / (1024.0 * 1024.0));
}

void log_host_growth(const char* tag, size_t old_bytes, size_t new_bytes)
{
    if (my_rank != 0) {
        return;
    }
    printf("[MEM][HOST] %s resize %.2f MB -> %.2f MB\n",
           tag,
           old_bytes / (1024.0 * 1024.0),
           new_bytes / (1024.0 * 1024.0));
}

void log_host_release(const char* tag, size_t bytes = 0)
{
    if (my_rank != 0) {
        return;
    }
    if (bytes > 0) {
        printf("[MEM][HOST] %s freed %.2f MB\n", tag, bytes / (1024.0 * 1024.0));
    } else {
        printf("[MEM][HOST] %s freed\n", tag);
    }
}

size_t vocab_code_total_bytes = 0;
size_t vocab_point_total_bytes = 0;

} // namespace

#define DELTA_R  100
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define EVALUATION_NEIGHBOUR_NUM 30
#define NODE_TRAINING_CONVERGE_THRESHOLD 0.65
#define NODE_TRAINING_REACTIVATE_THRESHOLD 0.65
#define EVALUATION_NEIGHBOUR_NUM_CONVERGE_RATIO 0.9

#define MAX_SENTENCE 15000
#define checkCUDAerr(err) {\
  cudaError_t cet = err;\
  if (cudaSuccess != cet) {\
    printf("%s %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(cet));\
    exit(0);\
  }\
}

vector<int> vertex_walker_stop_flag;
std::mutex mtx;
std::condition_variable cv;
bool hasResource = false;
std::atomic<bool> pauseWalk{false};
extern volatile bool stop_sampling_flag;
size_t vocab_hash_size = 900000000ULL;  // Default capacity, adjusted dynamically per dataset
constexpr size_t kVocabHashCap = 900000000ULL;  // Hard cap to avoid excessive allocations

static size_t CalculateVocabHashSize(size_t node_count) {
  if (node_count == 0) {
    return std::min<size_t>(kVocabHashCap, static_cast<size_t>(1024));
  }
  const double load_factor = 0.7;
  size_t desired = static_cast<size_t>(std::ceil(node_count / load_factor));
  if (desired > kVocabHashCap) desired = kVocabHashCap;
  if (desired < 1024) desired = 1024;
  return desired;
}

namespace {
struct AddWordToVocabProfile {
  double total_time = 0.0;
  double alloc_time = 0.0;
  double copy_time = 0.0;
  double realloc_time = 0.0;
  double hash_time = 0.0;
  double probe_time = 0.0;
  long long call_count = 0;
};

inline AddWordToVocabProfile &ProfileStorage() {
  static AddWordToVocabProfile profile;
  return profile;
}

inline void ResetAddWordToVocabProfile() {
  ProfileStorage() = AddWordToVocabProfile{};
}

inline void RecordAddWordToVocabSample(double total, double alloc, double copy, double realloc_t,
                                        double hash, double probe) {
  auto &profile = ProfileStorage();
  profile.total_time += total;
  profile.alloc_time += alloc;
  profile.copy_time += copy;
  profile.realloc_time += realloc_t;
  profile.hash_time += hash;
  profile.probe_time += probe;
  profile.call_count += 1;
}

inline AddWordToVocabProfile GetAddWordToVocabProfile() {
  return ProfileStorage();
}
}

MPI_Comm MPI_EMB_COMM;
MPI_Comm MPI_EVA_COMM;
int num_procs = 1;
int my_rank = 0;

float model_sync_period = 0.1f;
mutex sync_mtx;
condition_variable sync_cv;
bool trainBlocked = false;

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct TrainingConfig {
    int init_round;
    int batch_size;
    
    TrainingConfig(int init_round = 1, int batch_size = 16384) 
        : init_round(init_round), batch_size(batch_size) {}
};

// ========================================================
// GPU OPTIMIZATION CONFIGURATION SWITCHES
// ========================================================
// Toggle between original memory-copy batch implementation and GPU-direct implementation
// Set to true to enable GPU-side direct indexing optimization (eliminates memory copy bottleneck)
bool use_gpu_direct_indexing = true;

// Toggle between simple and shared-memory optimized kernel for direct indexing
// Only used when use_gpu_direct_indexing = true

namespace {
constexpr uint16_t kFullKeepThreshold = std::numeric_limits<uint16_t>::max();

struct FastRandomState {
  unsigned long long state;
  explicit FastRandomState(unsigned long long seed = 1ULL) : state(seed ? seed : 1ULL) {}
  inline uint32_t Next32() {
    state = state * 25214903917ULL + 11ULL;
    return static_cast<uint32_t>(state >> 16);
  }
  inline uint16_t Next16() {
    return static_cast<uint16_t>(Next32() >> 16);
  }
};

inline unsigned long long InitSeedForRank(int rank, unsigned long long extra = 0ULL) {
  unsigned long long seed = 0x9E3779B97F4A7C15ULL ^ ((static_cast<unsigned long long>(rank) + 1ULL) << 1) ^ extra;
  if (seed == 0ULL) {
    seed = 1ULL;
  }
  return seed;
}
}

bool use_optimized_direct_kernel = true;
// ========================================================

float *last_emb;

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, min_reduce = 1, reuseNeg = 1;
long long *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
float alpha = 0.025, starting_alpha, sample = 1e-3;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// FOR CUDA
int *vocab_codelen, *vocab_point, *d_vocab_codelen, *d_vocab_point;
char *vocab_code, *d_vocab_code;
int *d_table;
float *d_syn0, *d_syn1, *d_expTable;

static std::vector<uint16_t> BuildSubsamplingThresholds(double sample_value, long long total_train_words) {
  std::vector<uint16_t> thresholds;
  if (sample_value <= 0.0 || total_train_words <= 0) {
    return thresholds;
  }
  const double sample_train_words = sample_value * static_cast<double>(total_train_words);
  if (sample_train_words <= 0.0) {
    return thresholds;
  }
  if (vocab_size <= 0) {
    return thresholds;
  }
  thresholds.resize(static_cast<size_t>(vocab_size), kFullKeepThreshold);
  for (long long i = 0; i < vocab_size; ++i) {
    long long count = vocab[i].cn;
    if (count <= 0) {
      thresholds[static_cast<size_t>(i)] = kFullKeepThreshold;
      continue;
    }
    double prob = (sqrt(count / sample_train_words) + 1.0) * (sample_train_words / static_cast<double>(count));
    if (prob > 1.0) {
      prob = 1.0;
    } else if (prob < 0.0) {
      prob = 0.0;
    }
    if (prob >= 1.0) {
      thresholds[static_cast<size_t>(i)] = kFullKeepThreshold;
    } else {
      thresholds[static_cast<size_t>(i)] = static_cast<uint16_t>(prob * 65535.0 + 0.5);
    }
  }
  return thresholds;
}

__device__ float reduceInWarp(float f) {
  for (int i=warpSize/2; i>0; i/=2) {
    f += __shfl_sync(0xFFFFFFFF, f, i, 32);
  }
  return f;
}

__device__ void warpReduce(volatile float* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

// calculate cosine similarity of all nodes themselves
__global__ void cosine_similarity_kernel(float *d_vectors, float *d_result, long long  v, int dim){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("thread [%d,%d]\n",i,j);
	if( i < v && j < v) {
		float dot_product = 0.0f;
		float norm_i = 0.0f;
		float norm_j = 0.0f;

		// calculate dot_product and L2 norm.
		for (int k = 0; k < dim; ++k) {
			float vec_i = d_vectors[i * dim + k];
      float vec_j = d_vectors[j * dim + k];
			dot_product += vec_i * vec_j;
			norm_i += vec_i * vec_i;
			norm_j += vec_j * vec_j;	
		}
		norm_i = sqrt(norm_i);
		norm_j = sqrt(norm_j);
		// calculate similarity
		if(norm_i > 0.0f && norm_j > 0.0f) {
			d_result[i * v + j] = dot_product / (norm_i * norm_j);
		}else {
			d_result[i * v + j] = 0.0f; // prevent division by zero
		}
	}

}

__device__ float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}


__global__ void compute_kl_divergence_kernel(float *d_A,float *d_B, float *d_result, long long  v){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( idx < v * v) {
		float p = sigmoid(d_A[idx]);
		float q = sigmoid(d_B[idx]);
    // printf("sigmid(%d): %f, %f\n",idx,p,q);
		// 计算相对熵的部分贡献
		if(p > 0.0f && q > 0.0f){
			float contribution = p * logf(p /q );
      // printf("compute kl idx: %d val: %.2f\n",idx, contribution);
			atomicAdd(d_result, contribution); //  sum up
		}
	}
}

void  compute_kl_node_and_emb(float *h_node_sim, float *h_emb,float* h_result, long long  v,int dim){
	float *d_node_sim, *d_emb, *d_cosine, *d_result;
	cudaMalloc((void**)&d_emb, v * dim * sizeof(float));	
	cudaMalloc((void**)&d_node_sim, v * v * sizeof(float));	
	cudaMalloc((void**)&d_cosine, v * v * sizeof(float));	
	cudaMalloc((void**)&d_result, sizeof(float));
	
	cudaMemset(d_result, 0, sizeof(float));

	cudaMemcpy(d_node_sim,h_node_sim, v * v * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_emb,h_emb, v * dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(256);
	dim3 gridSize((v*v + blockSize.x - 1) / blockSize.x);

	// d_cosine 
	cosine_similarity_kernel<<<gridSize, blockSize>>>(d_emb,d_cosine,v,dim);

	// kl_divergence
	compute_kl_divergence_kernel<<<gridSize,blockSize>>>(d_node_sim,d_cosine,d_result,v);

	// copy result from device to host
	cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_node_sim);
	cudaFree(d_emb);
	cudaFree(d_cosine);
	cudaFree(d_result);
}
void write2file(float* ptr,size_t size,char* filename){
	FILE* f = fopen(filename,"w");
	if(f==NULL){
		printf("Failed to open %s\n",filename);
		return;
	}
	for(size_t i = 0;i<size; i++){
		fprintf(f,"%.3f ",ptr[i]);
	}
	fclose(f);
}

void  compute_kl_from_emb(float *emb1, float *emb2,float* h_result, long long v,int dim){
	float *d_emb1, *d_emb2, *d_cosine1, *d_cosine2, *d_result;
	cudaMalloc((void**)&d_emb1, v * dim * sizeof(float));	
	cudaMalloc((void**)&d_emb2, v * dim * sizeof(float));	
	cudaMalloc((void**)&d_cosine1, (size_t)v * v * sizeof(float));	
	cudaMalloc((void**)&d_cosine2, (size_t)v * v * sizeof(float));	
	cudaMalloc((void**)&d_result, sizeof(float));
	
	cudaMemset(d_result, 0, sizeof(float));


	cudaMemcpy(d_emb1,emb1, v * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_emb2,emb2, v * dim * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(16,16);
	dim3 gridSize((v+blockSize.x-1) / blockSize.x,(v+blockSize.y-1)/blockSize.y);

	// d_cosine 
	cosine_similarity_kernel<<<gridSize, blockSize>>>(d_emb1,d_cosine1,v,dim);
	cosine_similarity_kernel<<<gridSize, blockSize>>>(d_emb2,d_cosine2,v,dim);

	float *h_cosine = (float*)malloc( (size_t)v * v * sizeof(float));
	if(h_cosine == NULL)printf("Failed to allocate Mem\n");
  cudaDeviceSynchronize();
	cudaMemcpy(h_cosine,d_cosine1, (size_t)v * v * sizeof(float), cudaMemcpyDeviceToHost);
	write2file(h_cosine,(size_t)v * v,"cosine1.txt");
	cudaMemcpy(h_cosine,d_cosine2, (size_t)v * v  * sizeof(float), cudaMemcpyDeviceToHost);
	write2file(h_cosine,(size_t)v* v,"cosine2.txt");

	int kl_blockSize = 256;
  int kl_gridSize = (v * v + kl_blockSize - 1) / kl_blockSize;
	// kl_divergence
	compute_kl_divergence_kernel<<<kl_gridSize,kl_blockSize>>>(d_cosine1,d_cosine2,d_result,v);
  cudaDeviceSynchronize();

	// copy result from device to host
	cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_emb1);
	cudaFree(d_emb2);
	cudaFree(d_cosine1);
	cudaFree(d_cosine2);
	cudaFree(d_result);
}


template<unsigned int VSIZE>
__global__ void __sgNegReuse(const int window, const int layer1_size, const int negative, const int vocab_size, float alpha,
    const int* __restrict__ sen, const int* __restrict__ sentence_length,
    float *syn1, float *syn0, const int *negSample)
{
  __shared__ float neu1e[VSIZE];

  const int sentIdx_s = sentence_length[blockIdx.x];
  const int sentIdx_e = sentence_length[blockIdx.x + 1];
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int dxy = blockDim.x * blockDim.y;

  int _negSample;
  if (threadIdx.y < negative) {                                         // Get the negative sample
    _negSample = negSample[blockIdx.x * negative + threadIdx.y];
  }

  for (int sentPos = sentIdx_s; sentPos < sentIdx_e; sentPos++) {
    int word = sen[sentPos];                                            // Target word
    if (word == -1) continue;

    for (int a=0; a<window*2+1; a++) if (a != window) {
      int c = sentPos - window + a;                                     // The index of context word
      if (c >= sentIdx_s && c < sentIdx_e && sen[c] != -1) {
        int l1 = sen[c] * layer1_size;

        for (int i=tid; i<layer1_size; i+=dxy) {
          neu1e[i] = 0;
        }
        __syncthreads();

        int target, label, l2;
        float f = 0, g;
        if (threadIdx.y == negative) {                                  // Positive sample
          target = word;
          label = 1;
        } else {                                                        // Negative samples
          if (_negSample == word) goto NEGOUT;
          target = _negSample;
          label = 0;
        }
        l2 = target * layer1_size;

        for (int i=threadIdx.x; i<layer1_size; i+=blockDim.x) {         // Get gradient
          f += syn0[i + l1] * syn1[i + l2];
        }
        f = reduceInWarp(f);
        if      (f >  MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else {
          int tInt = (int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
          float t = exp((tInt / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
          t = t / (t + 1);
          g = (label - t) * alpha;
        }
        
        for (int i=threadIdx.x; i<layer1_size; i+=warpSize) {
          atomicAdd(&neu1e[i], g * syn1[i + l2]);
        }
        for (int i=threadIdx.x; i<layer1_size; i+=warpSize) {           // Update syn1 of negative sample
          syn1[i + l2] += g * syn0[i + l1];
        }

NEGOUT:
        __syncthreads();

        for (int i=tid; i<layer1_size; i+=dxy) {                        // Update syn0 of context word
          atomicAdd(&syn0[i + l1], neu1e[i]);
        }
      }
    }
  }
}

template<unsigned int FSIZE>
__global__ void skip_gram_kernel(int window, int layer1_size, int negative, int hs, int table_size, int vocab_size, float alpha,
    const float* __restrict__ expTable, const int* __restrict__ table, 
    const int* __restrict__ vocab_codelen, const int* __restrict__ vocab_point, const char* __restrict__ vocab_code,
    const int* __restrict__ sen, const int* __restrict__ sentence_length, float *syn1, float *syn0)
{
  __shared__ float f[FSIZE], g;

  int sent_idx_s = sentence_length[blockIdx.x];
  int sent_idx_e = sentence_length[blockIdx.x + 1]; 
  unsigned long next_random = blockIdx.x;

  if (threadIdx.x < layer1_size) for (int sentence_position = sent_idx_s; sentence_position < sent_idx_e; sentence_position++) {
    int word = sen[sentence_position];
    if (word == -1) continue;
    float neu1e = 0;
    next_random = next_random * (unsigned long)2514903917 + 11; 
    int b = next_random % window;

    for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      int c = sentence_position - window + a;
      if (c <  sent_idx_s) continue;
      if (c >= sent_idx_e) continue;
      int last_word = sen[c];
      if (last_word == -1) continue;
      int l1 = last_word * layer1_size;
      neu1e = 0;

      // HIERARCHICAL SOFTMAX
      if (hs) for (int d = vocab_codelen[word]; d < vocab_codelen[word+1]; d++) {
        int l2 = vocab_point[d] * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i) f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }

        if      (f[0] <= -MAX_EXP) continue;
        else if (f[0] >=  MAX_EXP) continue;
        else if (threadIdx.x == 0) {
          f[0] = expTable[(int)((f[0] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - vocab_code[d] - f[0]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[threadIdx.x + l2];
        atomicAdd(&syn1[threadIdx.x + l2], g * syn0[threadIdx.x + l1]);
      }

      // NEGATIVE SAMPLING
      if (negative > 0) for (int d = 0; d < negative + 1; d++) {
        int target, label;
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long)25214903917 + 11; 
          target = table[(next_random >> 16) % table_size];
          if (target == 0)    target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        int l2 = target * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = syn0[threadIdx.x +l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i)
            f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          if (f[0] >  MAX_EXP)
            g = (label - 1) * alpha;
          else if (f[0] < -MAX_EXP)
            g = (label - 0) * alpha;
          else
            g = (label - expTable[(int)((f[0]+MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[threadIdx.x + l2];
        atomicAdd(&syn1[threadIdx.x + l2], g * syn0[threadIdx.x + l1]);
      }

      atomicAdd(&syn0[threadIdx.x + l1], neu1e);
    }
  }
}

template<unsigned int FSIZE>
__global__ void cbow_kernel(int window, int layer1_size, int negative, int hs, int table_size, int vocab_size, float alpha,
    const float* __restrict__ expTable, const int* __restrict__ table,
    const int* __restrict__ vocab_codelen, const int* __restrict__ vocab_point, const char* __restrict__ vocab_code,
    const int* __restrict__ sen, const int* __restrict__ sentence_length, float *syn1, float *syn0)
{
  __shared__ float f[FSIZE], g;

  int sent_idx_s = sentence_length[blockIdx.x];
  int sent_idx_e = sentence_length[blockIdx.x + 1];
  unsigned long next_random = blockIdx.x;

  if (threadIdx.x < layer1_size) for (int sentence_position = sent_idx_s; sentence_position < sent_idx_e; sentence_position++) {
    int word = sen[sentence_position];
    if (word == -1) continue;
    float neu1 = 0;
    float neu1e = 0;
    next_random = next_random * (unsigned long)2514903917 + 11;
    int b = next_random % window;

    int cw = 0;
    for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      int c = sentence_position - window + a;
      if (c <  sent_idx_s) continue;
      if (c >= sent_idx_e) continue;
      int last_word = sen[c];
      if (last_word == -1) continue;
      neu1 += syn0[last_word * layer1_size + threadIdx.x];
      cw++;
    }

    if (cw) {
      neu1 /= cw;

      // HIERARCHICAL SOFTMAX
      if (hs) for (int d = vocab_codelen[word]; d < vocab_codelen[word+1]; d++) {
        int l2 = vocab_point[d] * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i)
            f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }

        if      (f[0] <= -MAX_EXP) continue;
        else if (f[0] >=  MAX_EXP) continue;
        else if (threadIdx.x == 0) {
          f[0] = expTable[(int)((f[0] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - vocab_code[d] - f[0]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[threadIdx.x + l2];
        atomicAdd(&syn1[threadIdx.x + l2], g * neu1);
      }

      // NEGATIVE SAMPLING
      if (negative > 0) for (int d = 0; d < negative + 1; d++) {
        int target, label;
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target==0)    target = next_random % (vocab_size - 1) + 1;
          if (target==word) continue;
          label = 0;
        }
        int l2 = target * layer1_size;

        if (threadIdx.x <  FSIZE) f[threadIdx.x] = neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        if (threadIdx.x >= FSIZE) f[threadIdx.x%(FSIZE)] += neu1 * syn1[threadIdx.x + l2];
        __syncthreads();
        for (int i=(FSIZE/2); i>0; i/=2) {
          if (threadIdx.x < i)
            f[threadIdx.x] += f[i + threadIdx.x];
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          if (f[0] > MAX_EXP)
            g = (label - 1) * alpha;
          else if (f[0] < -MAX_EXP)
            g = (label - 0) * alpha;
          else
            g = (label - expTable[(int)((f[0]+MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        }
        __syncthreads();

        neu1e += g * syn1[l2 + threadIdx.x];
        atomicAdd(&syn1[l2 + threadIdx.x], g * neu1);
      }

      for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        int c = sentence_position - window + a;
        if (c <  sent_idx_s) continue;
        if (c >= sent_idx_e) continue;
        int last_word = sen[c];
        if (last_word == -1) continue;
        atomicAdd(&syn0[last_word * layer1_size + threadIdx.x], neu1e);
      }
    }
  }
}

void InitVocabStructCUDA()
{
  vocab_codelen = (int *)malloc((vocab_size + 1) * sizeof(int));
  log_host_allocation("vocab codelen host", (vocab_size + 1) * sizeof(int));
  vocab_codelen[0] = 0;
  for (int i = 1; i < vocab_size + 1; i++) 
    vocab_codelen[i] = vocab_codelen[i-1] + vocab[i-1].codelen;
  vocab_point = (int *)malloc(vocab_codelen[vocab_size] * sizeof(int));
  vocab_code = (char *)malloc(vocab_codelen[vocab_size] * sizeof(char));
  log_host_allocation("vocab point host", vocab_codelen[vocab_size] * sizeof(int));
  log_host_allocation("vocab code host", vocab_codelen[vocab_size] * sizeof(char));

  checkCUDAerr(cudaMalloc((void **)&d_vocab_codelen, (vocab_size + 1) * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_vocab_point, vocab_codelen[vocab_size] * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_vocab_code, vocab_codelen[vocab_size] * sizeof(char)));
  printf("[InitVocabStructCUDA] CUDA malloc vocab_codelen((%zu MB)), vocab_point((%zu MB)), vocab_code((%zu MB)) ,total (%zu MB)!\n",
      (vocab_size + 1) * sizeof(int) / (1024 * 1024),
      vocab_codelen[vocab_size] * sizeof(int) / (1024 * 1024),
      vocab_codelen[vocab_size] * sizeof(char) / (1024 * 1024),
      ((vocab_size + 1) * sizeof(int) + vocab_codelen[vocab_size] * sizeof(int) + vocab_codelen[vocab_size] * sizeof(char)) / (1024 * 1024));
  for (int i=0; i<vocab_size; i++) {
    for (int j=0; j<vocab[i].codelen; j++) {
      vocab_code[vocab_codelen[i] + j] = vocab[i].code[j];
      vocab_point[vocab_codelen[i] + j] = vocab[i].point[j];
    }   
  }   

  checkCUDAerr(cudaMemcpy(d_vocab_codelen, vocab_codelen, (vocab_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMemcpy(d_vocab_point, vocab_point, vocab_codelen[vocab_size] * sizeof(int), cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMemcpy(d_vocab_code, vocab_code, vocab_codelen[vocab_size] * sizeof(char), cudaMemcpyHostToDevice));
}


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  log_host_allocation("unigram table host", table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
  // FOR CUDA
  checkCUDAerr(cudaMalloc((void **)&d_table, table_size*sizeof(int)));
  printf("[ %d ] [InitUnigramTable] CUDA malloc d_table (%zu MB) done!\n", my_rank, table_size * sizeof(int) / (1024 * 1024));
  checkCUDAerr(cudaMemcpy(d_table, table, table_size*sizeof(int), cudaMemcpyHostToDevice));
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) { if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  //  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  Timer total_timer;
  Timer step_timer;
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;

  step_timer.restart();
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  double alloc_time = step_timer.duration();

  step_timer.restart();
  strcpy(vocab[vocab_size].word, word);
  double copy_time = step_timer.duration();

  vocab[vocab_size].cn = 0;
  vocab_size++;

  step_timer.restart();
  if (vocab_size + 2 >= vocab_max_size) {
    size_t old_bytes = (size_t)vocab_max_size * sizeof(struct vocab_word);
    vocab_max_size += 1000;
    size_t new_bytes = (size_t)vocab_max_size * sizeof(struct vocab_word);
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    log_host_growth("vocab array", old_bytes, new_bytes);
  }
  double realloc_time = step_timer.duration();

  step_timer.restart();
  hash = GetWordHash(word);
  double hash_time = step_timer.duration();

  step_timer.restart();
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  double probe_time = step_timer.duration();

  int index = vocab_size - 1;
  vocab_hash[hash] = index;

  double total_time = total_timer.duration();
  RecordAddWordToVocabSample(total_time, alloc_time, copy_time, realloc_time, hash_time, probe_time);
  return index;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  size_t size;
  unsigned long long hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[0], vocab_size, sizeof(struct vocab_word), VocabCompare);
  for (size_t hash_idx = 0; hash_idx < vocab_hash_size; ++hash_idx) vocab_hash[hash_idx] = -1;
  size = vocab_size;
  train_words = 0;
  for (size_t a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  log_host_allocation("vocab shrink", (size_t)(vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  vocab_code_total_bytes = 0;
  vocab_point_total_bytes = 0;
  for (size_t a = 0; a < static_cast<size_t>(vocab_size); a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    vocab_code_total_bytes += MAX_CODE_LENGTH * sizeof(char);
    vocab_point_total_bytes += MAX_CODE_LENGTH * sizeof(int);
  }
  log_host_allocation("vocab code path total", vocab_code_total_bytes);
  log_host_allocation("vocab point path total", vocab_point_total_bytes);
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (size_t hash_idx = 0; hash_idx < vocab_hash_size; ++hash_idx) vocab_hash[hash_idx] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  size_t tree_array_len = static_cast<size_t>(vocab_size * 2 + 1);
  long long *count = (long long *)calloc(tree_array_len, sizeof(long long));
  long long *binary = (long long *)calloc(tree_array_len, sizeof(long long));
  long long *parent_node = (long long *)calloc(tree_array_len, sizeof(long long));
  log_host_allocation("huffman count array", tree_array_len * sizeof(long long));
  log_host_allocation("huffman binary array", tree_array_len * sizeof(long long));
  log_host_allocation("huffman parent array", tree_array_len * sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  log_host_release("huffman count array", tree_array_len * sizeof(long long));
  free(binary);
  log_host_release("huffman binary array", tree_array_len * sizeof(long long));
  free(parent_node);
  log_host_release("huffman parent array", tree_array_len * sizeof(long long));
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (size_t hash_idx = 0; hash_idx < vocab_hash_size; ++hash_idx) vocab_hash[hash_idx] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
  if (vocab_size > static_cast<long long>(static_cast<double>(vocab_hash_size) * 0.7)) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words); 
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

vector<vertex_id_t> id2offset;

void ReadVocabFromDegree(vector<vertex_id_t>& degrees){
  Timer total_timer;
  Timer section_timer;

  vertex_id_t v_num = degrees.size();
  char word[MAX_STRING];

  ResetAddWordToVocabProfile();

  section_timer.restart();
  for (size_t hash_idx = 0; hash_idx < vocab_hash_size; ++hash_idx) vocab_hash[hash_idx] = -1;
  double hash_init_time = section_timer.duration();

  vocab_size = 0;
  section_timer.restart();
  Timer sprintf_timer;
  double sprintf_time = 0;
  for (vertex_id_t v = 0; v < v_num; v++)
  {
    sprintf_timer.restart();
    std::sprintf(word,"%u",v);  // node ID 以字符串的形式存在 vocab 里面。
    sprintf_time += sprintf_timer.duration();
    int idx = AddWordToVocab(word);
    vocab[idx].cn = degrees[v];
  }
  double add_vocab_time = section_timer.duration();
  auto profile = GetAddWordToVocabProfile();
  double add_word_avg = profile.call_count ? profile.total_time / static_cast<double>(profile.call_count) : 0.0;
  printf("[ %d ] Sprintf time: %.3f,AddWordToVocab loop: %.3f\n", my_rank, sprintf_time, add_vocab_time - sprintf_time);
  printf("[ %d ] AddWordToVocab stats: calls=%lld, total=%.3f s, avg=%.6e s\n",
         my_rank, profile.call_count, profile.total_time, add_word_avg);
  printf("[ %d ]   alloc=%.3f s, copy=%.3f s, realloc=%.3f s, hash=%.3f s, probe=%.3f s\n",
         my_rank, profile.alloc_time, profile.copy_time, profile.realloc_time, profile.hash_time, profile.probe_time);

  printf("[ %d ] Add Word To Vocab OK\n",my_rank);
  printf("[ %d ] SortVocab Start\n",my_rank);

  section_timer.restart();
  SortVocab();
  double sort_time = section_timer.duration();

  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }

  section_timer.restart();
  id2offset.resize(vocab_size);
  for(vertex_id_t vi = 0; vi < vocab_size; vi++){
    char* endptr;
    vertex_id_t nid = (vertex_id_t)strtoul(vocab[vi].word, &endptr, 10);
    id2offset[nid] = vi;
  }
  double id_map_time = section_timer.duration();

  double total_time = total_timer.duration();
  printf("[ %d ] Vocab timings (s): hash_init=%.3f, add_vocab=%.3f, sort=%.3f, id_map=%.3f, total=%.3f\n",
         my_rank, hash_init_time, add_vocab_time, sort_time, id_map_time, total_time);
}

void ReadVocab() {
  int a;
  long long i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (size_t hash_idx = 0; hash_idx < vocab_hash_size; ++hash_idx) vocab_hash[hash_idx] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&last_emb, 128, (long long)vocab_size * layer1_size * sizeof(float));
  memset(last_emb,0,(long long)vocab_size * layer1_size * sizeof(float));

  if (last_emb == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
  int hs_cuda = 0,neg_cuda=0;
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
      syn1[a * layer1_size + b] = 0;
    checkCUDAerr(cudaMalloc((void **)&d_syn1, (long long)vocab_size * layer1_size * sizeof(float)));
    hs_cuda = vocab_size * layer1_size * sizeof(float) / (1024 * 1024);
    checkCUDAerr(cudaMemcpy(d_syn1, syn1, (long long)vocab_size * layer1_size * sizeof(float), cudaMemcpyHostToDevice));
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
      syn1neg[a * layer1_size + b] = 0;
    checkCUDAerr(cudaMalloc((void **)&d_syn1, (long long)vocab_size * layer1_size * sizeof(float)));
    neg_cuda = vocab_size * layer1_size * sizeof(float) / (1024 * 1024);
    checkCUDAerr(cudaMemcpy(d_syn1, syn1neg, (long long)vocab_size * layer1_size * sizeof(float), cudaMemcpyHostToDevice));
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
  }
  checkCUDAerr(cudaMalloc((void **)&d_syn0, (long long)vocab_size * layer1_size * sizeof(float)));
  int syn0_cuda = vocab_size * layer1_size * sizeof(float) / (1024 * 1024);
  printf("[ %d ] [InitNet] CUDA malloc d_syn0 (%zu MB), hs=%zu MB, neg=%zu MB, total=%zu MB\n",
      my_rank,
      syn0_cuda,
      hs_cuda, neg_cuda, hs_cuda + neg_cuda + syn0_cuda);
  checkCUDAerr(cudaMemcpy(d_syn0, syn0, (long long)vocab_size * layer1_size * sizeof(float), cudaMemcpyHostToDevice));

  CreateBinaryTree();
}

void cbowKernel(int *d_sen, int *d_sent_len, float alpha, int cnt_sentence, int reduSize)
{
  int bDim = layer1_size;
  int gDim = cnt_sentence;
  switch(reduSize) {
    case 128: cbow_kernel<64><<<gDim, bDim>>>
              (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
               d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
               d_sen, d_sent_len, d_syn1, d_syn0);
              break;
    case 256: cbow_kernel<128><<<gDim, bDim>>>
              (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
               d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
               d_sen, d_sent_len, d_syn1, d_syn0);
              break;
    case 512: cbow_kernel<256><<<gDim, bDim>>>
              (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
               d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
               d_sen, d_sent_len, d_syn1, d_syn0);
              break;
    default: printf("Can't support on vector size = %lld\n", layer1_size);
             exit(1);
             break;
  }

}

void sgKernel(int *d_sen, int *d_sent_len, int *d_negSample, float alpha, int cnt_sentence, int reduSize)
{
  int bDim= layer1_size;
  int gDim= cnt_sentence;
  printf("reuseNeg: %d\n",reuseNeg);
  printf("layer1_size: %d\n",layer1_size);
  if (reuseNeg) { // A sentence share negative samples
    dim3 bDimNeg(32, negative+1, 1);
    switch(layer1_size) {
      case 1: __sgNegReuse<1><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 10: __sgNegReuse<10><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 20: __sgNegReuse<20><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 50: __sgNegReuse<50><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 100: __sgNegReuse<100><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 200: __sgNegReuse<200><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 300: __sgNegReuse<300><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      case 128: __sgNegReuse<128><<<gDim, bDimNeg>>>
                (window, layer1_size, negative, vocab_size, alpha,
                 d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
                break;
      default: printf("Can't support on vector size = %lld\n", layer1_size);
               exit(1);
               break;
    }
  } else {
    printf("no reuseNeg\n");
    switch(reduSize) {
      case 32: skip_gram_kernel<16><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      case 64: skip_gram_kernel<32><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      case 128: skip_gram_kernel<64><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                 d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                 d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      case 256: skip_gram_kernel<128><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                 d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                 d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      case 512: skip_gram_kernel<256><<<gDim, bDim>>>
                (window, layer1_size, negative, hs, table_size, vocab_size, alpha,
                 d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
                 d_sen, d_sent_len, d_syn1, d_syn0);
                break;
      default: printf("Can't support on vector size = %lld\n", layer1_size);
               exit(1);
               break;
    }
  }
}
volatile bool halt_sync = false;
volatile bool pause_sync = false;
void all_sync(){
    // REVERTED: Back to simple averaging - all nodes train same vocabulary
    checkCUDAerr(cudaMemcpy(syn0, d_syn0, (size_t)vocab_size * layer1_size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCUDAerr(cudaDeviceSynchronize());
    
    MPI_Allreduce(MPI_IN_PLACE, syn0, (size_t)vocab_size* layer1_size , MPI_FLOAT, MPI_SUM, MPI_EMB_COMM);
    
    for(size_t i = 0; i < (size_t) vocab_size * layer1_size;i++){
        syn0[i] /= num_procs;
    }
    
    checkCUDAerr(cudaMemcpy(d_syn0, syn0, (size_t)vocab_size * layer1_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCUDAerr(cudaDeviceSynchronize());
}
double sync_spend_time = 0.0f;
double training_time = 0.0;
double saving_embedding_time = 0.0;
double train_model_init_time = 0.0;
double pure_training_time = 0.0;
double train_wait_walk_time = 0.0;
double train_wait_sync_time = 0.0;
double corpus_read_time = 0.0;
double corpus_copy_h2d_time = 0.0;
double emb_copy_d2h_time = 0.0;
double training_loop_time_accum = 0.0;
double training_kernel_time = 0.0;

double total_file_read_time = 0.0;

void sync_embedding_func()
{
  // 单机模式下不需要同步
  if (num_procs == 1) {
    return;
  }
  
  Timer sync_timer;
  int wait_time = 8000;
  chrono::steady_clock::time_point syncTime = chrono::steady_clock::now() + chrono::milliseconds(wait_time);
  int sync_times = 1;
  while(!halt_sync)
  {
    sleep(wait_time/1000);
    // usleep(100000); // 100ms instead of 1s for faster sync frequency
    // if(true == pause_sync) {
    //   syncTime = chrono::steady_clock::now() + chrono::milliseconds(1000); // next sync time.
    // }
    unique_lock<std::mutex> lock(sync_mtx);
    //wait_until syncTime.
    sync_cv.wait_until(lock,syncTime);

    if(halt_sync == true) break;

    if(true == pause_sync) {
      syncTime = chrono::steady_clock::now() + chrono::milliseconds(wait_time); // next sync time.
      continue;
    }
    //block the training thread; 
    trainBlocked = true;

    sync_timer.restart();
    // all_sync();
    sync_spend_time += sync_timer.duration();

    // trainBlocked = false; // unblock the training thread.
    // sync_cv.notify_one(); // wake trainer
    // printf("[ %d ] Full sync completed, sync times: %d\n", my_rank, sync_times++);
    // syncTime = chrono::steady_clock::now() + chrono::milliseconds(1000); // next sync time
    // continue; // 继续下一次同步循环，跳过部分同步逻辑
    
    // copyFrom GPU, MPI, write back to GPU 
    //  No.1 pick up the sync id;
    int sync_node_num;
    vector<vertex_id_t> sync_vocab_id_array; // vocab id is not node id. It's the idx in the vpcab
    if(my_rank == 0)
    {
      vector<vertex_id_t> degree_range(vocab_size);
      degree_range[0] = 0;
      vertex_id_t n = 1;
      for(vertex_id_t vi = 1; vi < vocab_size;vi++){
        if(vocab[vi].cn != vocab[vi-1].cn){
          degree_range[n] = vi;
          n++;
        }
      }
      degree_range[n]  = vocab_size;
      random_device rd;
      mt19937 gen(rd());
      for(vertex_id_t v = 1; v <= n; v++)
      {
        uniform_int_distribution<>dis(degree_range[v-1],degree_range[v]-1); 
        sync_vocab_id_array.push_back(dis(gen));
      }
      sync_node_num = sync_vocab_id_array.size();
    }
    // broadcast sync id amount
    MPI_Bcast(&sync_node_num,1,get_mpi_data_type<int>(),0,MPI_EMB_COMM);
    if(my_rank != 0)
    {
      sync_vocab_id_array.resize(sync_node_num);
    }
    MPI_Bcast(sync_vocab_id_array.data(), sync_node_num, get_mpi_data_type<int>(), 0, MPI_EMB_COMM);
    // printf("[ %d ] sync_vocab_id_array size: %ld\n",my_rank,sync_vocab_id_array.size());
    // embedding buffer
    float *h_sync_emb_buffer = (float*)malloc(sync_node_num * layer1_size *sizeof(float));
    if(h_sync_emb_buffer == NULL){
      printf("[ %d ] ERROR. malloc h_sync_emb_buffer fail\n",my_rank);
    }
    // load specific embedding from GPU 
    for(vertex_id_t i =0; i < sync_vocab_id_array.size();i++){
      checkCUDAerr(cudaMemcpy(h_sync_emb_buffer+i *layer1_size,
            d_syn0 + sync_vocab_id_array[i]*layer1_size,
            layer1_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCUDAerr(cudaDeviceSynchronize());
    // synchronize
    MPI_Allreduce(MPI_IN_PLACE, h_sync_emb_buffer,sync_node_num * layer1_size , MPI_FLOAT, MPI_SUM, MPI_EMB_COMM);
    for(vertex_id_t i = 0; i < sync_node_num * layer1_size; i++){
      h_sync_emb_buffer[i] /= num_procs;
    }
    // write back to the GPU 
    for(vertex_id_t i =0; i < sync_vocab_id_array.size();i++){
      checkCUDAerr(cudaMemcpy( d_syn0 +sync_vocab_id_array[i]*layer1_size,
            h_sync_emb_buffer+i *layer1_size,
            layer1_size * sizeof(float), cudaMemcpyHostToDevice));
    }
    checkCUDAerr(cudaDeviceSynchronize());
    sync_spend_time += sync_timer.duration(); 
    syncTime = chrono::steady_clock::now() + chrono::milliseconds(wait_time); // next sync time.
    trainBlocked = false; // unblock the traing thread.
    sync_cv.notify_one(); // wake trainer
    printf("[ %d ] Syncing Times No.%d, sync %d nodes\n",my_rank,sync_times++, sync_node_num);
  }
}

LR *lr_scheduler;
void TrainModelThreadMemory(const corpus_t& corpus_data);  // Declaration for memory-based training
void TrainModelThread(string data_path)
{
  printf("[ p%d ]=====================Train file %s============\n",my_rank,data_path.c_str());
  long long word, word_count = 0, last_word_count = 0;
  long long local_iter = iter;

  // use in kernel
  int total_sent_len, reduSize= 32;
  int *sen, *sentence_length, *d_sen, *d_sent_len;
  sen = (int *)malloc(MAX_SENTENCE * 100 * sizeof(int));
  sentence_length = (int *)malloc((MAX_SENTENCE + 1) * sizeof(int));

  checkCUDAerr(cudaMalloc((void **)&d_sen, MAX_SENTENCE * 100 * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_sent_len, (MAX_SENTENCE + 1) * sizeof(int)));
  printf("[ %d ] CUDA malloc d_sen (%zu MB), d_sent_len (%zu MB), total (%zu MB)\n", my_rank,
      MAX_SENTENCE * 100 * sizeof(int) / (1024 * 1024),
      (MAX_SENTENCE + 1) * sizeof(int) / (1024 * 1024),
      (MAX_SENTENCE * 100 + (MAX_SENTENCE + 1)) * sizeof(int) / (1024 * 1024));
  int *negSample = (int *)malloc(MAX_SENTENCE * negative * sizeof(int));
  int *d_negSample;
  checkCUDAerr(cudaMalloc(&d_negSample, MAX_SENTENCE * negative * sizeof(int)));
  printf("[ %d ] CUDA malloc d_negSample (%zu MB)\n", my_rank,
      MAX_SENTENCE * negative * sizeof(int) / (1024 * 1024));
  Timer subsampling_precompute_timer;
  std::vector<uint16_t> subsample_thresholds = BuildSubsamplingThresholds(sample, train_words);
  double subsampling_precompute_time = subsampling_precompute_timer.duration();
  if (!subsample_thresholds.empty()) {
    printf("[ %d ] Subsampling thresholds built in %.6f seconds\n", my_rank, subsampling_precompute_time);
  }
  FastRandomState fast_rng(InitSeedForRank(my_rank, 0x1ULL));

#ifndef CXX_UNLIKELY
#define CXX_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


  while (reduSize < layer1_size) {
    reduSize *= 2;
  }

  clock_t now;
  strcpy(train_file,data_path.c_str());
  
  Timer file_read_timer;
  printf("[ %d ] Starting file read: %s\n", my_rank, train_file);
  
  FILE *fi = fopen(train_file, "r");
  if(fi == nullptr) {
    printf("open [%s] fail\n",data_path.c_str());
    throw std::runtime_error("Data file open fail");
  }
  fseek(fi, 0, SEEK_SET);

  while (1) {
    unique_lock<mutex> lock(sync_mtx);
    sync_cv.wait(lock,[]{return !trainBlocked;});// 没有阻塞的时候才训练
                                                              
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Words/sec: %.2fk  ", 13, alpha,
            // word_count_actual / (float)(iter * train_words + 1) * 100,
            word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      // alpha = starting_alpha * (1 - word_count / (float)(iter * train_words + 1));
      // if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    total_sent_len = 0;
    sentence_length[0] = 0;
    int cnt_sentence = 0;

    while (cnt_sentence < MAX_SENTENCE) {                           // Read words
      int temp_sent_len = 0;
      char tSentence[MAX_SENTENCE_LENGTH];
      char *wordTok;
      if (feof(fi)) break;
      fgets(tSentence, MAX_SENTENCE_LENGTH + 1, fi);
      wordTok = strtok(tSentence, " \n\r\t");
      while(1) {
        if (wordTok == NULL) {
          word_count++;
          break;
        }
        word = SearchVocab(wordTok);
        wordTok = strtok(NULL, " \n\r\t");
        if (word == -1) continue;
        word_count++;
        if (word == 0) {
          word_count++;
          break;
        }
        if (!subsample_thresholds.empty()) {
          const uint16_t keep_threshold = subsample_thresholds[word];
          if (CXX_UNLIKELY(keep_threshold < kFullKeepThreshold)) {
            uint16_t random16 = fast_rng.Next16();
            if (random16 > keep_threshold) {
              continue;
            }
          }
        }
        sen[total_sent_len] = word;
        total_sent_len++;
        temp_sent_len++;
        if (temp_sent_len >= MAX_SENTENCE_LENGTH) break;
      }
      if (word == 0) {
        word_count++;
        break;
      }
      if (temp_sent_len >= MAX_SENTENCE_LENGTH) break;

      cnt_sentence++;
      sentence_length[cnt_sentence] = total_sent_len;
      if (total_sent_len >= (MAX_SENTENCE - 1) * 20) break;
    }

    if (feof(fi) || (word_count > train_words)) {                   // Initialize for iteration
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      for (int i=0; i<MAX_SENTENCE+1; i++)
        sentence_length[i] = 0;
      total_sent_len = 0;
      fseek(fi, 0, SEEK_SET);
      continue;
    }

    // Negative sampling in advance. A sentence shares negative samples
    for (int i = 0; i < cnt_sentence * negative; i++) {
      uint32_t randd = fast_rng.Next32();
      int tempSample = table[randd % table_size];
      if (tempSample == 0) {
        negSample[i] = static_cast<int>(randd % (vocab_size - 1)) + 1;
      } else {
        negSample[i] = tempSample;
      }
    }
    checkCUDAerr(cudaMemcpy(d_negSample, negSample, cnt_sentence * negative * sizeof(int), cudaMemcpyHostToDevice));
    cudaError_t cet = cudaMemcpy(d_sen, sen, total_sent_len * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaSuccess != cet)
    {
      printf("%s %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(cet));
      printf("copy size: %zu \n",total_sent_len*sizeof(int));
      exit(0);
    }
    checkCUDAerr(cudaMemcpy(d_sent_len, sentence_length, (cnt_sentence + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (cbow)
      cbowKernel(d_sen, d_sent_len, alpha, cnt_sentence, reduSize);
    else
    {
      // printf("[ %d ] sgKernel Invoke\n",my_rank);
      sgKernel(d_sen, d_sent_len, d_negSample, alpha, cnt_sentence, reduSize);
    }
  }
  cudaDeviceSynchronize();
  checkCUDAerr(cudaMemcpy(syn0, d_syn0, vocab_size * layer1_size * sizeof(float), cudaMemcpyDeviceToHost));

  double file_read_time = file_read_timer.duration();
  total_file_read_time += file_read_time;
  printf("[ %d ] File read completed in %lf seconds (total: %lf)\n", my_rank, file_read_time, total_file_read_time);
  
  fclose(fi);

  // free memory
  free(sen);
  free(sentence_length);
  free(negSample);
  cudaFree(d_sen);
  cudaFree(d_sent_len);
  cudaFree(d_negSample);
}

// New function to train directly with memory corpus data (no disk I/O)
void TrainModelThreadMemory(const corpus_t& corpus_data)
{
  printf("[ p%d ]=====================Train memory corpus (size: %zu)============\n", my_rank, corpus_data.size());
  long long word, word_count = 0, last_word_count = 0;
  long long local_iter = iter;

  // use in kernel
  Timer cuda_mem_init_timer;
  double init_time = 0.0;
  int total_sent_len, reduSize = 32;
  int *sen, *sentence_length, *d_sen, *d_sent_len;
  sen = (int *)malloc(MAX_SENTENCE * 100 * sizeof(int));
  sentence_length = (int *)malloc((MAX_SENTENCE + 1) * sizeof(int));

  checkCUDAerr(cudaMalloc((void **)&d_sen, MAX_SENTENCE * 100 * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_sent_len, (MAX_SENTENCE + 1) * sizeof(int)));
  printf("[ %d ] CUDA malloc d_sen (%zu MB), d_sent_len (%zu MB), total (%zu MB)\n", my_rank,
      MAX_SENTENCE * 100 * sizeof(int) / (1024 * 1024),
      (MAX_SENTENCE + 1) * sizeof(int) / (1024 * 1024),
      (MAX_SENTENCE * 100 + (MAX_SENTENCE + 1)) * sizeof(int) / (1024 * 1024));
  int *negSample = (int *)malloc(MAX_SENTENCE * negative * sizeof(int));
  int *d_negSample;
  checkCUDAerr(cudaMalloc(&d_negSample, MAX_SENTENCE * negative * sizeof(int)));
  printf("[ %d ] CUDA malloc d_negSample (%zu MB)\n", my_rank,
      MAX_SENTENCE * negative * sizeof(int) / (1024 * 1024));
  Timer subsampling_precompute_timer;
  std::vector<uint16_t> subsample_thresholds = BuildSubsamplingThresholds(sample, train_words);
  double subsampling_precompute_time = subsampling_precompute_timer.duration();
  if (!subsample_thresholds.empty()) {
    printf("[ %d ] Subsampling thresholds built in %.6f seconds\n", my_rank, subsampling_precompute_time);
  }
  FastRandomState fast_rng(InitSeedForRank(my_rank, 0x2ULL));

  while (reduSize < layer1_size) {
    reduSize *= 2;
  }
  init_time = cuda_mem_init_timer.duration();
  printf("[ %d ] CUDA Memory setup completed in %lf seconds(host to device)\n", my_rank, init_time);
  clock_t now;
  start = clock();

  // Process corpus data directly from memory instead of reading from file
  size_t corpus_index = 0;
  double total_corpus_read_time = 0.0;
  double total_copy_time = 0.0;
  double total_kernel_time = 0.0;
  double total_training_loop_time = 0.0;

  // Process sentences from memory corpus
  Timer corpus_read_timer;
  Timer h2d_copy_timer;
  Timer kernel_timer;
  Timer total_training_timer;
  // corpus read_timer breakdown
  Timer sequence_processing_timer;
  Timer vocab_conversion_timer;
  Timer subsampling_timer;
  Timer subsampling_prob_timer;
  Timer subsampling_rng_timer;
  Timer subsampling_compare_timer;
  Timer sentence_building_timer;
  Timer seq_other_timer;
  Timer wait_sync_timer;
  Timer negative_sampling_timer;
  // breakdown
  double sequence_processing_time = 0.0;
  double total_vocab_conversion_time = 0.0;
  double total_subsampling_time = 0.0;
  double total_subsampling_prob_time = 0.0;
  double total_subsampling_rng_time = 0.0;
  double total_subsampling_compare_time = 0.0;
  double total_sentence_building_time = 0.0;
  double total_sequence_other_time = 0.0;
  double wait_sync_time = 0.0;
  double negative_sampling_time = 0.0;
  
  while (corpus_index < corpus_data.size()) {
    total_training_timer.restart();
    corpus_read_timer.restart();
    // 多进程环境下才需要等待同步
    // OPTIMIZATION: No need to wait for sync during training
    if (num_procs > 1) {
      wait_sync_timer.restart();
      unique_lock<mutex> lock(sync_mtx);
      sync_cv.wait(lock,[]{return !trainBlocked;});// 没有阻塞的时候才训练
      wait_sync_time += wait_sync_timer.duration();
    }
                                                              
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Words/sec: %.2fk  ", 13, alpha,
            word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
    }
    
    total_sent_len = 0;
    sentence_length[0] = 0;
    int cnt_sentence = 0;


    sequence_processing_timer.restart();
    while (cnt_sentence < MAX_SENTENCE && corpus_index < corpus_data.size()) {
      const auto& sequence = corpus_data[corpus_index];
      int temp_sent_len = 0;
      seq_other_timer.restart();
      double iter_vocab_time = 0.0;
      double iter_subsampling_time = 0.0;
      double iter_sentence_building_time = 0.0;
      for (auto vertex_id : sequence) {
        vocab_conversion_timer.restart();
        word = id2offset[vertex_id];  // Convert vertex ID to vocab index
        double vocab_time = vocab_conversion_timer.duration();
        total_vocab_conversion_time += vocab_time;
        iter_vocab_time += vocab_time;
        if (word == -1) {
          continue;
        }
        word_count++;
        if (word == 0) {
          word_count++;  // Match file mode behavior
          break;  // End of sentence
        }

        if (!subsample_thresholds.empty()) {
          const uint16_t keep_threshold = subsample_thresholds[word];
          if (CXX_UNLIKELY(keep_threshold < kFullKeepThreshold)) {
            subsampling_timer.restart();
            subsampling_prob_timer.restart();
            double prob_time = subsampling_prob_timer.duration();
            total_subsampling_prob_time += prob_time;

            subsampling_rng_timer.restart();
            uint16_t random16 = fast_rng.Next16();
            double rng_time = subsampling_rng_timer.duration();
            total_subsampling_rng_time += rng_time;

            subsampling_compare_timer.restart();
            bool discard_token = random16 > keep_threshold;
            double compare_time = subsampling_compare_timer.duration();
            total_subsampling_compare_time += compare_time;

            double subsampling_time = subsampling_timer.duration();
            total_subsampling_time += subsampling_time;
            iter_subsampling_time += subsampling_time;
            if (discard_token) continue;
          }
        }

        sentence_building_timer.restart();
        sen[total_sent_len] = word;
        total_sent_len++;
        temp_sent_len++;
        double sentence_time = sentence_building_timer.duration();
        total_sentence_building_time += sentence_time;
        iter_sentence_building_time += sentence_time;
        if (temp_sent_len >= MAX_SENTENCE_LENGTH) break;
      }
      double iter_total_time = seq_other_timer.duration();
      double iter_other_time = iter_total_time - iter_vocab_time - iter_subsampling_time - iter_sentence_building_time;
      if (iter_other_time < 0.0) {
        iter_other_time = 0.0;
      }
      total_sequence_other_time += iter_other_time;
      // Check if sentence ended with word 0, matching file mode behavior
      if (word == 0) {
        word_count++;
      }

      cnt_sentence++;
      sentence_length[cnt_sentence] = total_sent_len;
      corpus_index++;
      if (total_sent_len >= (MAX_SENTENCE - 1) * 20) break;
    }
    sequence_processing_time += sequence_processing_timer.duration();

    if (cnt_sentence == 0) break;

    // Generate negative samples (match file mode behavior)
    negative_sampling_timer.restart();
    for (int i = 0; i < cnt_sentence * negative; i++) {
      uint32_t randd = fast_rng.Next32();
      int tempSample = table[randd % table_size];
      if (tempSample == 0) {
        negSample[i] = static_cast<int>(randd % (vocab_size - 1)) + 1;
      } else {
        negSample[i] = tempSample;
      }
    }
    negative_sampling_time += negative_sampling_timer.duration();

    double batch_read_time = corpus_read_timer.duration();
    total_corpus_read_time += batch_read_time;
    // Copy data to GPU and run training
    h2d_copy_timer.restart();
    checkCUDAerr(cudaMemcpy(d_sen, sen, total_sent_len * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDAerr(cudaMemcpy(d_sent_len, sentence_length, (cnt_sentence + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDAerr(cudaMemcpy(d_negSample, negSample, cnt_sentence * negative * sizeof(int), cudaMemcpyHostToDevice));
    double copy_time = h2d_copy_timer.duration();
    printf("[ %d ] Corpus data copied to GPU in %lf seconds \n", my_rank, copy_time);
    total_copy_time += copy_time;

    kernel_timer.restart();
    if (cbow) {
      cbowKernel(d_sen, d_sent_len, alpha, cnt_sentence, reduSize);
    } else {
      sgKernel(d_sen, d_sent_len, d_negSample, alpha, cnt_sentence, reduSize);
    }
    double kernel_time = kernel_timer.duration();
    printf("[ %d ] Corpus kernel executed in %lf seconds \n", my_rank, kernel_time);
    total_kernel_time += kernel_time;
    total_training_loop_time += total_training_timer.duration();
  }
  cudaDeviceSynchronize();

  printf("[ %d ] Total corpus read completed in %lf seconds \n", my_rank, total_corpus_read_time);
  corpus_read_time += total_corpus_read_time;
  printf("[ %d ] Total copy host to GPU completed in %lf seconds \n", my_rank, total_copy_time);
  corpus_copy_h2d_time += total_copy_time;
  printf("[ %d ] total kernel Corpus training completed in %lf seconds \n", my_rank, total_kernel_time);
  training_kernel_time += total_kernel_time;
  printf("[ %d ] Total training loop time accumulated %lf seconds \n", my_rank, total_training_loop_time);
  training_loop_time_accum += total_training_loop_time;
  Timer d2h_timer;
  checkCUDAerr(cudaMemcpy(syn0, d_syn0, vocab_size * layer1_size * sizeof(float), cudaMemcpyDeviceToHost));
  emb_copy_d2h_time += d2h_timer.duration();
  printf("[ %d ] Copy embedding GPU to host completed in %lf seconds \n", my_rank, emb_copy_d2h_time);

  // Detailed corpus read time breakdown
  printf("\n[ %d ] === Corpus read time Detailed Time Breakdown ===\n", my_rank);
  printf("[ %d ] Total corpus read completed in %lf seconds \n", my_rank, total_corpus_read_time);
  printf("[ %d ] - Sequence processing: %.6f seconds (%.2f%%)\n", my_rank, sequence_processing_time, sequence_processing_time / total_corpus_read_time * 100);
  printf("[ %d ]    - Vocab ID conversion: %.6f seconds (%.2f%%)\n", my_rank, total_vocab_conversion_time, total_vocab_conversion_time / total_corpus_read_time * 100);
  printf("[ %d ]    - Subsampling computation: %.6f seconds (%.2f%%)\n", my_rank, total_subsampling_time, total_subsampling_time / total_corpus_read_time * 100);
  if (total_subsampling_time > 0.0) {
    printf("[ %d ]      * Prob calc          : %.6f seconds (%.2f%% of subsampling)\n", my_rank, total_subsampling_prob_time, total_subsampling_prob_time / total_subsampling_time * 100);
    printf("[ %d ]      * RNG               : %.6f seconds (%.2f%% of subsampling)\n", my_rank, total_subsampling_rng_time, total_subsampling_rng_time / total_subsampling_time * 100);
    printf("[ %d ]      * Compare threshold : %.6f seconds (%.2f%% of subsampling)\n", my_rank, total_subsampling_compare_time, total_subsampling_compare_time / total_subsampling_time * 100);
    double total_subsampling_other_time = total_subsampling_time - total_subsampling_prob_time - total_subsampling_rng_time - total_subsampling_compare_time;
    if (total_subsampling_other_time < 0.0) {
      total_subsampling_other_time = 0.0;
    }
    printf("[ %d ]      * Other subsample ops: %.6f seconds (%.2f%% of subsampling)\n", my_rank, total_subsampling_other_time, total_subsampling_other_time / total_subsampling_time * 100);
  }
  printf("[ %d ]    - Sentence building: %.6f seconds (%.2f%%)\n", my_rank, total_sentence_building_time, total_sentence_building_time / total_corpus_read_time * 100);
  printf("[ %d ]    - Other sequence ops: %.6f seconds (%.2f%%)\n", my_rank, total_sequence_other_time, total_sequence_other_time / total_corpus_read_time * 100);
  printf("[ %d ] - wait syncing : %.6f seconds (%.2f%%)\n", my_rank, wait_sync_time, wait_sync_time / total_corpus_read_time * 100);
  printf("[ %d ] - Negative sampling: %.6f seconds (%.2f%%)\n", my_rank, negative_sampling_time, negative_sampling_time / total_corpus_read_time * 100);

  // free memory
  free(sen);
  free(sentence_length);
  free(negSample);
  cudaFree(d_sen);
  cudaFree(d_sent_len);
  cudaFree(d_negSample);
  train_wait_sync_time += wait_sync_time;
}

vector<vertex_id_t> g_v_degree;

void myIntersectition(const vector<vertex_id_t>& v1,const vector<vertex_id_t>& v2,vector<vertex_id_t>& v_intersection)
{
    int p1=0;
    int p2=0;
    int v1_sz = v1.size();
    int v2_sz = v2.size();
    const vector<vertex_id_t>*long_v;
    const vector<vertex_id_t>*short_v;
    if(v1_sz>v2_sz){
        long_v=&v1;
        short_v=&v2;
    }else{
        long_v=&v2;
        short_v=&v1;
    }
    int max_sz = max(v1_sz,v2_sz);
    int min_sz = min(v1_sz,v2_sz);
    
    int begin = 0;
    if(v1.empty()||v2.empty())return;
    if((*long_v)[max_sz-1]<(*short_v)[0]||((*long_v)[0]>(*short_v)[min_sz-1]))return;
    while(p1<max_sz&&p2<min_sz){
        int offset = 1;
        int last_p = offset;
        if((*short_v)[p2]<((*long_v)[p1])){
            p2++;
            continue;
        }
        while((*long_v)[p1+offset-1]<(*short_v)[p2]){
            offset=offset*2;
            last_p = p1+offset<max_sz?offset:max_sz-p1;
            if(p1+offset>=max_sz)break;
        }
        if((*long_v)[max_sz-1]<(*short_v)[p2]){
            p2++;
            break;
        }
        auto iter = lower_bound(long_v->begin()+(p1+offset/2),long_v->begin()+p1+last_p,(*short_v)[p2]); // 如果在 区间找到了
        int t = iter - long_v->begin();
        if(*iter==(*short_v)[p2]){
            v_intersection.push_back((*short_v)[p2]);
            p2++;
            p1=t++;
        }else{
            p2++;
            p1=(p1+offset/2);
        }
    };   
   
}

float cos_sim(float* v1,float* v2, int dim){
  float dot_product = 0.0;
  float v1_l2 = 0.0, v2_l2 = 0.0;
  for(int d = 0; d < dim; d++){
    dot_product += v1[d] * v2[d];
    v1_l2 += v1[d] * v1[d];
    v2_l2 += v2[d] * v2[d];
  }
  v1_l2 = sqrt(v1_l2);
  v2_l2 = sqrt(v2_l2);
  return dot_product/(v1_l2 * v2_l2);
}
__global__ void vector_cosine_similarity_kernel(
    const float* d_A,
    const float* d_B,
    float* d_results,
    int vector_length
    ){
  int i = blockIdx.x; // 第 i 个向量对（ 2～29 )
  int tid = threadIdx.x; // 线程号（0～blockDim.x -1 )
  extern __shared__ float s_data[]; // 动态共享内存
  float* s_dot = s_data;
  float* s_A2 = &s_data[blockDim.x];
  float* s_B2 = &s_data[2 * blockDim.x];

  // 初始化
  float a = 0.0f, b = 0.0f;
  if(tid < vector_length) {
    a = d_A[i * vector_length + tid];
    b = d_B[i * vector_length + tid];
  }
  // 计算点积平方和
  s_dot[tid] = a * b;
  s_A2[tid] = a * a;
  s_B2[tid] = b * b;

  __syncthreads();

  // 规约求和（树装规约）
  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    if(tid < stride){
      s_dot[tid] += s_dot[tid + stride];
      s_A2[tid] += s_A2[tid + stride];
      s_B2[tid] += s_B2[tid + stride];
    }
    __syncthreads();
  }
  // 计算余弦相似度（仅由线程0完成）
  if(tid == 0) {
    float sum_dot = s_dot[0];
    float sum_A2 = s_A2[0];
    float sum_B2 = s_B2[0];

    float norm_A = sqrtf(sum_A2);
    float norm_B = sqrtf(sum_B2);

    // 处理零向量
    if (norm_A == 0 || norm_B == 0){
      d_results[i] = 0.0f;
    } else {
      d_results[i] = sum_dot / (norm_A * norm_B);
    }
  }
}

// ========================================================
// GPU-SIDE DIRECT INDEXING OPTIMIZATION
// ========================================================
// New kernel that directly indexes embeddings on GPU to eliminate memory copies
// This kernel processes multiple node-neighbor pairs in a single batch
__global__ void batch_similarity_direct_index_kernel(
    const float* d_syn0,           // All embeddings on GPU 
    const vertex_id_t* d_node_ids, // Node IDs to evaluate
    const vertex_id_t* d_neighbor_ids, // Corresponding neighbor IDs
    const int* d_eval_counts,      // Number of evaluations per node
    const int* d_eval_offsets,     // Offset for each node's evaluations
    float* d_results,              // Output similarity results
    int vector_length,             // Embedding dimension (layer1_size)
    int total_evaluations,         // Total number of node-neighbor pairs
    const vertex_id_t* d_id2offset // ID to offset mapping
) {
    int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (eval_idx >= total_evaluations) return;
    
    // Get node and neighbor IDs for this evaluation
    vertex_id_t node_id = d_node_ids[eval_idx];
    vertex_id_t neighbor_id = d_neighbor_ids[eval_idx];
    
    // Convert IDs to embedding offsets
    vertex_id_t node_offset = d_id2offset[node_id];
    vertex_id_t neighbor_offset = d_id2offset[neighbor_id];
    
    // Calculate cosine similarity directly from GPU embeddings
    float dot_product = 0.0f;
    float node_norm = 0.0f;
    float neighbor_norm = 0.0f;
    
    // Compute dot product and norms
    for (int i = 0; i < vector_length; i++) {
        float node_val = d_syn0[node_offset * vector_length + i];
        float neighbor_val = d_syn0[neighbor_offset * vector_length + i];
        
        dot_product += node_val * neighbor_val;
        node_norm += node_val * node_val;
        neighbor_norm += neighbor_val * neighbor_val;
    }
    
    // Calculate cosine similarity
    node_norm = sqrtf(node_norm);
    neighbor_norm = sqrtf(neighbor_norm);
    
    if (node_norm == 0.0f || neighbor_norm == 0.0f) {
        d_results[eval_idx] = 0.0f;
    } else {
        d_results[eval_idx] = dot_product / (node_norm * neighbor_norm);
    }
}

// Optimized kernel using shared memory and warp-level reductions for better performance
__global__ void batch_similarity_direct_index_optimized_kernel(
    const float* d_syn0,
    const vertex_id_t* d_node_ids,
    const vertex_id_t* d_neighbor_ids, 
    float* d_results,
    int vector_length,
    int total_evaluations,
    const vertex_id_t* d_id2offset
) {
    int eval_idx = blockIdx.x;  // One block per evaluation pair
    int tid = threadIdx.x;      // Thread within block
    
    if (eval_idx >= total_evaluations) return;
    
    extern __shared__ float s_data[];
    float* s_dot = s_data;
    float* s_node_norm = &s_data[blockDim.x];
    float* s_neighbor_norm = &s_data[2 * blockDim.x];
    
    // Get node and neighbor offsets
    vertex_id_t node_offset = d_id2offset[d_node_ids[eval_idx]];
    vertex_id_t neighbor_offset = d_id2offset[d_neighbor_ids[eval_idx]];
    
    // Initialize shared memory
    float node_val = 0.0f, neighbor_val = 0.0f;
    if (tid < vector_length) {
        node_val = d_syn0[node_offset * vector_length + tid];
        neighbor_val = d_syn0[neighbor_offset * vector_length + tid];
    }
    
    s_dot[tid] = node_val * neighbor_val;
    s_node_norm[tid] = node_val * node_val;
    s_neighbor_norm[tid] = neighbor_val * neighbor_val;
    
    __syncthreads();
    
    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
            s_node_norm[tid] += s_node_norm[tid + stride];
            s_neighbor_norm[tid] += s_neighbor_norm[tid + stride];
        }
        __syncthreads();
    }
    
    // Calculate final similarity (thread 0 only)
    if (tid == 0) {
        float dot_product = s_dot[0];
        float node_norm = sqrtf(s_node_norm[0]);
        float neighbor_norm = sqrtf(s_neighbor_norm[0]);
        
        if (node_norm == 0.0f || neighbor_norm == 0.0f) {
            d_results[eval_idx] = 0.0f;
        } else {
            d_results[eval_idx] = dot_product / (node_norm * neighbor_norm);
        }
    }
}
// ========================================================

// Function declarations for batch processing
std::vector<float> batch_node_neighbor_average_cos_sim_chunked(
    const std::vector<vertex_id_t>& all_nodes,
    myEdgeContainer* csr,
    float* d_A_batch, 
    float* d_B_batch, 
    float* d_results_batch,
    int batch_size);

std::vector<float> batch_process_single_chunk(
    const std::vector<vertex_id_t>& all_nodes,
    size_t chunk_start,
    size_t chunk_size,
    myEdgeContainer* csr,
    float* d_A_batch,
    float* d_B_batch, 
    float* d_results_batch);

// ========================================================
// GPU-SIDE DIRECT INDEXING WRAPPER FUNCTION
// ========================================================
// New optimized function that eliminates memory copy bottleneck
// Maintains same interface as batch_node_neighbor_average_cos_sim_chunked
std::vector<float> batch_node_neighbor_direct_index(
    const std::vector<vertex_id_t>& all_nodes,
    myEdgeContainer* csr,
    int batch_size,
    bool use_optimized_kernel = true);
// ========================================================

float node_neighbour_average_cos_sim(vertex_id_t v_id,myEdgeContainer*csr,float* d_A,float* d_B,float* d_results){
  float sum_cos_sim = 0.0f;
  int nei_n = csr->adj_lists[v_id].end - csr->adj_lists[v_id].begin;
  //  Get neighbour set; If neighbour amout > 30,then chose thiry randomly
  vector<vertex_id_t> neighbor_set;
  for(auto it = csr->adj_lists[v_id].begin; it < csr->adj_lists[v_id].end; it++){
    neighbor_set.push_back(it->neighbour);
  }
  int evaluate_num = neighbor_set.size();
  if(evaluate_num > EVALUATION_NEIGHBOUR_NUM){
    evaluate_num = EVALUATION_NEIGHBOUR_NUM;
    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(neighbor_set.begin(), neighbor_set.end(), g);
  }

  for(int i =0; i < evaluate_num; i++){
    vertex_id_t nei = neighbor_set[i];
    vertex_id_t v_1 = id2offset[v_id];
    vertex_id_t v_2 = id2offset[nei];
    checkCUDAerr(
        cudaMemcpy(d_A + i*layer1_size, d_syn0+v_1*layer1_size, layer1_size*sizeof(float), cudaMemcpyDeviceToDevice);
        );
    checkCUDAerr(
        cudaMemcpy(d_B + i*layer1_size, d_syn0+v_2*layer1_size, layer1_size*sizeof(float), cudaMemcpyDeviceToDevice);
        );
    // float sim = cos_sim(syn0+v_1 * layer1_size, syn0+v_2*layer1_size,layer1_size);
    // sum_cos_sim += sim;
  }
  int threadsPerBlock = layer1_size > 200 ? 256 : 128;
  int blocks = evaluate_num;
  size_t sharedMemSize = 3 * threadsPerBlock * sizeof(float);
  vector_cosine_similarity_kernel<<<blocks,threadsPerBlock,sharedMemSize>>>(d_A, d_B, d_results, layer1_size);
  cudaDeviceSynchronize();
  float* h_results = new float[evaluate_num];
  cudaMemcpy(h_results,d_results,evaluate_num*sizeof(float),cudaMemcpyDeviceToHost);

  // // Find min and max values for normalization
  // float min_val = h_results[0];
  // float max_val = h_results[0];
  // for(int i = 1; i < evaluate_num; i++){
  //   if(h_results[i] < min_val) min_val = h_results[i];
  //   if(h_results[i] > max_val) max_val = h_results[i];
  // }
  
  // // Apply min-max normalization and sum
  // float cuda_cos_sim = 0.0f;
  // float range = max_val - min_val;
  // for(int i = 0; i < evaluate_num; i++){
  //   float normalized_val = (range > 0) ? (h_results[i] - min_val) / range : 0.1f;
  //   cuda_cos_sim += normalized_val;
  // }
  
  // cuda_cos_sim /= evaluate_num;
  
  // delete[] h_results;
  
  // return  cuda_cos_sim;

  float cuda_cos_sim = 0.0f;
  for(int i =0;i < evaluate_num; i++){
    cuda_cos_sim  += h_results[i];
  }
  
  cuda_cos_sim /= evaluate_num;
  float cpu_cos_sim = sum_cos_sim / nei_n;
  // printf("[ %d ] vid: %u cuda: %f cpu: %f\n",my_rank,v_id,cuda_cos_sim,cpu_cos_sim);
  return  cuda_cos_sim;
}

// Batch version of node_neighbour_average_cos_sim for improved performance
std::vector<float> batch_node_neighbor_average_cos_sim_chunked(
    const std::vector<vertex_id_t>& all_nodes,
    myEdgeContainer* csr,
    float* d_A_batch, 
    float* d_B_batch, 
    float* d_results_batch,
    int batch_size) {
    std::vector<float> all_results;
    all_results.reserve(all_nodes.size());
    
    printf("[ %d ] Batch processing %zu nodes in chunks of %d\n", 
           my_rank, all_nodes.size(), batch_size);
    
    // Process in chunks to manage GPU memory
    for(size_t chunk_start = 0; chunk_start < all_nodes.size(); chunk_start += batch_size) {
        size_t chunk_end = std::min(chunk_start + batch_size, all_nodes.size());
        size_t chunk_size = chunk_end - chunk_start;
        
        printf("[ %d ] Processing chunk %zu-%zu (%zu nodes)\n", 
               my_rank, chunk_start, chunk_end-1, chunk_size);
        
        // Process single chunk
        std::vector<float> chunk_results = batch_process_single_chunk(
            all_nodes, chunk_start, chunk_size, csr, d_A_batch, d_B_batch, d_results_batch);
        
        // Append results
        all_results.insert(all_results.end(), chunk_results.begin(), chunk_results.end());
    }
    
    printf("[ %d ] Batch processing completed, processed %zu nodes\n", my_rank, all_results.size());
    return all_results;
}

// Process a single chunk of nodes
std::vector<float> batch_process_single_chunk(
    const std::vector<vertex_id_t>& all_nodes,
    size_t chunk_start,
    size_t chunk_size,
    myEdgeContainer* csr,
    float* d_A_batch,
    float* d_B_batch, 
    float* d_results_batch) {
    
    std::vector<float> chunk_results;
    chunk_results.reserve(chunk_size);
    
    int total_evaluations = 0;
    std::vector<int> node_eval_counts(chunk_size);  // Store evaluation count per node
    
    // Step 1: Collect all neighbor pairs and copy to GPU memory
    for(size_t i = 0; i < chunk_size; i++) {
        vertex_id_t v_id = all_nodes[chunk_start + i];
        
        // Get neighbor set
        std::vector<vertex_id_t> neighbor_set;
        for(auto it = csr->adj_lists[v_id].begin; it < csr->adj_lists[v_id].end; it++) {
            neighbor_set.push_back(it->neighbour);
        }
        
        // Limit evaluation number
        int evaluate_num = neighbor_set.size();
        if(evaluate_num > EVALUATION_NEIGHBOUR_NUM) {
            evaluate_num = EVALUATION_NEIGHBOUR_NUM;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(neighbor_set.begin(), neighbor_set.end(), g);
        }
        
        node_eval_counts[i] = evaluate_num;
        
        // Copy embeddings to GPU memory
        for(int j = 0; j < evaluate_num; j++) {
            vertex_id_t nei = neighbor_set[j];
            vertex_id_t v_1 = id2offset[v_id];
            vertex_id_t v_2 = id2offset[nei];
            
            int eval_idx = total_evaluations + j;
            checkCUDAerr(
                cudaMemcpy(d_A_batch + eval_idx * layer1_size, 
                          d_syn0 + v_1 * layer1_size, 
                          layer1_size * sizeof(float), 
                          cudaMemcpyDeviceToDevice)
            );
            checkCUDAerr(
                cudaMemcpy(d_B_batch + eval_idx * layer1_size, 
                          d_syn0 + v_2 * layer1_size, 
                          layer1_size * sizeof(float), 
                          cudaMemcpyDeviceToDevice)
            );
        }
        
        total_evaluations += evaluate_num;
    }
    
    // Step 2: Launch CUDA kernel for all evaluations in this chunk
    if(total_evaluations > 0) {
        int threadsPerBlock = layer1_size > 200 ? 256 : 128;
        int blocks = total_evaluations;
        size_t sharedMemSize = 3 * threadsPerBlock * sizeof(float);
        
        vector_cosine_similarity_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
            d_A_batch, d_B_batch, d_results_batch, layer1_size);
        cudaDeviceSynchronize();
        
        // Step 3: Copy results back and compute averages
        float* h_results = new float[total_evaluations];
        cudaMemcpy(h_results, d_results_batch, total_evaluations * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Step 4: Compute average similarity for each node
        int result_offset = 0;
        for(size_t i = 0; i < chunk_size; i++) {
            int evaluate_num = node_eval_counts[i];
            
            if(evaluate_num == 0) {
                chunk_results.push_back(0.0f);
                continue;
            }
            
            float sum_similarity = 0.0f;
            for(int j = 0; j < evaluate_num; j++) {
                sum_similarity += h_results[result_offset + j];
            }
            
            float avg_similarity = sum_similarity / evaluate_num;
            chunk_results.push_back(avg_similarity);
            
            result_offset += evaluate_num;
        }
        
        delete[] h_results;
    }
    
    return chunk_results;
}

// ========================================================
// GPU-SIDE DIRECT INDEXING IMPLEMENTATION
// ========================================================
std::vector<float> batch_node_neighbor_direct_index(
    const std::vector<vertex_id_t>& all_nodes,
    myEdgeContainer* csr,
    int batch_size,
    bool use_optimized_kernel) {
    
    std::vector<float> all_results;
    all_results.reserve(all_nodes.size());
    
    printf("[ %d ] GPU Direct Index: Processing %zu nodes (batch_size=%d, optimized=%s)\n", 
           my_rank, all_nodes.size(), batch_size, use_optimized_kernel ? "true" : "false");
    
    // Process in chunks to manage GPU memory
    for(size_t chunk_start = 0; chunk_start < all_nodes.size(); chunk_start += batch_size) {
        size_t chunk_end = std::min(chunk_start + batch_size, all_nodes.size());
        size_t chunk_size = chunk_end - chunk_start;
        
        printf("[ %d ] GPU Direct: Processing chunk %zu-%zu (%zu nodes)\n", 
               my_rank, chunk_start, chunk_end-1, chunk_size);
        
        // Step 1: Collect all node-neighbor pairs for this chunk
        std::vector<vertex_id_t> node_ids;
        std::vector<vertex_id_t> neighbor_ids;
        std::vector<int> node_eval_counts(chunk_size);
        std::vector<int> eval_offsets(chunk_size);
        
        int total_evaluations = 0;
        
        for(size_t i = 0; i < chunk_size; i++) {
            vertex_id_t v_id = all_nodes[chunk_start + i];
            
            // Get neighbor set
            std::vector<vertex_id_t> neighbor_set;
            for(auto it = csr->adj_lists[v_id].begin; it < csr->adj_lists[v_id].end; it++) {
                neighbor_set.push_back(it->neighbour);
            }
            
            // Limit evaluation number
            int evaluate_num = neighbor_set.size();
            if(evaluate_num > EVALUATION_NEIGHBOUR_NUM) {
                evaluate_num = EVALUATION_NEIGHBOUR_NUM;
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(neighbor_set.begin(), neighbor_set.end(), g);
            }
            
            node_eval_counts[i] = evaluate_num;
            eval_offsets[i] = total_evaluations;
            
            // Add node-neighbor pairs
            for(int j = 0; j < evaluate_num; j++) {
                node_ids.push_back(v_id);
                neighbor_ids.push_back(neighbor_set[j]);
            }
            
            total_evaluations += evaluate_num;
        }
        
        if(total_evaluations == 0) {
            // Add zero results for nodes with no neighbors
            for(size_t i = 0; i < chunk_size; i++) {
                all_results.push_back(0.0f);
            }
            continue;
        }
        
        // Step 2: Allocate GPU memory for direct indexing
        vertex_id_t* d_node_ids = nullptr;
        vertex_id_t* d_neighbor_ids = nullptr;
        vertex_id_t* d_id2offset_gpu = nullptr;
        float* d_results = nullptr;
        
        checkCUDAerr(cudaMalloc((void**)&d_node_ids, total_evaluations * sizeof(vertex_id_t)));
        checkCUDAerr(cudaMalloc((void**)&d_neighbor_ids, total_evaluations * sizeof(vertex_id_t)));
        checkCUDAerr(cudaMalloc((void**)&d_id2offset_gpu, vocab_size * sizeof(vertex_id_t)));
        checkCUDAerr(cudaMalloc((void**)&d_results, total_evaluations * sizeof(float)));
        printf("[ %d ] GPU Direct: Allocated GPU memory for %d evaluations\n", my_rank, total_evaluations);
        printf("[ %d ] CUDA malloc sizes - node_ids: %zu MB, neighbor_ids: %zu MB, id2offset: %zu MB, results: %zu MB, total = %zu MB\n", 
               my_rank, 
               total_evaluations * sizeof(vertex_id_t)/(1024*1024), 
               total_evaluations * sizeof(vertex_id_t)/(1024*1024), 
               vocab_size * sizeof(vertex_id_t)/(1024*1024), 
               total_evaluations * sizeof(float)/(1024*1024),
               total_evaluations * sizeof(vertex_id_t) * 2/(1024*1024) + vocab_size * sizeof(vertex_id_t)/(1024*1024) + total_evaluations * sizeof(float)/(1024*1024));
        // Step 3: Copy data to GPU
        checkCUDAerr(cudaMemcpy(d_node_ids, node_ids.data(), 
                               total_evaluations * sizeof(vertex_id_t), cudaMemcpyHostToDevice));
        checkCUDAerr(cudaMemcpy(d_neighbor_ids, neighbor_ids.data(), 
                               total_evaluations * sizeof(vertex_id_t), cudaMemcpyHostToDevice));
        checkCUDAerr(cudaMemcpy(d_id2offset_gpu, id2offset.data(), 
                               vocab_size * sizeof(vertex_id_t), cudaMemcpyHostToDevice));
        
        // Step 4: Launch optimized GPU kernel (NO MEMORY COPIES!)
        if(use_optimized_kernel) {
            // Use shared memory optimized version
            int threadsPerBlock = layer1_size > 200 ? 256 : 128;
            int blocks = total_evaluations;
            size_t sharedMemSize = 3 * threadsPerBlock * sizeof(float);
            
            batch_similarity_direct_index_optimized_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
                d_syn0, d_node_ids, d_neighbor_ids, d_results, 
                layer1_size, total_evaluations, d_id2offset_gpu);
        } else {
            // Use simple version
            int threadsPerBlock = 256;
            int blocks = (total_evaluations + threadsPerBlock - 1) / threadsPerBlock;
            
            batch_similarity_direct_index_kernel<<<blocks, threadsPerBlock>>>(
                d_syn0, d_node_ids, d_neighbor_ids, nullptr, nullptr, 
                d_results, layer1_size, total_evaluations, d_id2offset_gpu);
        }
        
        cudaDeviceSynchronize();
        
        // Step 5: Copy results back and compute averages
        float* h_results = new float[total_evaluations];
        checkCUDAerr(cudaMemcpy(h_results, d_results, 
                               total_evaluations * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Step 6: Compute average similarity for each node
        for(size_t i = 0; i < chunk_size; i++) {
            int evaluate_num = node_eval_counts[i];
            int offset = eval_offsets[i];
            
            if(evaluate_num == 0) {
                all_results.push_back(0.0f);
                continue;
            }
            
            float sum_similarity = 0.0f;
            for(int j = 0; j < evaluate_num; j++) {
                sum_similarity += h_results[offset + j];
            }
            
            float avg_similarity = sum_similarity / evaluate_num;
            all_results.push_back(avg_similarity);
        }
        
        // Cleanup GPU memory
        cudaFree(d_node_ids);
        cudaFree(d_neighbor_ids);
        cudaFree(d_id2offset_gpu);
        cudaFree(d_results);
        delete[] h_results;
    }
    
    printf("[ %d ] GPU Direct Index: Completed processing %zu nodes\n", my_rank, all_results.size());
    return all_results;
}
// ========================================================

float find_supernode_topK_accurancy(float p,int k,myEdgeContainer*csr){
  float top_sum = 0;
  for(vertex_id_t v_i = 0; v_i < vocab_size*0.03; v_i ++){
    vector<std::pair<float,vertex_id_t>> supernode_sim;
    for(vertex_id_t v_j = 0; v_j < vocab_size * p; v_j ++){
      float sim = cos_sim(syn0+v_i * layer1_size, syn0+v_j*layer1_size,layer1_size);
      supernode_sim.push_back({sim,v_j});
    }
    sort(supernode_sim.begin(),supernode_sim.end(),[](std::pair<float,vertex_id_t>&p1,std::pair<float,vertex_id_t>&p2){
        return p1.first > p2.first;
        });
    vector<vertex_id_t> selected_topK(k);
    for(int i = 0 ;i< k; i++){
      // selected_topK[i] = supernode_sim[i].second;
      selected_topK[i] = supernode_sim[i].second ; 
    }
    vector<vertex_id_t> real_neighbor;
    for(auto it = csr->adj_lists[v_i].begin; it < csr->adj_lists[v_i].end; it++){
      real_neighbor.push_back(it->neighbour);
    }
    sort(real_neighbor.begin(),real_neighbor.end());
    sort(selected_topK.begin(),selected_topK.end());
    vector<vertex_id_t>result_set;
    myIntersectition(selected_topK, real_neighbor, result_set);
    top_sum += result_set.size();
  }
  return top_sum/ (float)(k * vocab_size *p);
}
double train_spend_time = 0.0;
double evaluate_spend_time = 0.0;
void TrainModel(SyncQueue& taskq,myEdgeContainer*csr, const TrainingConfig& config) {
  Timer total_init_timer;  // Total initialization timer
  printf("==========================Train Model In=====================\n");
  
  Timer vocab_timer;
  long a, b, c, d;
  FILE *fo;
  starting_alpha = alpha;
  ReadVocabFromDegree(g_v_degree);
  double vocab_time = vocab_timer.duration();
  printf("========================Read Vocab ok=======================\n");
  printf("[ %d ] Vocab initialization completed in %.6f seconds\n", my_rank, vocab_time);
  printf("vocab_size: %lu\n",vocab_size);
  // for(size_t i = 0; i < vocab_size * 0.10;i++){
  //   printf("id: %s, degree: %ld\n",vocab[i].word,vocab[i].cn);
  // }
  Timer train_timer;
  vector<float> H;
  float delta_H;
  int init_round = config.init_round;
  int batch_size = config.batch_size;
  

  // if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  // if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) printf("[ Warning ] output file missing\n");
  if (output_file[0] == 0) return;
  
  Timer net_timer;
  InitNet();
  double net_time = net_timer.duration();
  printf("[ %d ] InitNet Success in %.6f seconds\n",my_rank, net_time);
  
  Timer cuda_timer;
  if (hs > 0) InitVocabStructCUDA();
  if (negative > 0) InitUnigramTable();
  double cuda_time = cuda_timer.duration();

  start = clock();
  srand(time(NULL));

  float* kl = new float;
  // lr_scheduler = new  FixedLR(0.025);
  // lr_scheduler = new  StepDecayLR(0.025,0.5,3);
  lr_scheduler = new ExponentialDecayLR(0.025,0.1);

  int nu = 1;

  vertex_id_t part_vertex_num = ((vocab_size % num_procs) > 0) ? (vocab_size / num_procs + 1) : (vocab_size / num_procs); 
  
  Timer batch_cuda_timer;
  // Batch processing GPU memory (larger allocation)
  const int MAX_BATCH_EVALUATIONS = batch_size * EVALUATION_NEIGHBOUR_NUM;  // batch_size * 30 evaluations
  float *d_A_batch, *d_B_batch, *d_results_batch;
  
  printf("[ %d ] Allocating batch GPU memory for %d evaluations\n", my_rank, MAX_BATCH_EVALUATIONS);
  checkCUDAerr(cudaMalloc(&d_A_batch, MAX_BATCH_EVALUATIONS * layer1_size * sizeof(float)));
  checkCUDAerr(cudaMalloc(&d_B_batch, MAX_BATCH_EVALUATIONS * layer1_size * sizeof(float)));
  checkCUDAerr(cudaMalloc(&d_results_batch, MAX_BATCH_EVALUATIONS * sizeof(float)));
  printf("[ %d ] [Batch GPU memory allocation] CUDA malloc: A(%zu MB), B(%zu MB), Results(%zu MB), total(%zu MB)\n", 
         my_rank, 
         (MAX_BATCH_EVALUATIONS * layer1_size * sizeof(float)) / (1024 * 1024),
         (MAX_BATCH_EVALUATIONS * layer1_size * sizeof(float)) / (1024 * 1024),
         (MAX_BATCH_EVALUATIONS * sizeof(float)) / (1024 * 1024)
         , (2 * MAX_BATCH_EVALUATIONS * layer1_size * sizeof(float) + MAX_BATCH_EVALUATIONS * sizeof(float)) / (1024 * 1024));
  printf("[ %d ] Batch GPU memory allocation successful\n", my_rank);
  double batch_cuda_time = batch_cuda_timer.duration();
  printf("[ %d ] Batch CUDA memory setup time: %.6f seconds\n", my_rank, batch_cuda_time);

  double total_init_time = total_init_timer.duration();
  train_model_init_time = total_init_time;
  printf("==========init success================\n");
  printf("[ %d ] === INITIALIZATION TIMING BREAKDOWN ===\n", my_rank);
  printf("[ %d ] Vocab setup:         %.6f seconds\n", my_rank, vocab_time);
  printf("[ %d ] Network init:        %.6f seconds\n", my_rank, net_time);
  printf("[ %d ] Vocab CUDA setup:    %.6f seconds\n", my_rank, cuda_time);
  printf("[ %d ] Batch CUDA setup:    %.6f seconds\n", my_rank, batch_cuda_time);
  printf("[ %d ] TOTAL INIT TIME:     %.6f seconds\n", my_rank, total_init_time);
  printf("[ %d ] ===========================================\n", my_rank);

  thread* sync_thread = nullptr;
  if (num_procs > 1) {
    sync_thread = new thread(sync_embedding_func);
  }
  vertex_id_t last_eva_num = vocab_size;
  int train_iter = 0;
  bool stop_train_flag = false;
  
  // Round-level training timing statistics
  std::vector<double> round_wait_times;
  std::vector<double> round_training_times;
  std::vector<double> round_eval_times;
  std::vector<double> round_total_times;
  
  while(!stop_train_flag){
    Timer round_timer;  // Timer for entire training round
    Timer wait_timer;
    
    // 等待游走产生训练资料
    unique_lock<mutex> lock(mtx);
    cv.wait(lock,[]{return hasResource;});
    double wait_time = wait_timer.duration();
    
    corpus_t corpus_data = taskq.pop();  // Get corpus data directly instead of file path
    // 此时可以sample下一轮了
    hasResource = false; // 坑位被释放
    pauseWalk.store(false, std::memory_order_relaxed); // 继续游走
    cv.notify_one();
    cout << "====== POP CORPUS DATA (size: " << corpus_data.size() << ") ===" << endl;
    train_iter++;
    alpha = lr_scheduler->get_lr();
    pause_sync = false;
    
    MPI_Barrier(MPI_EMB_COMM);// stop sync thread until all the sync thread is ready to be halted
    printf("[ %d ] === TRAINING ROUND %d START === Wait: %.3fs\n",my_rank, train_iter, wait_time);
    
    Timer actual_train_timer;
    TrainModelThreadMemory(corpus_data);  // New function to train with memory data
    double actual_train_time = actual_train_timer.duration();
    printf("round %d \n",train_iter);
    printf("[ %d ] train corpus data finished, time: %.3fs\n",my_rank, actual_train_time);
    MPI_Barrier(MPI_EMB_COMM);
    pause_sync = true;
    if(train_iter >= init_round) {
      pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
    }
    // if (num_procs == 1) {
    //   if(train_iter >= init_round) {
    //     pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
    //   }
    //   // pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
    // } else {
    //   // pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
    // }
    // pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
    std::cout << std::endl;
    
    vertex_id_t eva_num = 0;
    printf("init_round: %d\n",init_round);
    // TODO: set a args to control whether reactivate inactive nodes
    bool reactivate_flag = true;
    if(train_iter >= init_round){
      printf("[ %d ] evaluation start (with synchronized embeddings)\n", my_rank);
      
      Timer eva_timer;
      // Step 1: Collect all nodes that need evaluation
      std::vector<vertex_id_t> nodes_to_evaluate;
      std::vector<vertex_id_t> nodes_inactive;
      size_t evaluated_node_count = 0;
      size_t inactive_node_count = 0;
      double rank_converged_ratio = 0.0;
      double rank_reactivated_ratio = 0.0;
      for(vertex_id_t v = part_vertex_num * my_rank; v < part_vertex_num * (my_rank + 1) && v < vocab_size; v++){
        if(vertex_walker_stop_flag[v] == 0){
          nodes_to_evaluate.push_back(v);
        } else if(reactivate_flag){
          nodes_inactive.push_back(v);
        }
      }
      evaluated_node_count = nodes_to_evaluate.size();
      inactive_node_count = nodes_inactive.size();
      
      printf("[ %d ] Found %zu nodes to evaluate\n", my_rank, nodes_to_evaluate.size());
      int converged_count = 0;
      if(!nodes_to_evaluate.empty()){
        // Step 2: Batch process all nodes with configurable implementation
        std::vector<float> similarities;
        
        if(use_gpu_direct_indexing) {
          // NEW: GPU-side direct indexing (eliminates memory copy bottleneck)
          printf("[ %d ] Using GPU Direct Indexing optimization\n", my_rank);
          similarities = batch_node_neighbor_direct_index(
            nodes_to_evaluate, csr, batch_size, use_optimized_direct_kernel);
        } else {
          // ORIGINAL: Memory copy based batch processing (preserved for comparison)
          printf("[ %d ] Using original memory-copy batch processing\n", my_rank);
          similarities = batch_node_neighbor_average_cos_sim_chunked(
            nodes_to_evaluate, csr, d_A_batch, d_B_batch, d_results_batch, batch_size);
        }
        
        // Step 3: Apply results and count non-converged nodes
        converged_count = 0;
        for(size_t i = 0; i < nodes_to_evaluate.size(); i++){
          vertex_id_t v = nodes_to_evaluate[i];
          float s = similarities[i];
          
          // Debug output for first few nodes
          if(v < 10) printf("Node %d similarity: %f\n", v, s);
          
          if(s > NODE_TRAINING_CONVERGE_THRESHOLD){
            vertex_walker_stop_flag[v] = 1;
            converged_count++;
          } else {
            eva_num++;
          }
        }
        
        printf("[ %d ] Batch evaluation completed: %d converged, %d non-converged\n", 
               my_rank, converged_count, eva_num);
        rank_converged_ratio = nodes_to_evaluate.empty() ? 0.0 :
          static_cast<double>(converged_count) / static_cast<double>(nodes_to_evaluate.size());
      }
      int reactivated_count = 0;
      if(!nodes_inactive.empty()){
        // evaluate to jusify whether reactivate inactive nodes
        // get the number of inactive nodes whose similarity < NODE_TRAINING_REACTIVATE_THRESHOLD
        reactivated_count = 0;
        std::vector<float> inactive_similarities;
        if(use_gpu_direct_indexing) {
          // NEW: GPU-side direct indexing (eliminates memory copy bottleneck)
          printf("[ %d ] Reactivation: Using GPU Direct Indexing optimization\n", my_rank);
          inactive_similarities = batch_node_neighbor_direct_index(
            nodes_inactive, csr, batch_size, use_optimized_direct_kernel);
        } else {
          // ORIGINAL: Memory copy based batch processing (preserved for comparison)
          printf("[ %d ] Reactivation: Using original memory-copy batch processing\n", my_rank);
          inactive_similarities = batch_node_neighbor_average_cos_sim_chunked(
            nodes_inactive, csr, d_A_batch, d_B_batch, d_results_batch, batch_size);
        }
        for(size_t i = 0; i < nodes_inactive.size(); i++){
          vertex_id_t v = nodes_inactive[i];
          float s = inactive_similarities[i];
          
          // Debug output for first few nodes
          if(v < 10) printf("Inactive Node %d similarity: %f\n", v, s);
          
          if(s < NODE_TRAINING_REACTIVATE_THRESHOLD){
            vertex_walker_stop_flag[v] = 0; // reactivate
            reactivated_count++;
            eva_num++;
          } 
        }
        printf("[ %d ] Reactivation completed: %d nodes reactivated\n", my_rank, reactivated_count);
        rank_reactivated_ratio = nodes_inactive.empty() ? 0.0 :
          static_cast<double>(reactivated_count) / static_cast<double>(nodes_inactive.size());
      } else {
        printf("[ %d ] Skipping evaluation for %zu inactive nodes\n", my_rank, nodes_inactive.size());
      }
      printf("[ %d ] Iter %d rank summary: converge %d/%zu (%.4f), reactivate %d/%zu (%.4f)\n",
             my_rank, train_iter,
             converged_count, evaluated_node_count, rank_converged_ratio,
             reactivated_count, inactive_node_count, rank_reactivated_ratio);
      printf("[ %d ] Total reactivated nodes: %d\n", my_rank, reactivated_count);
      printf("[ %d ] evaluation finished\n", my_rank);
      
      printf("[ %d ] vertex_walker_stop_flag size: %lu\n",my_rank,vertex_walker_stop_flag.size());
      MPI_Allreduce(MPI_IN_PLACE, vertex_walker_stop_flag.data(),vertex_walker_stop_flag.size(), MPI_INT, MPI_MAX, MPI_EVA_COMM);
      MPI_Allreduce(MPI_IN_PLACE, &eva_num, 1, get_mpi_data_type<vertex_id_t>(), MPI_SUM , MPI_EVA_COMM);
      // if (num_procs == 1) {
      //   pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
      // } else {
      //   // pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
      // }
      // pauseWalk.store(true, std::memory_order_relaxed); // 打断游走
      // 收敛了，每次减少的比例不多
      float eva_num_ratio = (float)eva_num / last_eva_num;
      printf("[ %d ] last_eva_num: %u current_eva_num: %u ratio: %f \n",my_rank,last_eva_num,eva_num,eva_num_ratio);
      if( last_eva_num != 0 && eva_num_ratio> EVALUATION_NEIGHBOUR_NUM_CONVERGE_RATIO ){
        halt_sync = true; // 停止同步
        stop_sampling_flag = true; // 停止采样
        stop_train_flag = true;    // 停止训练
      }
      printf("[ %d ]Iter %d Evaluate Num: %d Ratio: %f Time: %f s\n",my_rank,train_iter,eva_num,eva_num_ratio,eva_timer.duration());
      double eval_time = eva_timer.duration();
      double total_round_time = round_timer.duration();
      
      // Record round timing statistics
      round_wait_times.push_back(wait_time);
      round_training_times.push_back(actual_train_time);
      round_eval_times.push_back(eval_time);
      round_total_times.push_back(total_round_time);
      printf("[ %d ] === TRAINING ROUND %d COMPLETED === Wait: %.3fs, Train: %.3fs, Eval: %.3fs, Total: %.3fs\n",
        my_rank, train_iter, wait_time, actual_train_time, eval_time, total_round_time);
      last_eva_num = eva_num;
    } else {
      printf("[ %d ]Iter %d Skipping evaluation (init_round=%d)\n",my_rank,train_iter,init_round);
    }
  }
  
  // Print detailed round-level training timing statistics
  if (!round_wait_times.empty()) {
    printf("\n================== [ %d ] TRAINING ROUND-LEVEL TIMING STATISTICS ==================\n", my_rank);
    double total_wait_time = 0.0, total_training_time = 0.0, total_eval_time = 0.0, total_time = 0.0;
    
    for (size_t i = 0; i < round_wait_times.size(); i++) {
      total_wait_time += round_wait_times[i];
      total_training_time += round_training_times[i];
      total_eval_time += round_eval_times[i];
      total_time += round_total_times[i];
      printf("[ %d ] Round %2zu: Wait=%.3fs, Train=%.3fs, Eval=%.3fs, Total=%.3fs\n", 
             my_rank, i+1, round_wait_times[i], round_training_times[i], round_eval_times[i],
             round_total_times[i]);
    }
    
    printf("===================================================================================\n");
    printf("[ %d ] SUMMARY: Rounds=%zu, Wait=%.3fs, Train=%.3fs, Eval=%.3fs, Total=%.3fs\n",
           my_rank, round_wait_times.size(), total_wait_time, total_training_time, total_eval_time,
           total_time);
    printf("[ %d ] AVERAGES: Wait=%.3fs, Train=%.3fs, Eval=%.3fs, Total=%.3fs per round\n",
           my_rank, total_wait_time/round_wait_times.size(), total_training_time/round_wait_times.size(),
           total_eval_time/round_wait_times.size(), total_time/round_wait_times.size());
    printf("===================================================================================\n\n");
    
    // Update global timing variables
    train_wait_walk_time = total_wait_time;
    pure_training_time = total_training_time;
    evaluate_spend_time = total_eval_time;
  }
  
  // OPTIMIZATION: No sync thread to clean up since we disabled training-time sync
  if (num_procs > 1 && sync_thread != nullptr) {
    MPI_Barrier(MPI_EMB_COMM);
    halt_sync = true;
    sync_cv.notify_all();
    printf("[ %d ] Waiting Syncing Thread\n",my_rank);
    sync_thread->join();
    MPI_Barrier(MPI_EMB_COMM);
    printf("[ %d ] Syncing Thread Halt\n",my_rank);
    delete sync_thread;
  }

  printf("[%d] Train: %f Sync: %f EVA: %f \n",my_rank,pure_training_time,sync_spend_time,evaluate_spend_time);
  
  // // Free original evaluation GPU memory
  // cudaFree(d_A);
  // cudaFree(d_B);
  // cudaFree(d_results);
  
  // Free batch processing GPU memory
  printf("[ %d ] Freeing batch GPU memory\n", my_rank);
  cudaFree(d_A_batch);
  cudaFree(d_B_batch);
  cudaFree(d_results_batch);

  
  cudaFree(d_table);
  cudaFree(d_syn1);
  cudaFree(d_syn0);
  cudaFree(d_vocab_codelen);
  cudaFree(d_vocab_point);
  cudaFree(d_vocab_code);

  MPI_Barrier(MPI_EMB_COMM);// stop sync thread until all the sync thread is ready to be halted
  if(my_rank != 0) return;

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  fo = fopen(output_file, "wb");
  if(fo == NULL) printf("[ %d ] [%s] open fail\n", my_rank, output_file);
  if (classes == 0) {	
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    float closev, x;
    float *cent = (float *)calloc(classes * layer1_size, sizeof(float));

    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }

    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);


    free(centcn);
    free(cent);
    free(cl);
  }
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout<<"[ "<<my_rank<<" ] Save Embedding: " <<time_span.count() << " s" <<std::endl;
  saving_embedding_time = time_span.count();
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}
int train_corpus_cuda(int argc, char **argv,const vector<vertex_id_t>& degrees,SyncQueue& corpus_q,int _my_rank,myEdgeContainer* csr, const TrainingConfig& config) 
{
  Timer actual_training_timer;
  printf("[ %d ] Starting actual training execution...\n", _my_rank);

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len;

  cout <<_my_rank << " train_corpus_cuda invoke ok\n";
  MPI_Comm_dup(MPI_COMM_WORLD,&MPI_EMB_COMM);
  MPI_Comm_dup(MPI_COMM_WORLD,&MPI_EVA_COMM);
  MPI_Comm_size(MPI_EMB_COMM, &num_procs);
  MPI_Comm_rank(MPI_EMB_COMM, &my_rank);
  MPI_Get_processor_name(hostname, &hostname_len);

  printf("processor name: %s, number of processors: %d, rank: %d\n", hostname, num_procs, my_rank);

  vertex_walker_stop_flag.assign(degrees.size(),0);
  g_v_degree.assign(degrees.begin(), degrees.end());

  const size_t node_count = g_v_degree.size();
  vocab_hash_size = CalculateVocabHashSize(node_count);
  const size_t desired_vocab_capacity = node_count + 1000ULL;
  if (static_cast<size_t>(vocab_max_size) < desired_vocab_capacity) {
    vocab_max_size = static_cast<long long>(desired_vocab_capacity);
  }
  printf("[ %d ] Vocab hash size configured: %zu (node count: %zu)\n", my_rank, vocab_hash_size, node_count);

  printf("train_corpus_Cuda calling!!!!\n");
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-reuse-neg <int>\n");
    printf("\t\tA sentence share a negative sample set; (0 = not used / 1 = used)\n");

    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }

  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-emb_output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-reuse-neg", argc, argv)) > 0) reuseNeg = atoi(argv[i + 1]);
  Timer vocab_init_timer;
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (long long *)calloc(vocab_hash_size, sizeof(long long));
  expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
  log_host_allocation("vocab array", (size_t)vocab_max_size * sizeof(struct vocab_word));
  log_host_allocation("vocab hash", (size_t)vocab_hash_size * sizeof(long long));
  log_host_allocation("expTable host", (size_t)(EXP_TABLE_SIZE + 1) * sizeof(float));

  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  double vocab_init_time = vocab_init_timer.duration();
  printf("[ %d ] Vocab initialization (preparation only) completed in %.6f seconds\n", my_rank, vocab_init_time);
  Timer copy_timer;
  checkCUDAerr(cudaMalloc((void **)&d_expTable, (EXP_TABLE_SIZE + 1) * sizeof(float)));
  printf("[ %d ] CUDA malloc expTable on GPU\n", my_rank);
  printf("[ %d ] CUDA malloc d_expTable (%zu MB) done!\n", my_rank, (EXP_TABLE_SIZE + 1) * sizeof(float) / (1024 * 1024));
  checkCUDAerr(cudaMemcpy(d_expTable, expTable, (EXP_TABLE_SIZE + 1) * sizeof(float), cudaMemcpyHostToDevice));
  double copy_time = copy_timer.duration();
  printf("[ %d ] Copy expTable to GPU in %.6f seconds\n", my_rank, copy_time);
  TrainModel(corpus_q,csr,config);

  printf("[ %d ] [Sync Time Spend: %f s]\n",my_rank,sync_spend_time);
  // memory free
  size_t vocab_codelen_bytes = 0;
  size_t vocab_point_bytes = 0;
  size_t vocab_code_bytes = 0;
  if (vocab_codelen != nullptr)
  {
      vocab_codelen_bytes = (vocab_size + 1) * sizeof(int);
      size_t vocab_offset = static_cast<size_t>(vocab_codelen[vocab_size]);
      vocab_point_bytes = vocab_offset * sizeof(int);
      vocab_code_bytes = vocab_offset * sizeof(char);
  }
  free(table);
  log_host_release("unigram table host", static_cast<size_t>(table_size) * sizeof(int));
  free(syn0);
  log_host_release("syn0 host copy", static_cast<size_t>(vocab_size) * layer1_size * sizeof(float));
  free(syn1);
  log_host_release("syn1 host copy", static_cast<size_t>(vocab_size) * layer1_size * sizeof(float));
  free(syn1neg);
  log_host_release("syn1neg host copy", static_cast<size_t>(vocab_size) * layer1_size * sizeof(float));
  free(vocab);
  log_host_release("vocab array", static_cast<size_t>(vocab_size + 1) * sizeof(struct vocab_word));
  free(vocab_hash);
  log_host_release("vocab hash", static_cast<size_t>(vocab_hash_size) * sizeof(long long));
  free(expTable);
  log_host_release("expTable host", static_cast<size_t>(EXP_TABLE_SIZE + 1) * sizeof(float));
  if (vocab_codelen != nullptr)
  {
      free(vocab_codelen);
      log_host_release("vocab codelen host", vocab_codelen_bytes);
  }
  if (vocab_point != nullptr)
  {
      free(vocab_point);
      log_host_release("vocab point host", vocab_point_bytes);
  }
  if (vocab_code != nullptr)
  {
      free(vocab_code);
      log_host_release("vocab code host", vocab_code_bytes);
  }
  cudaFree(d_expTable);
  free(last_emb);
  
  training_time = actual_training_timer.duration();
  printf("[ %d ] Training execution completed. Actual training time: %lf seconds\n", _my_rank, training_time);

  return 0;
}
