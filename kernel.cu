#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <algorithm>
#include <chrono>

#define INF 0x3f3f3f3f
#define BLOCK_SIZE 256
#define NUM_ITER 30

using namespace std;
using namespace chrono;

static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

struct Graph {
    int* vertexArray;
    int* edgeArray;
    int* weightArray;
    int numVertices;
    int numEdges;
};

__global__ void generate_csr_gpu(int* vertexArray, int* edgeArray, int* weightArray, int V, int E, int maxWeight, int edgesPerNode, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        int start = tid * edgesPerNode;
        vertexArray[tid] = start;
        curandState state;
        curand_init(seed + tid, tid, 0, &state);
        for (int i = 0; i < edgesPerNode; ++i) {
            int dest = curand(&state) % V;
            while (dest == tid) dest = curand(&state) % V;
            edgeArray[start + i] = dest;
            weightArray[start + i] = 1 + (curand(&state) % maxWeight);
        }
    }
    if (tid == 0) vertexArray[V] = E;
}

__global__ void initArrays(bool* finalized, int* dist, int* updatingDist, int src, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        finalized[tid] = (tid == src);
        dist[tid] = (tid == src) ? 0 : INF;
        updatingDist[tid] = dist[tid];
    }
}

__global__ void Kernel1(const int* Va, const int* Ea, const int* Wa, bool* finalized, int* dist, int* updDist, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && finalized[tid]) {
        finalized[tid] = false;
        int start = Va[tid];
        int end = Va[tid + 1];
        for (int e = start; e < end; e++) {
            int v = Ea[e];
            int sum = dist[tid] + Wa[e];
         
            if (dist[tid] < INF && Wa[e] < INF && sum > 0)
                atomicMin(&updDist[v], sum); 
        }
    }
}

__global__ void Kernel2(bool* finalized, int* dist, int* updDist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (dist[tid] > updDist[tid]) {
            dist[tid] = updDist[tid];
            finalized[tid] = true;
        }
        updDist[tid] = dist[tid];
    }
}

static void save_vector(const vector<int>& vec, const string& filename) {
    ofstream fout(filename);
    for (size_t i = 0; i < vec.size(); ++i)
        fout << "Node " << i << ": " << (vec[i] >= INF / 2 ? -1 : vec[i]) << "\n";
    fout.close();
}

static void generate_graph_gpu_csr(Graph& g, int nodes, int edges, int maxWeight) {
    int edgesPerNode = edges / nodes;
    generate_csr_gpu << < (nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
        g.vertexArray, g.edgeArray, g.weightArray, nodes, edges, maxWeight, edgesPerNode, (unsigned int)time(0));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void dijkstraGPU(Graph& graph, int src, vector<int>& dist) {
    bool* d_fin;
    int* d_dist, * d_updDist;
    bool* h_fin = new bool[graph.numVertices];
    CUDA_CHECK(cudaMalloc(&d_fin, graph.numVertices * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_dist, graph.numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_updDist, graph.numVertices * sizeof(int)));
    initArrays << < (graph.numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
        d_fin, d_dist, d_updDist, src, graph.numVertices);
    CUDA_CHECK(cudaDeviceSynchronize());
    do {
        for (int k = 0; k < NUM_ITER; k++) {
            Kernel1 << < (graph.numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
                graph.vertexArray, graph.edgeArray, graph.weightArray, d_fin, d_dist, d_updDist, graph.numVertices, graph.numEdges);
            CUDA_CHECK(cudaGetLastError());
            Kernel2 << < (graph.numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
                d_fin, d_dist, d_updDist, graph.numVertices);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(h_fin, d_fin, graph.numVertices * sizeof(bool), cudaMemcpyDeviceToHost));
    } while (any_of(h_fin, h_fin + graph.numVertices, [](bool v) { return v; }));
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, graph.numVertices * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_fin);
    cudaFree(d_dist);
    cudaFree(d_updDist);
    delete[] h_fin;
}

int main() {
    int nodes = 1000000;
    int edges = 10000000;
    int maxWeight = 100000;
    int startNode = 0;
    cout << "Estimated memory: " << ((nodes + 1) * sizeof(int) + edges * 2 * sizeof(int)) / (1024.0 * 1024.0) << " MB" << endl;
    Graph g;
    g.numVertices = nodes;
    g.numEdges = edges;
    CUDA_CHECK(cudaMallocManaged(&g.vertexArray, (nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&g.edgeArray, edges * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&g.weightArray, edges * sizeof(int)));
    auto t1 = high_resolution_clock::now();
    generate_graph_gpu_csr(g, nodes, edges, maxWeight);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = high_resolution_clock::now();
    vector<int> dist(nodes, INF);
    auto t3 = high_resolution_clock::now();
    dijkstraGPU(g, startNode, dist);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t4 = high_resolution_clock::now();
    save_vector(dist, "distances_cuda.txt");
    cout << "Graph gen+copy: " << duration<double>(t2 - t1).count() << " sec\n";
    cout << "CUDA Dijkstra:  " << duration<double>(t4 - t3).count() << " sec\n";
    cudaFree(g.vertexArray);
    cudaFree(g.edgeArray);
    cudaFree(g.weightArray);
    return 0;
}
