// Stokastisk 0.4.0 EXPERIMENTAL: host-staged GPU->GPU copy using only the CUDA Runtime API.
//   src device --cudaMemcpyAsync--> pinned host buffer[k] --cudaMemcpyAsync--> dst device
// Chunks rotate through n pinned buffers; cudaEvents hand each buffer from the D2H stage to the H2D stage and back
// (cudaStreamWaitEvent works across devices). No kernels, so the object carries no device code; it links libcudart.so.13.
// Build: nvcc -O2 -std=c++17 -shared -Xcompiler -fPIC -gencode arch=compute_75,code=sm_75 host_staged_xfer.cpp -o libstokastisk_xfer.so -lcudart
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

namespace {
thread_local char g_err[512] = "";
bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::snprintf(g_err, sizeof g_err, "%s: %s", what, cudaGetErrorString(e));
    return false;
}
#define CK(x) do { if (!ok((x), #x)) return false; } while (0)

struct Ctx {
    int src, dst, nbuf; size_t chunk;
    cudaStream_t s_src = nullptr, s_dst = nullptr;
    std::vector<void*> host; std::vector<cudaEvent_t> d2h, h2d;
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    bool used[64] = {false};
};

bool init(Ctx* c) {
    CK(cudaSetDevice(c->src)); CK(cudaStreamCreateWithFlags(&c->s_src, cudaStreamNonBlocking));
    CK(cudaSetDevice(c->dst)); CK(cudaStreamCreateWithFlags(&c->s_dst, cudaStreamNonBlocking));
    CK(cudaEventCreate(&c->t0)); CK(cudaEventCreate(&c->t1));
    c->host.assign(c->nbuf, nullptr); c->d2h.assign(c->nbuf, nullptr); c->h2d.assign(c->nbuf, nullptr);
    for (int i = 0; i < c->nbuf; ++i) {
        CK(cudaHostAlloc(&c->host[i], c->chunk, cudaHostAllocPortable));       // portable: usable from both devices' contexts
        CK(cudaSetDevice(c->src)); CK(cudaEventCreateWithFlags(&c->d2h[i], cudaEventDisableTiming));
        CK(cudaSetDevice(c->dst)); CK(cudaEventCreateWithFlags(&c->h2d[i], cudaEventDisableTiming));
    }
    return true;
}
}  // namespace

extern "C" {

const char* stokastisk_xfer_last_error() { return g_err; }

void* stokastisk_xfer_create(int src, int dst, size_t chunk_bytes, int n_buffers) {
    g_err[0] = 0;
    if (src == dst || chunk_bytes == 0 || n_buffers < 1 || n_buffers > 64) { std::snprintf(g_err, sizeof g_err, "invalid arguments"); return nullptr; }
    Ctx* c = new Ctx{src, dst, n_buffers, chunk_bytes};
    if (!init(c)) { return nullptr; }
    return c;
}

// Blocking. Writes the elapsed GPU time (ms, dst-device events) to *ms. Returns 0 on success.
int stokastisk_xfer_transfer(void* h, const void* src_ptr, void* dst_ptr, size_t nbytes, double* ms) {
    Ctx* c = static_cast<Ctx*>(h); g_err[0] = 0;
    if (!c) return 1;
    for (int i = 0; i < c->nbuf; ++i) c->used[i] = false;
    if (!ok(cudaSetDevice(c->dst), "cudaSetDevice(dst)")) return 2;
    if (!ok(cudaEventRecord(c->t0, c->s_dst), "cudaEventRecord(t0)")) return 3;
    size_t i = 0;
    for (size_t off = 0; off < nbytes; off += c->chunk, ++i) {
        size_t n = std::min(c->chunk, nbytes - off); int k = int(i % c->nbuf);
        if (c->used[k]) {   // buffer k is still being read by the previous H2D: the source stream must wait before overwriting it
            if (!ok(cudaSetDevice(c->src), "setdev")) return 4;
            if (!ok(cudaStreamWaitEvent(c->s_src, c->h2d[k], 0), "wait h2d")) return 5;
        }
        if (!ok(cudaSetDevice(c->src), "setdev")) return 4;
        if (!ok(cudaMemcpyAsync(c->host[k], static_cast<const char*>(src_ptr) + off, n, cudaMemcpyDeviceToHost, c->s_src), "D2H")) return 6;
        if (!ok(cudaEventRecord(c->d2h[k], c->s_src), "record d2h")) return 7;
        if (!ok(cudaSetDevice(c->dst), "setdev")) return 4;
        if (!ok(cudaStreamWaitEvent(c->s_dst, c->d2h[k], 0), "wait d2h")) return 8;      // H2D only after this chunk's D2H completed
        if (!ok(cudaMemcpyAsync(static_cast<char*>(dst_ptr) + off, c->host[k], n, cudaMemcpyHostToDevice, c->s_dst), "H2D")) return 9;
        if (!ok(cudaEventRecord(c->h2d[k], c->s_dst), "record h2d")) return 10;
        c->used[k] = true;
    }
    if (!ok(cudaEventRecord(c->t1, c->s_dst), "record t1")) return 11;
    if (!ok(cudaStreamSynchronize(c->s_dst), "sync dst")) return 12;
    if (!ok(cudaSetDevice(c->src), "setdev") || !ok(cudaStreamSynchronize(c->s_src), "sync src")) return 13;
    float f = 0; if (!ok(cudaEventElapsedTime(&f, c->t0, c->t1), "elapsed")) return 14;
    if (ms) *ms = f;
    return 0;
}

void stokastisk_xfer_destroy(void* h) {
    Ctx* c = static_cast<Ctx*>(h); if (!c) return;
    cudaSetDevice(c->src);
    for (auto e : c->d2h) if (e) cudaEventDestroy(e);
    if (c->s_src) cudaStreamDestroy(c->s_src);
    cudaSetDevice(c->dst);
    for (auto e : c->h2d) if (e) cudaEventDestroy(e);
    if (c->t0) cudaEventDestroy(c->t0); if (c->t1) cudaEventDestroy(c->t1);
    if (c->s_dst) cudaStreamDestroy(c->s_dst);
    for (auto p : c->host) if (p) cudaFreeHost(p);
    delete c;
}

}  // extern "C"
