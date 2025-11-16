#include <cstdint>
#include <cuda_runtime.h>
#include <cuhash.hpp>

using SHA256_THREAD = decltype(cupqc::SHA2_256() + cupqc::Thread());

__device__ inline void store_le32(uint8_t* p, uint32_t x) {
    p[0] = static_cast<uint8_t>(x);
    p[1] = static_cast<uint8_t>(x >> 8);
    p[2] = static_cast<uint8_t>(x >> 16);
    p[3] = static_cast<uint8_t>(x >> 24);
}

__device__ inline void store_le64(uint8_t* p, uint64_t x) {
    p[0] = static_cast<uint8_t>(x);
    p[1] = static_cast<uint8_t>(x >> 8);
    p[2] = static_cast<uint8_t>(x >> 16);
    p[3] = static_cast<uint8_t>(x >> 24);
    p[4] = static_cast<uint8_t>(x >> 32);
    p[5] = static_cast<uint8_t>(x >> 40);
    p[6] = static_cast<uint8_t>(x >> 48);
    p[7] = static_cast<uint8_t>(x >> 56);
}

__device__ inline void hmac_sha256_8byte_key_two_msgs(
    const uint8_t key8[8],
    const uint8_t* msg1, size_t len1,
    const uint8_t* msg2, size_t len2,
    uint8_t out32[32]
) {
    uint8_t k0[64];
    for (int i = 0; i < 8; ++i) k0[i] = key8[i];
    for (int i = 8; i < 64; ++i) k0[i] = 0;

    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = static_cast<uint8_t>(k0[i] ^ 0x36);
        opad[i] = static_cast<uint8_t>(k0[i] ^ 0x5c);
    }

    uint8_t inner[32];
    {
        SHA256_THREAD h;
        h.reset();
        h.update(ipad, 64);
        h.update(msg1, len1);
        h.update(msg2, len2);
        h.finalize();
        h.digest(inner, 32);
    }
    {
        SHA256_THREAD h;
        h.reset();
        h.update(opad, 64);
        h.update(inner, 32);
        h.finalize();
        h.digest(out32, 32);
    }
}

__global__ void getY_kernel(
    const uint8_t* __restrict__ B,
    uint64_t bsz, uint64_t blen,
    uint64_t key, uint64_t salt, uint32_t m,
    float* __restrict__ Y
) {
    uint64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bsz || j >= bsz) return;

    float out = 0.0f;
    if (j > i) {
        uint32_t tkIdx = static_cast<uint32_t>(j / blen);
        uint32_t bitIdx = static_cast<uint32_t>(j % blen);

        uint8_t header[20];
        store_le64(header + 0, salt);
        store_le32(header + 8, tkIdx);
        store_le32(header + 12, bitIdx);
        store_le32(header + 16, m);

        uint8_t keyle[8];
        store_le64(keyle, key);

        uint8_t digest[32];
        hmac_sha256_8byte_key_two_msgs(keyle, header, 20, B, i, digest);

        uint32_t be = (uint32_t(digest[0]) << 24) | (uint32_t(digest[1]) << 16) |
                      (uint32_t(digest[2]) << 8) | uint32_t(digest[3]);
        out = float(double(be) / 4294967295.0);
    }
    Y[i * bsz + j] = out;
}

extern "C" void getY(
    void* B_ptr, uint64_t bsz, uint64_t blen,
    uint64_t key, uint64_t salt, uint32_t m,
    void* Y_ptr
) {
    const uint8_t* B = static_cast<const uint8_t*>(B_ptr);
    float* Y = static_cast<float*>(Y_ptr);

    uint8_t *d_B = nullptr;
    float *d_Y = nullptr;

    cudaMalloc(&d_B, bsz);
    cudaMalloc(&d_Y, sizeof(float) * bsz * bsz);
    cudaMemcpy(d_B, B, bsz, cudaMemcpyHostToDevice);
    cudaMemset(d_Y, 0, sizeof(float) * bsz * bsz);

    dim3 block(16, 16);
    dim3 grid((unsigned)((bsz + block.x - 1) / block.x),
              (unsigned)((bsz + block.y - 1) / block.y));

    getY_kernel<<<grid, block, 0>>>(d_B, bsz, blen, key, salt, m, d_Y);

    cudaMemcpy(Y, d_Y, sizeof(float) * bsz * bsz, cudaMemcpyDeviceToHost);
    cudaFree(d_B);
    cudaFree(d_Y);
}

__global__ void getYSingle_kernel(
    const uint8_t* __restrict__ B,
    uint64_t bsz, uint64_t blen,
    uint64_t key, uint64_t salt, uint32_t m,
    uint64_t rowIdx, uint64_t tkIdx, uint64_t bitIdx,
    float* __restrict__ out
) {
    uint64_t j = tkIdx * blen + bitIdx;
    float value = 0.0f;

    if (j > rowIdx) {
        uint8_t header[20];
        store_le64(header + 0, salt);
        store_le32(header + 8, static_cast<uint32_t>(tkIdx));
        store_le32(header + 12, static_cast<uint32_t>(bitIdx));
        store_le32(header + 16, m);

        uint8_t keyle[8];
        store_le64(keyle, key);

        uint8_t digest[32];
        hmac_sha256_8byte_key_two_msgs(keyle, header, 20, B, rowIdx, digest);

        uint32_t be = (uint32_t(digest[0]) << 24) | (uint32_t(digest[1]) << 16) |
                      (uint32_t(digest[2]) << 8) | uint32_t(digest[3]);
        value = float(double(be) / 4294967295.0);
    }
    out[0] = value;
}

extern "C" void getYSingle(
    void* B_ptr, uint64_t bsz, uint64_t blen,
    uint64_t key, uint64_t salt, uint32_t m,
    uint64_t rowIdx, uint64_t tkIdx, uint64_t bitIdx,
    void* out_ptr
) {
    const uint8_t* B = static_cast<const uint8_t*>(B_ptr);
    float* out = static_cast<float*>(out_ptr);

    uint8_t* d_B = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_B, bsz);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_B, B, bsz, cudaMemcpyHostToDevice);

    getYSingle_kernel<<<1, 1, 0>>>(d_B, bsz, blen, key, salt, m, rowIdx, tkIdx, bitIdx, d_out);

    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_B);
    cudaFree(d_out);
}
