// Microbenchmark del pico de los tensor cores: bucles de mma.sync SIN accesos a memoria.
//
// Cada warp carga una vez sus fragmentos A y B (de un buffer con datos aleatorios del tipo
// correspondiente, para que la potencia sea la de datos reales y no la de ceros) y ejecuta
// `iters` x ILP instrucciones mma.sync sobre ILP acumuladores independientes. Solo hay una
// escritura final por hilo. Asi el unico limite es el ritmo de la unidad MMA: el techo que
// pueden alcanzar los kernels de los DSLs en sm_120/sm_121 (que emiten esta misma MMA).
//
// Cada instruccion va en su propio bloque #ifdef CON_<NOMBRE>: pico_mma.py compila antes
// cada una por separado y solo activa las que ptxas acepta para la GPU (las block-scaled de
// FP4/MXFP8 exigen la variante especifica de la arquitectura, sm_120a/sm_121a).
//
// Interfaz C (ctypes): disponible(v), lanzar(v, ilp, bloques, hilos, iters, buf, out, stream).
#include <cstdint>
#include <cuda_runtime.h>

// Identificadores de variante (deben coincidir con VARIANTES en pico_mma.py).
enum {
    F16_F32 = 0,   // m16n8k16 f16 x f16 -> f32
    F16_F16 = 1,   // m16n8k16 f16 x f16 -> f16
    BF16_F32 = 2,  // m16n8k16 bf16 x bf16 -> f32
    TF32_F32 = 3,  // m16n8k8  tf32 x tf32 -> f32
    E4M3_F32 = 4,  // m16n8k32 e4m3 x e4m3 -> f32
    E4M3_F16 = 5,  // m16n8k32 e4m3 x e4m3 -> f16
    S8_S32 = 6,    // m16n8k32 s8 x s8 -> s32
    F8F6F4_E2M1 = 7,      // kind::f8f6f4 m16n8k32 e2m1 (FP4 en contenedor de 8 bits) -> f32
    MXF8_E4M3 = 8,        // kind::mxf8f6f4 block_scale 1X m16n8k32 e4m3 (MXFP8) -> f32
    MXF4_E2M1 = 9,        // kind::mxf4nvf4 block_scale 2X m16n8k64 e2m1 (MXFP4, ue8m0) -> f32
    NVF4_E2M1 = 10,       // kind::mxf4nvf4 block_scale 4X m16n8k64 e2m1 (NVFP4, ue4m3) -> f32
    N_VARIANTES = 11
};

// Acumulador: 4 registros f32 (por defecto), 4 s32 (int8) o 2 f16x2 (acumulacion en f16).
template <int V> struct Acc { using T = float; static constexpr int R = 4; };
template <> struct Acc<S8_S32> { using T = uint32_t; static constexpr int R = 4; };
template <> struct Acc<F16_F16> { using T = uint32_t; static constexpr int R = 2; };
template <> struct Acc<E4M3_F16> { using T = uint32_t; static constexpr int R = 2; };

// Una instruccion MMA. a: 4 x b32, b: 2 x b32, c: Acc<V>::R registros (in/out).
template <int V> __device__ __forceinline__ void mma(typename Acc<V>::T* c, const uint32_t* a, const uint32_t* b);

// K: restriccion del acumulador, "+f" (f32) o "+r" (s32).
#define MMA_ACC4(INSTR, K)                                                                \
    asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"        \
                 : K(c[0]), K(c[1]), K(c[2]), K(c[3])                                      \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#define MMA_ACC2(INSTR)                                                                   \
    asm volatile(INSTR " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"                    \
                 : "+r"(c[0]), "+r"(c[1])                                                  \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
// Block-scaled: factores de escala de A y B (un registro cada uno) con byte-id/thread-id = 0.
#define MMA_ESCALADO(INSTR)                                                               \
    do {                                                                                  \
        const uint16_t cero = 0;                                                          \
        asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3},"      \
                     " {%10}, {%11,%12}, {%13}, {%14,%15};\n"                              \
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])                      \
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),   \
                       "r"(a[0]), "h"(cero), "h"(cero), "r"(b[0]), "h"(cero), "h"(cero));  \
    } while (0)

// Las instrucciones que no se compilan se quedan sin cuerpo (disponible() devuelve 0).
template <int V> __device__ __forceinline__ void mma(typename Acc<V>::T*, const uint32_t*, const uint32_t*) {}

#ifdef CON_F16_F32
template <> __device__ __forceinline__ void mma<F16_F32>(typename Acc<F16_F32>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC4("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "+f");
}
#endif
#ifdef CON_F16_F16
template <> __device__ __forceinline__ void mma<F16_F16>(typename Acc<F16_F16>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC2("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16");
}
#endif
#ifdef CON_BF16_F32
template <> __device__ __forceinline__ void mma<BF16_F32>(typename Acc<BF16_F32>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC4("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "+f");
}
#endif
#ifdef CON_TF32_F32
template <> __device__ __forceinline__ void mma<TF32_F32>(typename Acc<TF32_F32>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC4("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", "+f");
}
#endif
#ifdef CON_E4M3_F32
template <> __device__ __forceinline__ void mma<E4M3_F32>(typename Acc<E4M3_F32>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC4("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", "+f");
}
#endif
#ifdef CON_E4M3_F16
template <> __device__ __forceinline__ void mma<E4M3_F16>(typename Acc<E4M3_F16>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC2("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16");
}
#endif
#ifdef CON_S8_S32
template <> __device__ __forceinline__ void mma<S8_S32>(typename Acc<S8_S32>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC4("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", "+r");
}
#endif
#ifdef CON_F8F6F4_E2M1
template <> __device__ __forceinline__ void mma<F8F6F4_E2M1>(typename Acc<F8F6F4_E2M1>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ACC4("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32", "+f");
}
#endif
#ifdef CON_MXF8_E4M3
template <> __device__ __forceinline__ void mma<MXF8_E4M3>(typename Acc<MXF8_E4M3>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ESCALADO("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0");
}
#endif
#ifdef CON_MXF4_E2M1
template <> __device__ __forceinline__ void mma<MXF4_E2M1>(typename Acc<MXF4_E2M1>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ESCALADO("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0");
}
#endif
#ifdef CON_NVF4_E2M1
template <> __device__ __forceinline__ void mma<NVF4_E2M1>(typename Acc<NVF4_E2M1>::T* c, const uint32_t* a, const uint32_t* b) {
    MMA_ESCALADO("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3");
}
#endif

// iters x ILP MMAs por warp sobre ILP acumuladores independientes (ocultan la latencia).
template <int V, int ILP>
__global__ void bucle_mma(int iters, const uint32_t* __restrict__ buf, uint32_t* __restrict__ out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    using T = typename Acc<V>::T;
    uint32_t a[4], b[2];
    T c[ILP][Acc<V>::R];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = buf[(t * 8 + i) & 4095];
#pragma unroll
    for (int i = 0; i < 2; ++i) b[i] = buf[(t * 8 + 4 + i) & 4095];
#pragma unroll
    for (int j = 0; j < ILP; ++j)
#pragma unroll
        for (int r = 0; r < Acc<V>::R; ++r) c[j][r] = 0;

    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < ILP; ++j) mma<V>(c[j], a, b);
    }

    uint32_t x = 0;
#pragma unroll
    for (int j = 0; j < ILP; ++j)
#pragma unroll
        for (int r = 0; r < Acc<V>::R; ++r) {
            T v = c[j][r];
            x ^= *reinterpret_cast<uint32_t*>(&v);
        }
    out[t] = x;  // mantiene vivos los acumuladores
}

template <int V>
static cudaError_t lanzar_v(int ilp, int bloques, int hilos, int iters, const uint32_t* buf, uint32_t* out,
                            cudaStream_t s) {
    switch (ilp) {
        case 1: bucle_mma<V, 1><<<bloques, hilos, 0, s>>>(iters, buf, out); break;
        case 2: bucle_mma<V, 2><<<bloques, hilos, 0, s>>>(iters, buf, out); break;
        case 4: bucle_mma<V, 4><<<bloques, hilos, 0, s>>>(iters, buf, out); break;
        case 8: bucle_mma<V, 8><<<bloques, hilos, 0, s>>>(iters, buf, out); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

extern "C" {

int disponible(int v) {
    switch (v) {
#ifdef CON_F16_F32
        case F16_F32: return 1;
#endif
#ifdef CON_F16_F16
        case F16_F16: return 1;
#endif
#ifdef CON_BF16_F32
        case BF16_F32: return 1;
#endif
#ifdef CON_TF32_F32
        case TF32_F32: return 1;
#endif
#ifdef CON_E4M3_F32
        case E4M3_F32: return 1;
#endif
#ifdef CON_E4M3_F16
        case E4M3_F16: return 1;
#endif
#ifdef CON_S8_S32
        case S8_S32: return 1;
#endif
#ifdef CON_F8F6F4_E2M1
        case F8F6F4_E2M1: return 1;
#endif
#ifdef CON_MXF8_E4M3
        case MXF8_E4M3: return 1;
#endif
#ifdef CON_MXF4_E2M1
        case MXF4_E2M1: return 1;
#endif
#ifdef CON_NVF4_E2M1
        case NVF4_E2M1: return 1;
#endif
        default: return 0;
    }
}

// Devuelve el cudaError_t del lanzamiento (0 = correcto).
int lanzar(int v, int ilp, int bloques, int hilos, int iters, const void* buf, void* out, void* stream) {
    if (!disponible(v)) return cudaErrorInvalidValue;
    auto b = static_cast<const uint32_t*>(buf);
    auto o = static_cast<uint32_t*>(out);
    auto s = static_cast<cudaStream_t>(stream);
    switch (v) {
        case F16_F32: return lanzar_v<F16_F32>(ilp, bloques, hilos, iters, b, o, s);
        case F16_F16: return lanzar_v<F16_F16>(ilp, bloques, hilos, iters, b, o, s);
        case BF16_F32: return lanzar_v<BF16_F32>(ilp, bloques, hilos, iters, b, o, s);
        case TF32_F32: return lanzar_v<TF32_F32>(ilp, bloques, hilos, iters, b, o, s);
        case E4M3_F32: return lanzar_v<E4M3_F32>(ilp, bloques, hilos, iters, b, o, s);
        case E4M3_F16: return lanzar_v<E4M3_F16>(ilp, bloques, hilos, iters, b, o, s);
        case S8_S32: return lanzar_v<S8_S32>(ilp, bloques, hilos, iters, b, o, s);
        case F8F6F4_E2M1: return lanzar_v<F8F6F4_E2M1>(ilp, bloques, hilos, iters, b, o, s);
        case MXF8_E4M3: return lanzar_v<MXF8_E4M3>(ilp, bloques, hilos, iters, b, o, s);
        case MXF4_E2M1: return lanzar_v<MXF4_E2M1>(ilp, bloques, hilos, iters, b, o, s);
        case NVF4_E2M1: return lanzar_v<NVF4_E2M1>(ilp, bloques, hilos, iters, b, o, s);
        default: return cudaErrorInvalidValue;
    }
}

const char* nombre_error(int e) { return cudaGetErrorString(static_cast<cudaError_t>(e)); }

}  // extern "C"
