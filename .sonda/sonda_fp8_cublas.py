"""Sonda: ¿por qué cuBLASLt FP8 (torch._scaled_mm) supera el pico de ficha de la RTX 5090 con
acumulacion en FP32 (419 TFLOP/s; job 20499: 599-624)?

Hipotesis: (a) el kernel acumula en FP16 (ficha: 838); (b) el reloj real supera con creces el boost
de ficha (2 407 MHz); (c) en sm_120 la acumulacion FP32 no va a mitad de velocidad.
Para cada variante de 8192^3: nombre del kernel CUDA (dice el tipo de acumulacion), TFLOP/s de
do_bench y en regimen sostenido, reloj SM y potencia. use_fast_accum=True como contraste.
La respuesta definitiva la da run_pico_mma (e4m3·f32acc frente a e4m3·f16acc).

Uso: VALIDAR=0 bash ejecutar.sh encolar exp .sonda/sonda_fp8_cublas.py
"""
import time

import torch

import comun


def sostenido(fn, segundos, descarte):
    """Copia de comun.sostenido (el clon de pascal aun no la tiene): (llamadas/s, reloj, potencia)."""
    fn()
    torch.cuda.synchronize()
    with comun.Muestreador() as m:
        t0 = time.perf_counter()
        n, t_ini = 0, None
        while True:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            ahora = time.perf_counter() - t0
            if t_ini is None and ahora >= descarte:
                t_ini, n = ahora, 0
            elif t_ini is not None:
                n += 20
            if ahora >= segundos:
                break
    reloj, potencia = m.mediana(desde=t_ini)
    return n / (ahora - t_ini), reloj, potencia


comun.imprimir_contexto()
n = 8192
flops = 2 * n ** 3
torch.manual_seed(0)
a = torch.randn((n, n), device="cuda", dtype=torch.float16)
b = torch.randn((n, n), device="cuda", dtype=torch.float16)
a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
b8_col = b8.t().contiguous().t()
s = torch.tensor(1.0, device="cuda")
m = comun.Muestreador()
print(f"reloj SM maximo (NVML): {m.reloj_maximo()} MHz; boost de ficha: 2407 MHz")
# Limites que pueden frenar la GPU (cuBLAS fp16 dio 209 TFLOP/s el 2-oct, job 20499, y 168 el 4-oct
# con el mismo kernel, job 20638): limite de potencia frente al de fabrica (TGP 575 W), driver.
import subprocess  # noqa: E402

print(subprocess.run(["nvidia-smi", "--query-gpu=driver_version,power.limit,power.default_limit,power.max_limit,"
                      "clocks.max.sm,clocks.applications.graphics,clocks_throttle_reasons.active,pstate",
                      "--format=csv"], capture_output=True, text=True).stdout)

casos = {
    "cuBLAS fp16 (torch.matmul)": lambda: torch.matmul(a, b),
    "cuBLASLt FP8 use_fast_accum=False": lambda: torch._scaled_mm(a8, b8_col, scale_a=s, scale_b=s,
                                                                  out_dtype=torch.float16, use_fast_accum=False),
    "cuBLASLt FP8 use_fast_accum=True": lambda: torch._scaled_mm(a8, b8_col, scale_a=s, scale_b=s,
                                                                 out_dtype=torch.float16, use_fast_accum=True),
}
for nombre, fn in casos.items():
    try:
        kernels = comun.kernels_cuda(fn)
        r = comun.medir(fn, flops=flops)
        ritmo, reloj, pot = sostenido(fn, 6, 2)
    except Exception as e:
        print(f"== {nombre}: ERROR {type(e).__name__}: {e}\n")
        continue
    print(f"== {nombre}")
    print(f"   kernels: {'; '.join(kernels)}")
    print(f"   do_bench: {r['tflops']} TFLOP/s a {r['reloj_mhz']} MHz")
    print(f"   sostenido: {ritmo * flops / 1e12:.1f} TFLOP/s a {reloj} MHz, {pot} W")
    if reloj:
        print(f"   -> {ritmo * flops / 1e12 / (reloj / 2407):.1f} TFLOP/s escalado al boost de ficha (2407 MHz)\n")
