# Sondas

Scripts de diagnóstico de usar y tirar: comprueban una hipótesis concreta (si algo compila, por
qué un kernel rinde poco, si una medida es fiable). **No son benchmarks**: no guardan resultados en
`results/` ni entran en el resumen. Su salida queda en `logs/<nombre>-<JOBID>.out`, que no se
versiona. Por eso, cuando una conclusión de la memoria depende de una sonda, la salida se archiva
en `../../tfm/docs/TFM/resultados/sondas/` y la conclusión se anota en la
bitácora (`../../tfm/docs/TFM/bitacora.md`).

Se lanzan con el ejecutor estándar, eligiendo la imagen con `DSL=` y sin validación previa:

```bash
DSL=cutlass VALIDAR=0 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp .sonda/<sonda>.py
```

## Índice

| sonda | pregunta | job(s) | respuesta | salida archivada |
|:---|:---|:---|:---|:---|
| `sonda_attention.py` (+ `lanzar.sh`, `solo_tlx.sh`) | ¿qué FlashAttention oficiales funcionan en sm_120/121, imagen por imagen? | 20639, 20640 | Solo los basados en `mma.sync`: el de Gluon aborta (`tcgen05`), los de TLX usan `wgmma`/`tcgen05` y en CuTe solo sirve el FA2 de Ampere (bitácora 2026-10-05, punto 1) | — |
| `gluon_attention_forward.py` | copia del ejemplo oficial de atención de Gluon, para la sonda anterior | 20639, 20640 | `LLVM ERROR: Cannot select: intrinsic tcgen05.wait.st` | — |
| `tlx_hopper_fa_ws.py` | tutorial de atención con *warp specialization* de TLX, para la sonda anterior | — | Usa `wgmma`/`tcgen05`: no aplica a sm_120/121 | — |
| `prueba_bancos.sh` | prueba rápida de los bancos de atención (formas pequeñas, entorno `prueba-attn`) | 20645–20647 | Barrido de parámetros de CuTe necesario: con su configuración por defecto se queda en el ~60 % de SDPA (bitácora 2026-10-05, punto 3) | — |
| *(fp8 de CuTe, ya retiradas)* | ¿admite el `dense_gemm` de CuTe entradas FP8? | 20778–20781 | No: su MMA es fp16/bf16; el FP8 de CuTe en Blackwell GeForce es block-scaled MXFP8 (bitácora 2026-10-06; commits `6b62823`, `9a267a9`, revertidas en `f18987f`) | — |
| `sonda_cute_clusters.py` | ¿rinde el matmul de CuTe el 24 % de cuBLAS en GB10 porque lanza pocos clusters persistentes? | 20844 | **No.** Forzando 48, 96 o 192 clusters da 23.6–23.9 TFLOP/s. Además, `HardwareInfo.get_max_active_clusters` falla con `CUDA_ERROR_INVALID_CONTEXT` si se llama antes de crear el contexto | `../../tfm/docs/TFM/resultados/sondas/sonda_cute_clusters-20844.txt` |
| `sonda_cute_tiles.py` | ¿mejora el matmul de CuTe en GB10 cambiando el tile CTA? | 20846 | Algo: el mejor es 128×256×64 con 37.8 TFLOP/s (31 % del pico), lejos de cuBLAS (96). Causa abierta | `../../tfm/docs/TFM/resultados/sondas/sonda_cute_tiles-20846.txt` |
| `sonda_potencia.py` | ¿es fiable la potencia de NVML en GB10? ¿Cómo evoluciona el reloj bajo carga? | 20845 | La de `nvmlDeviceGetPowerUsage` va ~1 s retrasada; la instantánea (`NVML_FI_DEV_POWER_INSTANT`), no. NVML actualiza cada ~0.5 s. El reloj baja de 2 410 a ~2 180 MHz tras ~0.5 s de GEMM | `../../tfm/docs/TFM/resultados/sondas/sonda_potencia-20845.txt` |
| `sonda_fp8_cublas.py` | ¿Por qué cuBLASLt FP8 supera en la RTX 5090 el pico de ficha con acumulación FP32 (419 TFLOP/s)? ¿Por qué cuBLAS fp16 dio 168 en vez de 209 el 4-oct? | 20852 (pascal, en cola) | Pendiente: kernel de cuBLASLt, reloj, potencia, límite de potencia y `use_fast_accum` | — |

Relacionado, aunque no es una sonda: el primer intento de `run_pico_mma` (job 20833) falló porque
`nvcc -arch=sm_121a -shared` genera PTX para `compute_121`. Resumen de sus errores de ptxas:
`../../tfm/docs/TFM/resultados/sondas/run_pico_mma-20833.txt`.
