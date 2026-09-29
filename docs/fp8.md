# Reducción de precisión: FP8 en los tensor cores

*Sección consolidada para la memoria del TFM. Datos: `results/experimento_fp8/ultimo/` (job 20399). Hardware: NVIDIA GB10 (sm_121), torch 2.9, triton 3.8.0, CUDA 13.0.*

## Motivación

La sección de TMA (§ [docs/tma.md](tma.md)) concluye que el matmul denso en FP16 está pegado a ~90–100 TFLOP/s y que ese techo **no lo pone la memoria sino la unidad MMA** de sm_121: activar TMA no acelera. La única palanca que queda para subir el techo de los tensor cores en esta GPU es **cambiar de precisión**. Los tensor cores de Blackwell ejecutan FP8 al doble de throughput que FP16, así que se estudia si un kernel FP8 rompe la barrera y a qué coste de precisión.

## Implementación

Se añade un kernel (`src/triton_kernels/matmul_fp8.py`) idéntico al baseline salvo en el tipo de las entradas: A y B en **FP8 e4m3** (`torch.float8_e4m3fn`), `tl.dot` con acumulación en FP32 y salida en FP16. Mantener el resto igual (swizzling L2, esquema de bloques) aísla el efecto de la precisión. Se compara con:

- **fp16**: baseline Triton (§ baseline).
- **fp8 (Triton)**: el kernel anterior.
- **fp8 (cuBLASLt)**: `torch._scaled_mm`, la ruta optimizada de NVIDIA, como referencia superior.

El error se cuantifica como error relativo de Frobenius frente al resultado **exacto en FP32** de las mismas matrices lógicas.

## Resultados

![Matmul FP8 vs FP16 en GB10](resultados/figuras/experimento_fp8.png)

| tamaño | fp16 (TFLOP/s) | fp8 Triton | fp8 cuBLASLt | speedup fp8/fp16 | error fp8 |
|:---|---:|---:|---:|---:|---:|
| 4096³ | 93.8 | 147.7 | 172.3 | +57 % | 3.75 % |
| 8192³ | 92.4 | 150.0 | 192.3 | +62 % | 3.75 % |

*Error FP16 de referencia: 0.02 %. Tabla completa: `resultados/experimento_fp8.md`.*

## Análisis

1. **FP8 rompe la barrera de los ~100 TFLOP/s** que el matmul denso FP16 no rebasaba: 150 TFLOP/s con el kernel Triton y hasta 192 con cuBLASLt. Esto **confirma la hipótesis de la sección TMA**: el límite en FP16 era el cómputo (la MMA), no el transporte de datos.
2. **El origen del throughput es la instrucción MMA.** En FP8 el PTX emite `mma.sync.aligned.m16n8k32...e4m3.e4m3.f32`: el mismo tipo de MMA que en FP16 pero con **K = 32** (el doble que el `m16n8k16` de FP16), que es exactamente de donde sale el factor de aceleración.
3. **El coste es de precisión, no de estabilidad.** El error relativo sube a 3.75 % (frente a 0.02 % en FP16) por los ~3 bits de mantisa de e4m3, pero es estable y acotado, y para las mismas entradas es idéntico en Triton y cuBLASLt (es un límite del formato, no del kernel).
4. **El kernel Triton deja margen frente a cuBLASLt** (150 vs 192 TFLOP/s ≈ 78 %). Además, el +62 % no alcanza el 2× teórico: hay sobrecoste de *pipelining* y un espacio de *autotuning* no óptimo. Como en FP8 se mueve la mitad de bytes, el kernel puede acercarse a ser *memory-bound*, y ahí el TMA por descriptores (§ TMA), que en FP16 no aportaba, sí podría rendir.

## Intento de cerrar el hueco: FP8 + TMA (job 20400)

El kernel Triton FP8 alcanza ~78–85 % de cuBLASLt. La hipótesis era que, al mover FP8 la
mitad de bytes, el kernel podría volverse *memory-bound* y el TMA por descriptores
—inútil en FP16 (§ TMA)— sí ayudaría. Se implementa `matmul_fp8_tma` (FP8 + descriptores
`tl.make_tensor_descriptor`, autotuning ampliado: `BLOCK_K` hasta 256, `num_stages` hasta 5).

| tamaño | fp8 | fp8+TMA | cuBLASLt | fp8+TMA / cuBLASLt |
|:---|---:|---:|---:|---:|
| 4096³ | 147.7 | 150.1 (`carga=TMA`) | 176.7 | 85.0 % |
| 8192³ | 148.3 | 150.5 (`carga=TMA`) | 182.3 | 82.5 % |

El kernel **emite TMA de verdad** (verificado en PTX) pero solo gana **+1.5 %**: la hipótesis
es incorrecta, FP8 tampoco está limitado por la memoria. El *autotuner* ni siquiera eligió los
bloques grandes ni más *stages* (se quedó en `BLOCK_K` 64–128, `stages` 3–4), lo que descarta
también tiling y *pipelining* como cuello. **El hueco restante (~18 %) no está en el movimiento
de datos ni en el tiling, sino en el *scheduling*** que usa cuBLASLt (kernels persistentes,
especialización de warps, epílogo optimizado), fuera del alcance del kernel de una onda por tile.

## Conclusión

La reducción a FP8 e4m3 es la técnica que **efectivamente sube el techo** de los tensor cores en GB10 (+57–62 %), a cambio de un error relativo acotado del ~3.75 %. Es coherente con que la MMA fuese el cuello en FP16. El kernel Triton llega a ~83 % de cuBLASLt y el hueco **no** se cierra con TMA (es un problema de *scheduling*, no de datos): cerrarlo requeriría un kernel persistente, línea que se deja indicada. La vía con mayor recorrido para subir el techo es la **estructura dispersa 2:4**, única que puede acercarse a la cifra de catálogo (§ sparsity).

**Reproducir:** `sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp experimento_fp8 --sizes 4096 8192`
