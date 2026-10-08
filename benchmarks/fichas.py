"""Picos teoricos de la FICHA TECNICA de NVIDIA por GPU: la linea horizontal de las graficas.

Solo cifras publicadas por NVIDIA, nunca derivadas:
    TFLOP/s   tensor cores, DENSO (sin sparsity 2:4) y con acumulacion en FP32, que es como
              acumulan todos los kernels del TFM (matmul, FlashAttention, FP8 incluido)
    GB/s      ancho de banda de la memoria principal

GeForce RTX 5090: NVIDIA RTX Blackwell GPU Architecture (whitepaper), tabla 3 (pags. 46-47),
    a la frecuencia boost de la ficha (2 407 MHz):
    https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
GB10 (DGX Spark): DGX Spark User Guide, "Hardware Overview":
    https://docs.nvidia.com/dgx/dgx-spark/hardware.html
    NVIDIA solo publica "1 PFLOP FP4 con sparsity", 273 GB/s y 140 W de TDP del SoC: no hay pico
    oficial de fp16 ni de FP8, asi que en GB10 solo se dibuja el de memoria.
"""

FICHAS = {
    "NVIDIA GeForce RTX 5090": {
        "fuente": "whitepaper RTX Blackwell, tabla 3",
        "TFLOP/s": {"fp16": 209.5, "bf16": 209.5, "fp8": 419.0, "tf32": 104.8},
        "GB/s": 1792.0,
    },
    "NVIDIA GB10": {
        "fuente": "DGX Spark User Guide, Hardware Overview",
        "TFLOP/s": {},
        "GB/s": 273.0,
    },
}

COLOR = "#C0392B"


def ficha(gpu):
    """Ficha de una GPU por su nombre (admite el de resumen, p. ej. 'NVIDIA GB10 (sm_121)')."""
    gpu = (gpu or "").split(" (sm_")[0]
    return FICHAS.get(gpu)


def pico(gpu, unidad, precision="fp16"):
    """(valor, etiqueta) del pico de ficha, o None si NVIDIA no lo publica para esa GPU."""
    f = ficha(gpu)
    if f is None:
        return None
    if unidad == "GB/s":
        v = f.get("GB/s")
        return (v, f"Pico ficha NVIDIA: {v:g} GB/s") if v else None
    v = f.get("TFLOP/s", {}).get(precision.lower())
    return (v, f"Pico ficha NVIDIA {precision.upper() if precision == 'fp8' else precision}: {v:g} TFLOP/s") if v else None


def precision_de(texto):
    """Precision de una columna o experimento: 'Matmul FP8' / 'run_matmul_fp8' -> fp8; si no, fp16."""
    t = (texto or "").lower()
    for p in ("fp8", "bf16", "tf32"):
        if p in t:
            return p
    return "fp16"


def dibujar(eje, valor, etiqueta, x0=None, x1=None, texto=True, estilo="--"):
    """Linea del pico (discontinua; estilo=":" para distinguir una segunda precision en el mismo
    eje); entre x0 y x1 (datos) o a todo el ancho del eje."""
    if x0 is None:
        eje.axhline(valor, color=COLOR, linestyle=estilo, linewidth=1.3, zorder=4, label=etiqueta)
        x_txt = eje.get_xlim()[0]
    else:
        eje.hlines(valor, x0, x1, colors=COLOR, linestyles=estilo, linewidth=1.6, zorder=4, label=etiqueta)
        x_txt = x0
    if texto:
        eje.annotate(f"{valor:g}", (x_txt, valor), xytext=(2, 2), textcoords="offset points",
                     fontsize=8, color=COLOR, va="bottom", ha="left", zorder=5)
    lo, hi = eje.get_ylim()
    if valor * 1.08 > hi:
        eje.set_ylim(lo, valor * 1.12)
