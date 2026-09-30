"""RMSNorm en Triton + TLX frente a PyTorch (operador memory-bound). Imagen: triton-tlx.

Kernel: MBKernels/triton_tlx/rmsnorm.py (persistente, pipeline de --stages buffers en
memoria compartida con cp.async). Se compara con run_rmsnorm_triton (mismo computo, sin
prefetch explicito) en results/rmsnorm_metrics.csv. La columna 'detalle' guarda si el PTX
contiene cp.async (la copia asincrona que pide TLX) y la configuracion de lanzamiento.
Banco comun a todos los DSLs: benchmarks/banco_rmsnorm.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_rmsnorm_tlx [--stages 3] [--rows ...]
Resultados: results/run_rmsnorm_tlx/<fecha>_job<JOBID>/ y results/rmsnorm_metrics.csv
"""
import torch

import banco_rmsnorm
import comun
from MBKernels.triton_tlx.rmsnorm import configuracion, rmsnorm

if __name__ == "__main__":
    p = banco_rmsnorm.parser(__doc__)
    p.add_argument("--stages", type=int, default=3, help="buffers del pipeline (>= 2)")
    args = p.parse_args()

    def detalle(x, w):
        _, kernel = rmsnorm(x, w, args.eps, num_stages=args.stages, devolver_kernel=True)
        carga = comun.copias_asincronas(kernel)["carga"]
        _, num_warps, ctas = configuracion(x.shape[1], x.element_size(), args.stages)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        return f"carga={carga} stages={args.stages} num_warps={num_warps} grid={min(x.shape[0], sms * ctas)}"

    banco_rmsnorm.ejecutar(args, "tlx", lambda x, w, eps: rmsnorm(x, w, eps, num_stages=args.stages), detalle)
