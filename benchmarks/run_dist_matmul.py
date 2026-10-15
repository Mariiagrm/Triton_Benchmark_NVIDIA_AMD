"""Matmul en DOS nodos GB10 (p. ej. hennessy + patterson) unidos por el cable QSFP (RoCE 200 Gb/s).

Lo lanza 'ejecutar.sh dist' con torchrun: un proceso (rank) por nodo y NCCL entre ambos.
Rank 0 es el nodo del repo (alli quedan los resultados). Mide:
    1. all_reduce de NCCL (busbw en GB/s): comprueba que la red va por RoCE y no por TCP.
    2. GEMM M = N = K repartida por columnas de B (cada nodo calcula N/2 columnas de C):
         local        -> cada nodo la GEMM ENTERA, sin red (referencia de un nodo, cuBLAS)
         dist         -> cada nodo solo su mitad, sin comunicar (techo del reparto)
         dist_gather  -> su mitad + all_gather de C en ambos nodos (el coste real)
       TFLOP/s siempre sobre el trabajo TOTAL 2*M*N*K; potencia_w es la SUMA de los nodos.

No usa comun.medir(): do_bench decide cuantas repeticiones hace segun el tiempo de cada nodo
y, con colectivas, un numero distinto de llamadas por rank cuelga el trabajo. Aqui las
repeticiones son fijas e iguales en ambos ranks; el tiempo es el del rank mas lento.

Uso (desde el login):
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar dist run_dist_matmul [--sizes 8192 16384]
Resultados (solo rank 0): results/<nodo0>-<nodo1>/run_dist_matmul/<fecha>_job<JOBID>/
"""
import argparse
import datetime
import os

import torch
import torch.distributed as dist

import comun
import validation


def cronometrar(fn, iters, warmup):
    """ms por llamada (el del rank mas lento) y, por nodo, reloj SM y potencia durante la medida."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ini, fin = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with comun.Muestreador() as m:
        ini.record()
        for _ in range(iters):
            fn()
        fin.record()
        torch.cuda.synchronize()
    ms = torch.tensor([ini.elapsed_time(fin) / iters], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    por_nodo = [None] * dist.get_world_size()
    dist.all_gather_object(por_nodo, m.mediana())
    return round(ms.item(), 4), por_nodo


def fila_medida(ms, por_nodo):
    relojes = [r for r, _ in por_nodo]
    potencias = [p for _, p in por_nodo]
    return {"ms": ms, "reloj_mhz": min((r for r in relojes if r), default=None),
            "relojes_mhz": ";".join(map(str, relojes)),
            "potencia_w": round(sum(potencias), 1) if all(potencias) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[8192, 16384], help="M = N = K")
    p.add_argument("--dtypes", nargs="+", choices=list(comun.DTYPES), default=["fp16"])
    p.add_argument("--mib", type=int, nargs="+", default=[64, 256, 1024], help="tamanos del all_reduce")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Ambos nodos deben usar el mismo software: si no, la comparacion (y a veces NCCL) falla.
    yo = {"rank": rank, "host": os.environ.get("TFM_HOST"), "gpu": torch.cuda.get_device_name(0),
          "driver_nvidia": os.environ.get("TFM_DRIVER"), "kernel_linux": os.environ.get("TFM_KERNEL"),
          "imagen": os.environ.get("TFM_IMAGEN"), "torch": torch.__version__,
          "nccl": ".".join(map(str, torch.cuda.nccl.version())),
          "nccl_iface": os.environ.get("NCCL_SOCKET_IFNAME"), "nccl_ib_hca": os.environ.get("NCCL_IB_HCA")}
    nodos = [None] * world
    dist.all_gather_object(nodos, yo)
    for clave in ("torch", "nccl"):
        if len({n[clave] for n in nodos}) > 1:
            raise RuntimeError(f"{clave} distinto entre nodos: {[(n['host'], n[clave]) for n in nodos]}. "
                               f"Reconstruye la imagen en ambos ('encolar imagen').")
    if rank == 0:
        comun.imprimir_contexto()
        for n in nodos:
            print(f"   rank {n['rank']}: {n['host']} | driver {n['driver_nvidia']} | nccl {n['nccl']} | "
                  f"iface {n['nccl_iface']}", flush=True)
        print(flush=True)

    filas, resumen = [], {}

    # 1) all_reduce: busbw = bytes * 2(n-1)/n / t (convenio de nccl-tests)
    for mib in args.mib:
        x = torch.ones(mib * 2**20 // 2, dtype=torch.float16, device="cuda")
        ms, por_nodo = cronometrar(lambda: dist.all_reduce(x), args.iters, args.warmup)
        busbw = round(x.numel() * 2 / (ms * 1e-3) * 2 * (world - 1) / world / 1e9, 2)
        filas.append({"variante": "all_reduce", "dtype": "fp16", "MiB": mib, "nodos": world,
                      **fila_medida(ms, por_nodo), "gbs": busbw})
        if rank == 0:
            print(f"all_reduce {mib:5d} MiB: {ms:8.3f} ms  busbw {busbw:6.2f} GB/s", flush=True)
        resumen[f"all_reduce_{mib}MiB_gbs"] = busbw
    if rank == 0 and resumen[f"all_reduce_{args.mib[-1]}MiB_gbs"] < 5:
        print("AVISO: busbw < 5 GB/s: NCCL parece ir por TCP (NET/Socket), no por RoCE. Mira las "
              "lineas 'NET/' del log.", flush=True)

    # 2) GEMM repartida por columnas: misma semilla -> A y B iguales en ambos nodos
    for dt in args.dtypes:
        dtype = comun.DTYPES[dt]
        for n in args.sizes:
            M = N = K = n
            torch.manual_seed(0)
            a = torch.randn((M, K), device="cuda", dtype=dtype)
            b = torch.randn((K, N), device="cuda", dtype=dtype)
            b_mia = b[:, rank * N // world:(rank + 1) * N // world].contiguous()
            trozos = [torch.empty((M, N // world), device="cuda", dtype=dtype) for _ in range(world)]

            def gather():
                dist.all_gather(trozos, torch.matmul(a, b_mia))

            gather()
            validation.comprobar_matmul(torch.cat(trozos, dim=1), a, b, f"matmul distribuido {dt} {n}")
            flops = 2 * M * N * K
            tflops = {}
            for nombre, fn, nn in (("local", lambda: torch.matmul(a, b), 1),
                                   ("dist", lambda: torch.matmul(a, b_mia), world),
                                   ("dist_gather", gather, world)):
                ms, por_nodo = cronometrar(fn, args.iters, args.warmup)
                tflops[nombre] = round(flops / (ms * 1e-3) / 1e12, 2)
                filas.append({"variante": nombre, "dtype": dt, "M": M, "N": N, "K": K, "nodos": nn,
                              **fila_medida(ms, por_nodo), "tflops": tflops[nombre]})
                if rank == 0:
                    print(f"{dt} {M}x{N}x{K} [{nombre:11s}] {ms:8.3f} ms  {tflops[nombre]:7.2f} TFLOP/s", flush=True)
            acel = round(tflops["dist_gather"] / tflops["local"], 3)
            if rank == 0:
                print(f"   -> aceleracion real con {world} nodos (dist_gather/local): x{acel}\n", flush=True)
            resumen[f"{dt}_{n}"] = {**{f"{k}_tflops": v for k, v in tflops.items()}, "aceleracion": acel}
            del a, b, b_mia, trozos
            torch.cuda.empty_cache()

    if rank == 0:
        comun.guardar(filas, parametros={**vars(args), "nodos": nodos, "master": os.environ.get("MASTER_ADDR")},
                      resumen=resumen)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
