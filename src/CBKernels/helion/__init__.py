"""Kernels compute-bound en Helion (imagen helion)."""


def config_elegida(kernel, args):
    """Configuracion que el autotuning de Helion eligio para estos argumentos (BoundKernel._config),
    como texto clave=valor para la columna 'detalle'. Vacio si no se puede leer."""
    try:
        cfg = kernel.bind(tuple(args))._config
        d = dict(cfg.config) if hasattr(cfg, "config") else dict(cfg)
    except Exception:
        return ""
    claves = ("block_sizes", "num_warps", "num_stages", "pid_type", "indexing")
    return " ".join(f"{k}={str(d[k]).replace(' ', '')}" for k in claves if k in d and d[k] is not None)
