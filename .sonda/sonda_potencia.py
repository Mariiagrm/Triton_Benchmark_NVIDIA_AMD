"""Sonda: fiabilidad de la potencia de NVML en GB10. Desde reposo, lanza cuBLAS 8192^3 en bucle
4 s y muestrea cada 20 ms reloj SM, potencia instantanea (NVML_FI_DEV_POWER_INSTANT) y potencia
promedio (nvmlDeviceGetPowerUsage). Imprime medianas por tramos de 0.25 s.
Uso: VALIDAR=0 bash ejecutar.sh encolar exp .sonda/sonda_potencia.py
"""
import threading
import time

import pynvml
import torch

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
campo = getattr(pynvml, "NVML_FI_DEV_POWER_INSTANT", 186)


def leer():
    v = pynvml.nvmlDeviceGetFieldValues(h, [campo])[0]
    inst = v.value.uiVal / 1000 if v.nvmlReturn == 0 else None
    return (pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), inst, pynvml.nvmlDeviceGetPowerUsage(h) / 1000)


a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
torch.matmul(a, b)
torch.cuda.synchronize()
time.sleep(5)  # reposo
muestras, parar = [], threading.Event()


def bucle():
    t0 = time.perf_counter()
    while not parar.is_set():
        muestras.append((time.perf_counter() - t0, *leer()))
        time.sleep(0.02)


hilo = threading.Thread(target=bucle)
hilo.start()
time.sleep(0.5)  # 0.5 s de reposo dentro de la serie
t_fin = time.perf_counter() + 4
while time.perf_counter() < t_fin:
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
time.sleep(1.5)  # vuelta a reposo
parar.set()
hilo.join()
print(f"campo instantaneo disponible: {muestras[0][2] is not None}")
print(" tramo(s)   reloj  P_inst  P_prom   (carga entre 0.5 y 4.5 s)")
for i in range(int(muestras[-1][0] / 0.25) + 1):
    tr = [m for m in muestras if i * 0.25 <= m[0] < (i + 1) * 0.25]
    if tr:
        med = lambda k: sorted(x[k] for x in tr if x[k] is not None)[len(tr) // 2] if any(x[k] is not None for x in tr) else None
        print(f" {i * 0.25:5.2f}  {med(1):7}  {med(2)!s:>6}  {med(3)!s:>6}")
