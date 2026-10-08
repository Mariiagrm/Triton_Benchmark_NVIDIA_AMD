#!/bin/bash
#SBATCH -J sonda-attn
# Ejecuta la sonda en las 4 imagenes, en el repo del --chdir.
for d in triton-tlx gluon helion cutlass; do
    DSL=$d VALIDAR=0 bash ./ejecutar.sh exp .sonda/sonda_attention.py 2>&1 | grep -v "^== \(Raiz\|Estado\)"
done
