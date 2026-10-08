#!/bin/bash
for e in ${EXPS:-run_attention_triton run_attention_helion run_attention_cutlass}; do
    TFM_MAQUINA=prueba-attn VALIDAR=0 bash ./ejecutar.sh exp $e --seqlens 1024 2048 --head-dims 64 128 2>&1 | grep -v "^== \(Raiz\|Estado\)"
done
