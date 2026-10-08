#!/bin/bash
DSL=triton-tlx VALIDAR=0 bash ./ejecutar.sh exp .sonda/sonda_attention.py 2>&1 | grep -v "^== \(Raiz\|Estado\)"
