# Resultados descartados

`hennessy/*/20261014-16*_job21058`: intento del trabajo 21058 que empezo a las 18:00 (hora local).
Al arrancar, la GPU de hennessy ya tenia dos procesos de intentos anteriores que siguieron
corriendo tras caerse patterson (el log lo avisa: "hay procesos usando la GPU"). Las medidas
salen a ~40 % de lo normal (matmul fp16 ~41 TFLOP/s frente a ~92 en el resto de intentos).
Se conservan aqui, fuera de results/, para que no entren en las tablas ni en el resumen.
