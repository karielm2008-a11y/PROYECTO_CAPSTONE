# Comprobación del paquete — 8 de octubre de 2026

- El script y requirements.txt se copiaron sin modificar desde la entrega ampliada.
- Se verificaron la sintaxis, la importación y la ejecución de `--help`.
- Se regeneraron satisfactoriamente las 12 figuras originales y la matriz ponderada adicional desde las tablas agregadas conservadas, en una carpeta temporal.
- La figura de sensibilidad y sus tablas se incluyen desde la ejecución anterior; en esta preparación no se reentrenaron los modelos de sensibilidad ni los modelos principales.
- Se comprobó que las diez filas del CSV de verificación conservado tienen `coincide_precision_publicada=True`.
- La ejecución completa previa y la reejecución de las funciones añadidas están documentadas en `resultados_referencia/validacion_ampliacion.json`. El hash de `environment.json` corresponde a la versión ejecutada antes del último ajuste visual de sensibilidad; el hash de la entrega final consta en `validacion_ampliacion.json` y coincide con el script incluido.
- Las tablas y PNG son resultados históricos agregados; los registros y rutas internas de esos archivos se preservan como procedencia. No son dependencias de ejecución.
- Esta edición incluye los ZIP originales de microdatos y tabulados. Se verificó la integridad del ZIP de microdatos y la huella SHA-256 de su CSV de personas, que coincide con la base utilizada en el estudio. No se incluyen predicciones individuales ni modelos serializados.

La preparación del paquete no constituye una nueva reproducción integral. Para realizarla, ejecutar el comando del README con el ZIP original y revisar el nuevo CSV de verificación.
