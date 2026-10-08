# Informalidad en Ecuador — ENEMDU anual 2025

Código del análisis de la tesis de Maestría en Inteligencia de Negocios y Ciencia de Datos, UDLA. Compara regresión logística y Random Forest para estimar la pertenencia al sector informal de la población ocupada de 15 años o más. Su alcance es predictivo y asociativo. La organización de referencia es el Ministerio del Trabajo del Ecuador.

## Contenido

- `analisis_enemdu_2025.py`: script completo e independiente. Incluye preparación, entrenamiento, evaluación, sensibilidad, odds ratios y gráficos. Se conserva sin cambios respecto de la entrega ampliada del 27 de septiembre de 2026.
- `requirements.txt`: versiones fijadas del entorno de reproducción.
- `data/raw/`: incluye el ZIP original de los microdatos anuales 2025 (personas, vivienda, diccionarios y metadatos) y el ZIP de tabulados oficiales. Ambos ZIP están permitidos en `.gitignore`.
- `resultados_referencia/`: tablas agregadas, figuras y registros de la ejecución anterior, para comparación. No se utilizan como entrada para entrenar.
- `VALIDACION.md`: alcance de la comprobación realizada al preparar este paquete.
- `manifest_sha256.csv`: huellas de los archivos de esta entrega.

El análisis se concentra en un solo script para evitar versiones divergentes entre notebooks y módulos históricos. No requiere ningún notebook, modelo preentrenado ni resultado previo.

## Instalación

Se utilizó Python 3.12.14. Desde la carpeta del proyecto:

```bash
python -m venv .venv
```

Activar el entorno en Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Datos y ejecución completa

La base ya está incluida: **no es necesario descargarla ni descomprimirla** para ejecutar el análisis. Se conserva el ZIP original del INEC, sin modificar su contenido. Incluye la base de personas, la base de vivienda y sus diccionarios y metadatos. También se adjuntan los tabulados oficiales como material de consulta; el análisis utiliza el CSV de personas del ZIP de microdatos.

Fuente: [INEC, ENEMDU anual](https://www.ecuadorencifras.gob.ec/enemdu-anual/). La institución conserva la autoría de estos datos; incluirlos en este repositorio no les asigna una licencia nueva.

Después de instalar las dependencias, ejecutar desde la carpeta del proyecto:

```bash
python analisis_enemdu_2025.py --data "data/raw/2_BDD_DATOS_ABIERTOS_ENEMDU_2025_CSV.zip" --output resultados_enemdu
```

También se admite el CSV de personas directamente:

```bash
python analisis_enemdu_2025.py --data "data/raw/BDDenemdu_personas_2025_anual.csv" --output resultados_enemdu
```

La carpeta de salida debe estar vacía o no existir. Para repetir la ejecución sin sobrescribir, utilizar otra carpeta y excluirla también de Git. El script vuelve a entrenar los modelos y genera localmente los datos procesados, modelos, tablas, gráficos, entorno y huellas SHA-256.

SHA-256 del CSV de personas utilizado en el estudio:

`da372d29cae3cbf8ab121c1359fa0895144a1f74b68cc38d1cc332d4a3a022c4`

## Procedimiento implementado

1. Lectura, limpieza, controles y huellas SHA-256. Universo ocupado de 15 años o más: `secemp=1` se recodifica como formal (0) y `secemp=2` como informal (1). Se excluyen empleo doméstico y no clasificados.
2. Partición de entrenamiento/prueba agrupada por `upm + panelm + vivienda`, semilla 2025; cinco folds agrupados dentro del entrenamiento.
3. Pesos `fexp` normalizados a media uno durante el ajuste; pesos originales en las métricas ponderadas.
4. Validación de candidatos y entrenamiento principal: logística con C=1; Random Forest con 150 árboles, profundidad máxima 20, mínimo 20 observaciones por hoja y `max_features='sqrt'`.
5. Discriminación, AP (average precision), accuracy, precisión, sensibilidad, especificidad, F1, matrices de confusión, Brier y calibración. Umbrales explorados con predicciones fuera de fold del entrenamiento.
6. Bootstrap pareado de 300 remuestreos de viviendas de prueba, semilla 2026; importancia por permutación; desempeño por área y sexo; asociaciones mediante V de Cramér ponderada.
7. Sensibilidad con conjuntos contextual, principal y ampliado, sin nueva optimización. El contextual incluye sexo, edad, estado civil, etnia, instrucción, área y provincia; el principal añade rama y ocupación; el ampliado añade categoría ocupacional (`p42`).
8. Contrastes de odds ratios y etiquetas; exportación de todas las figuras.

## Resultados de referencia

Estos valores proceden de la ejecución conservada, no de un entrenamiento nuevo realizado para empaquetar el repositorio.

| Indicador | Logística | Random Forest |
|---|---:|---:|
| ROC-AUC principal | 0,8705 | 0,8874 |
| Brier principal | 0,1433 | 0,1365 |

Universo: 154.330 observaciones; entrenamiento: 123.613; prueba: 30.717. Proporción informal ponderada del universo binario: 53,92 %. IC del 95 % de la diferencia de AUC (RF menos logística): [0,0121; 0,0215].

La comprobación automática se guarda en `outputs/tables/verificacion_tesis.csv` dentro de la carpeta de salida. La coincidencia se evalúa a la precisión publicada en la tesis; no equivale a igualdad con cifras redondeadas antes del cálculo. Las cifras de referencia no intervienen en la estimación.

## Gráficos

Se generan 14 PNG: los 13 gráficos analíticos incluidos tras añadir sensibilidad y una matriz de confusión ponderada adicional:

| Archivo | Contenido |
|---|---|
| profiles.png | Área e instrucción |
| calibration.png | Calibración |
| importance.png | Importancia por permutación |
| subgroups.png | Área y sexo |
| territory.png | Intensidad y magnitud por provincia |
| age.png | Grupos de edad |
| associations.png | V de Cramér |
| roc_pr.png | Curvas ROC y precisión–sensibilidad |
| confusion.png | Matrices sin ponderar |
| thresholds.png | Umbrales en validación |
| paired_uncertainty.png | Intervalos de diferencias entre modelos |
| denominators.png | Comparación de denominadores y ponderación |
| sensibilidad_predictores.png | Sensibilidad por conjuntos de predictores |
| confusion_weighted.png | Matrices ponderadas adicionales |

La sensibilidad se dibuja dentro de `sensibilidad_predictores`; las demás figuras se generan mediante `generar_figuras` y `figura_confusion_ponderada`. El archivo histórico `docs/figures.json` generado por el script enumera únicamente las 12 figuras originales: no debe usarse como índice final de la tesis ni como inventario completo de PNG.

## Límites de interpretación

Validación interna de un solo año. La clave de vivienda representa posiciones del panel, no identidad longitudinal confirmada. El bootstrap mantiene fijos los modelos y no incorpora el diseño muestral completo del INEC. La calibración es diagnóstica; no se aplicó recalibración. Los umbrales exploratorios no representan una política institucional aprobada. Los odds ratios proceden de una logística regularizada, sin valores p ni intervalos de diseño; varias etiquetas utilizan catálogos INEC de años anteriores, circunstancia consignada en la tabla. Las tablas completas contienen categorías escasas que fueron excluidas de la selección editorial de odds ratios de la tesis; no deben interpretarse como asociaciones estables.

## Subir a GitHub

Descomprimir **solo el paquete exterior de esta entrega** y subir el contenido de la carpeta del proyecto, conservando su estructura, incluidos los dos ZIP de `data/raw/`. Los ZIP del INEC deben permanecer comprimidos: el script lee directamente el de microdatos.

El `.gitignore` permite expresamente los dos ZIP originales y excluye datos procesados, predicciones individuales y modelos generados. Si se utiliza una carpeta de salida distinta a `resultados_enemdu`, añadirla también a `.gitignore`.


No se asigna una licencia de redistribución nueva al código en esta entrega. El autor puede añadir la licencia que decida; los datos del INEC y los documentos de terceros conservan sus propias condiciones.
