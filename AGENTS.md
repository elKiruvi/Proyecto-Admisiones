# AGENTS.md

## 1. Contexto del proyecto

Este repositorio pertenece al curso **Ciencia de Datos en Producción**.

El objetivo del proyecto es desarrollar una prueba de concepto (POC) de ciencia de datos siguiendo un proceso reproducible y controlado de:

1. Obtención de datos.
2. Exploración inicial.
3. Análisis exploratorio de datos (EDA).
4. Feature Engineering.
5. Modelo Baseline.
6. Selección del mejor modelo.
7. Interpretación del modelo.
8. Demo funcional.

La primera etapa del proyecto es una POC manual basada principalmente en notebooks. No adelantar innecesariamente componentes de producción que pertenecen a etapas posteriores del curso.

---

## 2. Estructura del repositorio

La estructura actual del proyecto es:

```text
data/
models/
notebooks/
    1-data/
    2-exploration/
    3-analysis/
    4-feat_eng/
    5-models/
    6-interpretation/
    7-deploy/
    8-reports/
scripts/
src/
tests/
```

### Propósito de cada directorio

* `data/`: datos del proyecto. Separar datos RAW, intermedios y procesados cuando corresponda.
* `models/`: modelos y artefactos serializados.
* `notebooks/`: experimentación, análisis y desarrollo de la POC.
* `scripts/`: scripts auxiliares y automatizaciones.
* `src/`: código Python reutilizable y código que posteriormente pueda pasar a producción.
* `tests/`: pruebas automatizadas.

No cambiar esta estructura sin una razón técnica clara.

---

# 3. Entorno de desarrollo

## Python

* Utilizar la versión de Python definida por el proyecto.
* No crear entornos virtuales adicionales.
* El entorno virtual oficial del proyecto es `.venv`.

## Dependencias

Este proyecto utiliza `uv` para gestionar dependencias.

Preferir:

```bash
uv run <comando>
```

y:

```bash
uv sync
```

cuando corresponda.

No instalar paquetes globalmente.

No utilizar `pip install` directamente si la dependencia debe quedar registrada en el proyecto.

No agregar una dependencia nueva sin justificar primero por qué es necesaria y verificar si la funcionalidad puede resolverse con dependencias existentes.

---

# 4. Regla fundamental: analizar antes de modificar

Antes de modificar código o notebooks:

1. Inspeccionar los archivos relevantes.
2. Comprender la estructura existente.
3. Identificar dependencias y relaciones.
4. Revisar los cambios existentes en Git.
5. Determinar el cambio mínimo necesario.

Para cambios pequeños se puede proceder directamente después del análisis.

Para cambios importantes o que afecten múltiples archivos:

1. Presentar primero un plan.
2. Identificar archivos que serán modificados.
3. Explicar riesgos o posibles efectos secundarios.
4. Esperar aprobación humana antes de implementar.

No modificar archivos innecesariamente.

No reescribir archivos completos cuando sea suficiente realizar un cambio localizado.

---

# 5. Uso de referencias `@`

Cuando sea posible, utilizar referencias explícitas a archivos o directorios relevantes.

Ejemplo:

```text
@notebooks/3-analysis/
@src/
@tests/
```

Preferir trabajar sobre archivos específicos cuando el contexto sea conocido:

```text
@notebooks/3-analysis/eda.ipynb
```

Evitar pedir o ejecutar cambios genéricos sobre todo el repositorio cuando solo se necesita modificar una parte.

---

# 6. Reglas para notebooks

Los notebooks son artefactos principales de la POC.

Los notebooks deben utilizarse para:

* exploración;
* análisis;
* experimentación;
* visualización;
* comparación de modelos;
* documentación de resultados;
* conclusiones de cada etapa.

### Reglas

* No eliminar análisis existente sin justificación.
* No borrar resultados, gráficos o conclusiones simplemente para "limpiar" el notebook.
* No modificar un notebook existente sin comprender primero su propósito.
* Mantener una secuencia lógica y reproducible de ejecución.
* Evitar código innecesariamente duplicado.
* Documentar decisiones importantes mediante Markdown.
* Mantener separadas las celdas de explicación, procesamiento y resultados.
* No introducir arquitectura de producción innecesaria dentro de los notebooks.
* Cuando una transformación o función deba reutilizarse posteriormente, considerar moverla a `src/`.

Los notebooks deben permitir entender qué se hizo, por qué se hizo y cuáles fueron los resultados.

---

# 7. Reglas de datos

Los datos originales deben preservarse.

Nunca modificar destructivamente el dataset RAW.

Preferir un flujo:

```text
RAW
 ↓
Exploración / limpieza
 ↓
Datos procesados
 ↓
Feature Engineering
 ↓
Modelamiento
```

No sobrescribir datos originales para ahorrar espacio.

Antes de eliminar registros:

* identificar la razón;
* evaluar el impacto;
* documentar la decisión;
* verificar que la eliminación esté justificada por el problema.

---

# 8. Outliers

Los outliers **no deben eliminarse automáticamente**.

La presencia de un outlier no implica que deba ser eliminado.

Antes de modificar o eliminar outliers:

1. Detectarlos.
2. Analizar su naturaleza.
3. Determinar si representan errores, ruido o casos válidos.
4. Evaluar su impacto sobre el problema.
5. Documentar la decisión.

Si no existe justificación suficiente, conservarlos.

---

# 9. Data Leakage

Evitar cualquier forma de data leakage.

Las transformaciones que aprendan parámetros de los datos deben ajustarse únicamente sobre los datos de entrenamiento.

Cuando corresponda:

```text
Train
 ↓
fit preprocessing
 ↓
transform Train/Test
```

No utilizar información del conjunto de test para tomar decisiones de entrenamiento, selección de variables o ajuste de hiperparámetros.

Preferir `Pipeline` y `ColumnTransformer` de scikit-learn para mantener el procesamiento reproducible.

---

# 10. Feature Engineering

Las transformaciones deben estar justificadas por el análisis exploratorio.

No crear features solamente por aumentar la cantidad de variables.

Cuando corresponda utilizar:

* imputación;
* escalado;
* encoding;
* transformaciones numéricas;
* selección de variables;
* creación de atributos derivados.

Las transformaciones destinadas al modelamiento deben implementarse mediante pipelines de scikit-learn.

---

# 11. Modelamiento

Antes de seleccionar modelos:

1. Definir claramente el tipo de problema.
2. Definir la variable objetivo.
3. Definir las métricas.
4. Establecer un baseline.
5. Separar Train/Test.
6. Utilizar validación cruzada cuando corresponda.

El modelo final debe demostrar una mejora razonable respecto al baseline.

No seleccionar un modelo únicamente porque tenga el mayor score en una única partición de datos.

Considerar:

* media de validación cruzada;
* desviación estándar;
* desempeño en test;
* estabilidad;
* complejidad;
* tiempo de entrenamiento;
* interpretabilidad;
* riesgo de overfitting.

---

# 12. Baseline

Siempre establecer primero una referencia sencilla.

Puede utilizarse:

* una heurística;
* un modelo dummy;
* otra solución simple y justificable.

El objetivo es responder:

> ¿El modelo de Machine Learning realmente mejora una solución sencilla?

No omitir el baseline.

---

# 13. Evaluación

Las métricas deben estar alineadas con el objetivo del problema.

Para regresión, considerar cuando corresponda:

* MAE;
* RMSE;
* R².

No asumir automáticamente cuál es la métrica principal.

La métrica principal debe estar justificada en el notebook.

Cuando se utilice validación cruzada, reportar al menos:

* media;
* desviación estándar.

No seleccionar un modelo basándose únicamente en una métrica sin analizar el contexto.

---

# 14. Model Selection

Cuando la selección sea manual:

* evaluar múltiples tipos de modelos;
* utilizar el mismo pipeline de preprocessing;
* comparar mediante validación cruzada;
* descartar modelos claramente inferiores;
* optimizar hiperparámetros de los mejores candidatos;
* evaluar nuevamente sobre test;
* analizar overfitting y underfitting.

No realizar tuning sobre el conjunto de test.

El conjunto de test debe conservarse para la evaluación final.

---

# 15. Interpretación

La interpretación debe analizar:

* variables importantes;
* comportamiento del modelo;
* errores;
* learning curves;
* escalabilidad;
* overfitting;
* underfitting;
* posibles causas de los errores.

Cuando sea apropiado investigar si los errores están relacionados con:

* outliers;
* calidad de datos;
* variables;
* encoding;
* transformaciones;
* distribución de los datos;
* insuficiencia del modelo.

No afirmar causalidad sin evidencia.

---

# 16. Calidad de código

Todo código Python reutilizable debe:

* tener nombres descriptivos;
* utilizar type hints cuando corresponda;
* tener funciones pequeñas y reutilizables;
* incluir docstrings apropiados;
* evitar duplicación innecesaria;
* seguir las convenciones del proyecto.

Utilizar las herramientas de calidad configuradas en el repositorio.

Antes de considerar una tarea terminada, ejecutar cuando corresponda:

```bash
uv run pytest
uv run ruff check .
uv run mypy .
uv run pre-commit run --all-files
```

Si alguno de estos comandos falla, investigar y corregir el problema antes de declarar la tarea terminada.

No desactivar reglas de linting o testing únicamente para hacer que CI pase.

---

# 17. Testing

Cada nueva funcionalidad reutilizable debe considerar pruebas.

Las pruebas deben ubicarse en:

```text
tests/
```

Utilizar `pytest`.

Los tests deben verificar comportamiento, no simplemente aumentar artificialmente el porcentaje de cobertura.

Cuando se corrija un bug importante, considerar agregar un test que evite su regresión.

---

# 18. Git y Gitflow

Todo nuevo trabajo debe realizarse en una rama independiente.

Seguir el modelo de branching definido por el curso.

Flujo esperado:

```text
Issue
 ↓
Feature branch
 ↓
Implementación
 ↓
Validación
 ↓
Commit
 ↓
Push
 ↓
Pull Request
 ↓
Code Review
 ↓
CI/CD
 ↓
Merge
```

No trabajar directamente sobre `main` para desarrollar funcionalidades.

Los commits deben ser atómicos.

Utilizar Conventional Commits.

Ejemplos:

```text
feat: add initial data download notebook
feat: add exploratory data analysis
feat: add preprocessing pipeline
feat: add baseline model
fix: correct missing value handling
docs: update project documentation
test: add regression model tests
refactor: simplify preprocessing pipeline
```

---

# 19. Pull Requests

Cada Issue debe terminar en un Pull Request cuando corresponda.

Antes de crear el PR:

1. Ejecutar tests.
2. Ejecutar Ruff.
3. Ejecutar Mypy cuando corresponda.
4. Ejecutar pre-commit.
5. Revisar `git diff`.
6. Revisar que no existan archivos innecesarios.
7. Confirmar que el cambio corresponde únicamente al Issue.

El PR debe pasar CI/CD y tener la revisión requerida por el curso.

No mezclar múltiples funcionalidades no relacionadas en un mismo PR.

---

# 20. Git: acciones prohibidas

El agente NO debe ejecutar automáticamente:

```bash
git push
git push --force
git reset --hard
git clean -fd
git checkout -- .
git restore .
```

Tampoco debe eliminar archivos o datos importantes sin autorización explícita.

El usuario debe aprobar las operaciones potencialmente destructivas.

Antes de realizar cambios importantes en Git, utilizar:

```bash
git status
git diff
```

---

# 21. Dependencias y configuración

No modificar automáticamente:

* `pyproject.toml`;
* `uv.lock`;
* `.pre-commit-config.yaml`;
* `.gitignore`;
* configuración de CI/CD;

si el cambio no es necesario para la tarea actual.

Antes de agregar una dependencia:

1. Explicar por qué es necesaria.
2. Verificar si ya existe una solución disponible.
3. Evaluar impacto sobre el entorno.
4. Solicitar aprobación.

No instalar herramientas globalmente para resolver un problema que pertenece al proyecto.

---

# 22. Simplicidad y evitar sobreingeniería

Preferir siempre la solución más sencilla que cumpla los requisitos.

No introducir sin justificación:

* frameworks adicionales;
* patrones de diseño;
* clases innecesarias;
* abstracciones excesivas;
* microservicios;
* bases de datos;
* APIs;
* contenedores;
* sistemas distribuidos.

La arquitectura debe evolucionar conforme lo requieran las siguientes etapas del curso.

No adelantar infraestructura de producción durante la POC si no es necesaria.

---

# 23. Uso de documentación externa

Cuando se trabaje con una librería cuya API pueda haber cambiado, utilizar documentación actualizada cuando esté disponible.

Priorizar documentación oficial.

Para librerías integradas mediante Context7, utilizar Context7 para verificar:

* APIs;
* parámetros;
* ejemplos;
* cambios de versión;
* prácticas recomendadas.

No inventar APIs.

Si existe incertidumbre sobre el comportamiento de una librería, verificarlo antes de implementarlo.

---

# 24. Memoria del proyecto

Utilizar memoria persistente únicamente para decisiones relevantes del proyecto.

Ejemplos de información que puede ser útil recordar:

* decisiones arquitectónicas;
* decisiones sobre métricas;
* decisiones sobre preprocessing;
* decisiones sobre outliers;
* problemas encontrados y sus soluciones;
* decisiones sobre modelos;
* restricciones impuestas por el curso.

No almacenar como memoria permanente información trivial como comandos ejecutados o resultados temporales.

---

# 25. Uso de agentes y herramientas de IA

Las herramientas de IA son asistentes de desarrollo, no sustitutos de la revisión humana.

El agente debe:

```text
Analizar
 ↓
Proponer
 ↓
Implementar
 ↓
Validar
 ↓
Mostrar cambios
```

El usuario mantiene la decisión final sobre:

* arquitectura;
* dependencias;
* cambios destructivos;
* Git;
* commits;
* push;
* merge;
* decisiones metodológicas.

No asumir que una respuesta generada por un modelo es correcta.

---

# 26. Regla para completar tareas

Una tarea no debe considerarse terminada simplemente porque el código fue generado.

Debe cumplir:

```text
Requerimiento
    ↓
Implementación
    ↓
Validación
    ↓
Tests
    ↓
Lint
    ↓
Revisión del diff
    ↓
Documentación
    ↓
Commit / PR
```

Antes de decir que una tarea está terminada, comprobar explícitamente qué requisitos del Issue fueron cumplidos y cuáles no.

Si algún requisito no puede cumplirse, indicarlo claramente.

---

# 27. Prioridad de las instrucciones

En caso de conflicto, seguir este orden:

1. Requisitos explícitos del proyecto y del curso.
2. Código y configuración existentes del repositorio.
3. Este `AGENTS.md`.
4. Instrucciones específicas del Issue o tarea actual.
5. Preferencias generales de desarrollo.

No modificar una decisión previamente establecida sin explicar el motivo.

---

# 28. Principio general

El objetivo no es producir la mayor cantidad de código.

El objetivo es producir una solución:

* correcta;
* reproducible;
* mantenible;
* verificable;
* explicable;
* alineada con los requisitos del curso.

**Analizar primero. Cambiar lo mínimo necesario. Validar siempre.**
