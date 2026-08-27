# AGENTS.md — Proyecto-Admisiones

## 1. Propósito del archivo

Este archivo define las reglas operativas para agentes de IA que trabajen en
`Proyecto-Admisiones`.

El objetivo es mantener el proyecto reproducible, incremental, simple y alineado
con los Issues del curso de Ciencia de Datos en Producción.

### Jerarquía de autoridad

Cuando exista una duda, aplicar este orden:

1. Requerimiento real del Issue que se está implementando.
2. Este `AGENTS.md`.
3. Estructura, README y convenciones existentes del repositorio.
4. Documentación oficial de las herramientas utilizadas.
5. Suposiciones del agente.

**Nunca convertir una suposición en un requisito.**

Si el Issue y una interpretación del agente parecen entrar en conflicto,
detener la implementación y revisar el Issue antes de introducir cambios.

---

# 2. Principios fundamentales

## 2.1 Implementar por Issue

Cada Issue representa una etapa concreta del proyecto.

El agente debe:

- leer el Issue completo antes de implementar;
- identificar entregables y criterios de aceptación;
- determinar qué está explícitamente dentro del alcance;
- determinar qué queda fuera;
- evitar implementar funcionalidades de Issues posteriores;
- no introducir infraestructura por anticipación.

### Regla de oro

> Implementar lo que el Issue necesita, no lo que el agente imagina que el
> proyecto necesitará más adelante.

---

## 2.2 Evitar sobreingeniería

Preferir la solución más sencilla que cumpla el requerimiento.

No agregar:

- servicios externos;
- bases de datos;
- infraestructura;
- frameworks;
- dependencias;
- pipelines;
- sistemas de almacenamiento;
- APIs;
- despliegues;

salvo que el Issue actual los requiera explícitamente.

Una herramienta puede ser técnicamente útil y aun así estar fuera del alcance.

OpenCode no debe modificar `pyproject.toml`, `uv.lock` ni
`.github/workflows/*` en trabajo no relacionado con el Issue.

Evitar scope creep y limpiezas no relacionadas.

---

## 2.3 Diferenciar hechos, supuestos e inferencias

Cuando la información no esté documentada:

- decir `no documentado`;
- marcar cualquier supuesto como supuesto;
- no presentar inferencias como hechos;
- no inventar fuentes, métricas, procesos de actualización o requisitos.

Esto es especialmente importante en los notebooks de entendimiento del problema.

---

# 3. Estructura del proyecto

Respetar la estructura existente.

Áreas principales:

```text
data/
├── 01_raw/
├── 02_intermediate/
├── 03_primary/
├── 04_feature/
├── 05_model_input/
├── 06_models/
├── 07_model_output/
└── 08_reporting/

notebooks/
├── 1-data/
├── 2-exploration/
├── 3-analysis/
├── 4-feat_eng/
├── 5-models/
├── 6-interpretation/
├── 7-deploy/
└── 8-reports/

src/
tests/
scripts/
conf/
models/
```

No asumir que todas las carpetas tienen que utilizarse en cada Issue.

La etapa correspondiente al Issue determina qué carpetas y archivos deben
modificarse.

---

# 4. Datos RAW

## 4.1 Regla de inmutabilidad

`data/01_raw/` es la fuente RAW y debe considerarse inmutable.

Los agentes:

- pueden leer los archivos RAW;
- pueden inspeccionarlos;
- pueden validar su estructura;
- pueden copiarlos cuando el Issue lo requiera;

pero **no deben modificar destructivamente los archivos RAW**.

Las limpiezas y transformaciones deben producir nuevos artefactos en las capas
correspondientes.

## 4.2 Antes de modificar datos

Comprobar:

```bash
git status --short
```

y verificar que el cambio pertenece al Issue actual.

Después de trabajar con datos RAW, cuando corresponda:

```bash
git diff -- data/01_raw/
```

debe permanecer vacío.

## 4.3 No "corregir" silenciosamente el RAW

Si se encuentran:

- valores faltantes;
- duplicados;
- outliers;
- nombres inconsistentes;
- espacios;
- valores sospechosos;
- formatos inesperados;

primero documentar la observación.

No eliminar ni modificar automáticamente esos registros en RAW.

---

# 5. Proyecto de admisiones

El dataset principal actual es:

```text
data/01_raw/Admission_Predict.csv
```

Archivo auxiliar:

```text
data/01_raw/Informacion.txt
```

El target conocido es:

```text
Chance of Admit
```

El problema identificado en el Issue #1 es regresión supervisada.

No asumir que estas decisiones permanecen iguales para futuras etapas sin revisar
el Issue correspondiente y los resultados obtenidos posteriormente.

---

# 6. Notebooks

## 6.1 Principios

Los notebooks sirven para:

- exploración;
- análisis;
- experimentación;
- documentación de etapas del proyecto.

Deben ser:

- reproducibles;
- ejecutables de principio a fin;
- claros;
- enfocados en la etapa correspondiente.

## 6.2 No adelantar etapas

Un notebook de obtención/entendimiento de datos no debe convertirse en un
notebook de modelado.

Un notebook de EDA no debe incorporar automáticamente:

- entrenamiento;
- despliegue;
- tracking;
- producción;

si el Issue no lo solicita.

## 6.3 Rutas

Preferir rutas reproducibles y relativas al repositorio.

Cuando sea necesario localizar el repositorio desde un notebook, usar una
estrategia robusta basada en la estructura real del proyecto, no rutas absolutas
dependientes del equipo del usuario.

## 6.4 Kernel

El proyecto utiliza Python dentro del entorno `.venv`.

Para verificar el intérprete desde un notebook:

```python
import sys
print(sys.executable)
```

Debe apuntar al entorno Python del proyecto.

No asumir que el Python de Windows es el mismo entorno que el Python de WSL.

## 6.5 Dependencias entre notebooks

Los notebooks forman una cadena lógica de dependencias:

```text
01 (datos) → 02 (exploración) → 03 (análisis) → 04 (feature engineering)
→ 05.1/05.2 (modelos) → 06 (interpretación) → demo
```

Si un cambio afecta datos, esquema, split train/test, preprocesamiento,
feature engineering, entradas o salidas del modelo, u otro artefacto
consumido aguas abajo, revisar los notebooks dependientes afectados.

Cuando las entradas o supuestos de un notebook aguas abajo cambiaron:

- re-ejecutar los notebooks dependientes afectados de principio a fin;
- verificar sus outputs después del cambio.

No modificar ni re-ejecutar notebooks aguas abajo solamente porque un notebook
aguas arriba cambió si sus entradas, supuestos y outputs siguen siendo
válidos.

Notas explícitas:

- cambios en `04.1` pueden invalidar `05.x` y `06.1`;
- cambios en `05.2` pueden invalidar el artefacto serializado y la demo.

Evitar churn innecesario en cascada sobre notebooks no afectados.

## 6.6 Notebooks históricos

Los notebooks históricos son evidencia, no deuda técnica.

No reescribir notebooks históricos de EDA/modelado solamente para imponer
prácticas más modernas cuando sus limitaciones ya están documentadas.

Preservar los resultados históricos salvo que el Issue exija explícitamente
un cambio metodológico.

---

# 7. Python y dependencias

## 7.1 Gestor

Usar `uv` para:

- sincronizar el entorno;
- ejecutar Python;
- ejecutar tests;
- ejecutar herramientas del proyecto.

Ejemplos:

```bash
uv sync
uv run pytest
uv run python ...
```

## 7.2 No agregar dependencias innecesarias

Antes de agregar una dependencia:

1. comprobar si ya existe;
2. comprobar si la funcionalidad puede resolverse con las herramientas actuales;
3. confirmar que el Issue realmente necesita la nueva dependencia.

No ejecutar `uv add` como solución automática a cualquier problema.

Los cambios en `pyproject.toml`/`uv.lock` requieren justificación explícita y
deben estar exigidos por el Issue actual. No incluirlos en trabajo no
relacionado.

Si una dependencia nueva es necesaria:

- justificarla;
- modificar `pyproject.toml`;
- actualizar `uv.lock`;
- validar el entorno;
- revisar el diff.

---

# 8. Transformaciones y Machine Learning

Cuando el Issue llegue a etapas de modelado:

## 8.1 Data leakage

Evitar que información del conjunto de validación/test influya en el
entrenamiento.

Reglas concretas:

- todo preprocesamiento que aprenda parámetros de los datos debe vivir dentro
  de un `scikit-learn Pipeline`/`ColumnTransformer`;
- ese preprocesamiento debe ajustarse de forma independiente dentro de cada
  fold de cross-validation;
- nunca ajustar el preprocesamiento sobre el conjunto de entrenamiento
  completo antes de la cross-validation;
- el conjunto de test permanece aislado hasta la evaluación final;
- el split train/test se define una única vez en la capa de feature
  engineering y se consume aguas abajo; no re-dividir repetidamente.

## 8.2 Pipelines

Para flujos de Machine Learning, preferir `scikit-learn Pipeline` y
`ColumnTransformer` cuando sean apropiados.

Reglas concretas:

- usar `ColumnTransformer` para tipos de datos mixtos;
- usar un único `Pipeline` completo desde el entrenamiento hasta la
  inferencia;
- el artefacto serializado debe contener preprocesamiento más modelo;
- no ajustar ni transformar datos manualmente fuera del `Pipeline` cuando esa
  transformación forma parte del flujo del modelo.

Esto ayuda a mantener:

- reproducibilidad;
- separación de transformaciones;
- prevención de leakage;
- consistencia entre entrenamiento e inferencia.

## 8.3 Baseline

Antes de afirmar que un modelo funciona bien, establecer un baseline
razonable.

Comparar modelos contra ese baseline usando métricas previamente justificadas.

## 8.4 Outliers

No eliminar outliers automáticamente.

Primero:

1. detectarlos;
2. entender su significado;
3. evaluar su impacto;
4. justificar cualquier tratamiento.

## 8.5 Valores faltantes

No imputar automáticamente.

La estrategia debe pertenecer al Issue/etapa correspondiente y estar
justificada.

## 8.6 Inmutabilidad del artefacto

El artefacto:

```text
models/05_model_selection_pipeline.joblib
```

es producido por el notebook `05.2` y consumido por el código de
inferencia/demo.

No:

- modificar manualmente el artefacto;
- regenerar ni reentrenar el artefacto fuera del flujo que lo produce.

El reentrenamiento o un cambio de modelo exige un Issue explícito.

---

# 9. Evaluación

Las métricas deben derivarse del objetivo del problema.

No elegir una métrica solamente porque sea común.

Documentar:

- qué mide;
- por qué es apropiada;
- cómo se interpreta;
- cuál es el baseline;
- qué desempeño sería útil.

Para el problema de `Chance of Admit`, Issue #1 estableció como primera
intuición histórica:

- RMSE principal;
- MAE complementaria;
- R² como métrica complementaria potencial.

Baseline inicial documentado (intuición histórica del Issue #1, valores sin
modificar):

```text
predecir siempre la media del target
media ≈ 0.733034
RMSE ≈ 0.142125
```

Estos valores son una referencia inicial del Issue #1. El Issue #6 estableció
la metodología definitiva de selección de modelos (cross-validation anidada)
y produjo el artefacto final serializado. El esquema de evaluación debe
tomarse de los resultados del Issue #6, no de la intuición inicial.

---

# 10. Git y Gitflow

## 10.1 Nunca trabajar directamente sobre main

Las tareas deben desarrollarse en ramas asociadas a Issues.

Flujo:

```text
Issue
  ↓
rama desde la base correcta
  ↓
implementación
  ↓
validación
  ↓
commit
  ↓
push
  ↓
Pull Request
  ↓
review
  ↓
CI/CD
  ↓
merge
```

## 10.2 Base de la rama

Antes de crear una rama:

```bash
git fetch origin
git status
```

Confirmar cuál es la rama base correcta.

Preferir crear la rama desde el `origin/main` actualizado cuando el Issue
corresponda a una nueva entrega sobre `main`.

## 10.3 PRs limpios

Un PR debe contener únicamente los cambios relacionados con su Issue.

No incluir accidentalmente:

- configuración de agentes;
- cambios de otro Issue;
- cambios personales;
- dependencias no relacionadas;
- archivos generados;
- caches;
- notebooks antiguos;
- archivos temporales.

## 10.4 No usar `git add -A` indiscriminadamente

Antes de staging:

```bash
git status --short
```

Agregar explícitamente los archivos relacionados con la tarea.

Después:

```bash
git diff --cached --check
git diff --cached --stat
```

## 10.5 Commits

Usar Conventional Commits.

Ejemplos:

```text
feat: add raw admissions data and problem understanding notebook
fix: correct missing value handling
docs: update project documentation
test: add validation for data loader
chore: update development tooling
```

El mensaje debe describir el cambio, no el proceso interno del agente.

## 10.6 Convenciones de rama y trazabilidad

Preferir nombres de rama siguiendo la convención observada del proyecto:

```text
feature/issue-N-*
fix/issue-N-*
```

Un Issue por PR.

Cada commit debe representar una unidad de trabajo revisable.

Preservar la trazabilidad:

```text
Issue → rama → commit → PR → merge
```

---

# 11. Operaciones Git peligrosas

No ejecutar sin autorización explícita y sin comprobar primero el estado:

```text
git reset --hard
git clean
git restore
git checkout -- <file>
git push --force
git branch -D
git stash drop
git stash clear
```

Especialmente:

- no borrar ramas antes de verificar que su trabajo está preservado;
- no eliminar stashes sin inspeccionarlos;
- no hacer force push;
- no restaurar archivos que puedan contener trabajo del usuario.

Si OpenCode bloquea una operación Git, **no intentar evadir el bloqueo**.
Indicar al usuario el comando que debe ejecutar manualmente.

---

# 12. Archivos generados y entorno local

No versionar:

- caches;
- credenciales;
- API keys;
- `.env`;
- entornos virtuales;
- archivos temporales;
- artefactos locales de agentes;
- archivos generados que no sean entregables.

El directorio local de Gentle AI:

```text
.atl/
```

es un artefacto del entorno y no pertenece a los entregables del proyecto.
Ya está cubierto por `.gitignore` y debe permanecer fuera de los commits.
No debe agregarse al staging bajo ninguna circunstancia.

---

# 13. Secretos

Nunca escribir en:

- notebooks;
- código;
- `pyproject.toml`;
- `README.md`;
- commits;
- PRs;

secretos como:

- API keys;
- passwords;
- tokens;
- credenciales.

Si una integración futura requiere secretos:

- utilizar variables de entorno;
- documentar únicamente nombres de variables;
- usar placeholders;
- mantener secretos fuera de Git.

Pero una integración de este tipo debe existir únicamente si el Issue la exige.

---

# 14. Hopsworks y Feature Stores

**No existe una obligación general de usar Hopsworks en este proyecto.**

En particular:

> Hopsworks fue eliminado deliberadamente del Issue #1 porque el Issue real
> únicamente exigía obtención de datos RAW + entendimiento del problema.

No introducir:

```text
Hopsworks
Feature Store
Feature Group
hsfs
```

en una tarea solamente por recordar que fueron considerados anteriormente.

Si un Issue futuro exige explícitamente una de estas tecnologías:

1. leer el Issue;
2. comprobar la documentación actual;
3. verificar compatibilidad con el entorno;
4. evaluar si la dependencia es necesaria;
5. implementarla únicamente dentro del alcance de ese Issue.

---

# 15. Calidad de código

Mantener código:

- legible;
- pequeño;
- determinista;
- testeable;
- documentado cuando la lógica no sea obvia.

Evitar:

- funciones gigantes;
- duplicación;
- lógica innecesaria;
- código muerto;
- comentarios que describan obviedades;
- abstracciones prematuras.

---

# 16. Validaciones

Antes de considerar terminada una tarea, ejecutar las validaciones aplicables.

Base:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Validación con cobertura (paridad con CI):

```bash
uv run pytest --cov --cov-branch --cov-fail-under=60
```

Ruff y mypy no son dependencias directas del proyecto: se ejecutan a través
de pre-commit, usando la configuración en `.code_quality/`.

Para notebooks:

- ejecutar de principio a fin;
- comprobar que no haya errores;
- revisar outputs relevantes;
- comprobar que no haya cambios accidentales en RAW.

No declarar una tarea completa solo porque el notebook "corre".

Debe cumplir el Issue.

---

# 17. Revisión del diff

Antes del commit:

```bash
git status --short
git diff --check
git diff --stat
git diff
```

Si hay archivos inesperados:

**detenerse y analizarlos.**

Después del staging:

```bash
git diff --cached --check
git diff --cached --stat
```

El diff debe responder una pregunta:

> "¿Todos estos cambios son necesarios para este Issue?"

Si la respuesta es no, sacar esos cambios del staging.

---

# 18. CI/CD y Pull Requests

Los PRs son parte del entregable.

Antes del merge:

- CI debe estar verde;
- los checks requeridos deben pasar;
- debe existir la revisión requerida por el curso;
- el PR debe estar limitado al Issue;
- no debe contener secretos;
- no debe incluir cambios accidentales.

No hacer merge antes de cumplir los criterios del Issue.

---

# 19. Documentación

Actualizar documentación solo cuando el Issue lo requiera o cuando sea una
tarea explícita de documentación.

No modificar README/AGENTS por conveniencia durante un Issue de datos si no es
necesario.

Las decisiones importantes deben quedar documentadas en el lugar adecuado.

Mantener `CHANGELOG.md` con la estructura Keep a Changelog ya existente.

El trabajo significativo de un Issue debe quedar representado bajo
`[Unreleased]` en las secciones `Added`/`Changed`/`Fixed` correspondientes.

---

# 20. Engram

Engram es la memoria persistente del proyecto.

Usarlo para guardar:

- decisiones de arquitectura;
- decisiones de alcance;
- restricciones;
- aprendizajes;
- estado relevante para futuras sesiones.

No guardar cada comando ejecutado ni detalles triviales.

Decisión importante ya establecida:

```text
Issue #1 = RAW + entendimiento del problema
Hopsworks = fuera de alcance de Issue #1
PR #15 = mergeado
```

Cuando una decisión cambie, actualizar la memoria anterior en vez de crear
duplicados innecesarios.

---

# 21. Context7

Context7 se utiliza para consultar documentación externa actualizada.

No usar Context7 como memoria del proyecto.

Usarlo cuando una tarea requiera verificar:

- APIs;
- comportamiento de librerías;
- versiones;
- documentación oficial;
- configuración de herramientas.

Preferir documentación oficial cuando sea posible.

---

# 22. OpenCode / agentes

El proyecto utiliza OpenCode y herramientas de agentes.

Los agentes deben trabajar bajo el alcance del Issue y este archivo.

Si existen permisos que bloquean operaciones destructivas o Git:

- respetarlos;
- no intentar evadirlos;
- pedir al usuario que ejecute manualmente la operación si es necesario.

No modificar configuración de agentes como parte de un Issue de datos salvo que
sea explícitamente necesario.

---

# 23. Modos de trabajo recomendados

## Gentle Orchestrator

Usar para:

- recuperar contexto;
- coordinar agentes;
- consultar memoria;
- decidir workflow.

## Plan

Usar cuando:

- el Issue todavía no está completamente entendido;
- existen decisiones de arquitectura;
- hay alternativas relevantes;
- se necesita definir alcance.

## Build

Usar cuando:

- el alcance ya fue decidido;
- la implementación está clara;
- se deben modificar archivos;
- se deben ejecutar validaciones.

### Regla

```text
¿No sabemos qué hacer?
→ Plan

¿Ya sabemos qué hacer?
→ Build
```

---

# 24. Economía de tokens y agentes

Los prompts deben ser:

- cortos;
- específicos;
- accionables.

No repetir contexto que ya está disponible mediante:

- `AGENTS.md`;
- Engram;
- checkpoint del proyecto;
- archivos del repositorio.

No usar modelos/agentes costosos para tareas triviales.

Para una modificación pequeña:

```text
archivo
+
cambio exacto
+
validación
```

es preferible a un prompt enorme.

---

# 25. Continuidad entre sesiones

Cuando una nueva sesión empiece:

1. leer el checkpoint del proyecto si existe;
2. revisar `AGENTS.md`;
3. comprobar Git;
4. revisar el Issue actual;
5. no asumir que el estado de una rama antigua sigue siendo válido.

Comandos iniciales recomendados:

```bash
git status
git branch -vv
git log --oneline -5
```

Si se va a trabajar sobre `main`:

```bash
git fetch origin
git switch main
git pull origin main
```

---

# 26. Issue #1 — referencia histórica

Issues #1–#8 completados mediante PRs mergeados; ver historial de Git.

---

# 27. Stash pendiente

Existe un stash previo:

```text
stash@{0}
```

con cambios anteriores de `pyproject.toml`/`uv.lock` no relacionados con el
Issue #1.

**No eliminarlo sin inspección.**

---

# 28. Checklist antes de terminar cualquier Issue

### Alcance

- [ ] Leí el Issue completo.
- [ ] Identifiqué entregables.
- [ ] Identifiqué criterios de aceptación.
- [ ] No implementé funcionalidades fuera del alcance.
- [ ] Separé hechos de supuestos.

### Datos

- [ ] RAW permanece intacto.
- [ ] No eliminé datos sin justificación.
- [ ] No introduje leakage.
- [ ] Las transformaciones están en la capa correcta.

### Código

- [ ] No agregué dependencias innecesarias.
- [ ] El código es reproducible.
- [ ] Los tests aplicables pasan.
- [ ] El notebook ejecuta correctamente.

### Git

- [ ] Estoy en la rama correcta.
- [ ] Revisé `git status`.
- [ ] Revisé el diff.
- [ ] No incluí cambios de otros Issues.
- [ ] No incluí archivos locales/caches.
- [ ] El commit sigue Conventional Commits.

### PR

- [ ] Push correcto.
- [ ] PR hacia la rama correcta.
- [ ] Archivos del PR corresponden al Issue.
- [ ] CI verde.
- [ ] Review requerida completada.
- [ ] Merge solo después de cumplir los criterios.

---

# 29. Regla final

Antes de cada cambio importante, responder mentalmente:

1. **¿Qué exige exactamente el Issue?**
2. **¿Este cambio es necesario para cumplirlo?**
3. **¿Estoy modificando algo que pertenece a otra etapa?**
4. **¿Estoy tocando RAW?**
5. **¿Estoy agregando complejidad innecesaria?**
6. **¿Este cambio terminará correctamente aislado en el PR?**

Si alguna respuesta es incierta:

**detenerse, revisar documentación/Issue y preguntar antes de modificar.**
