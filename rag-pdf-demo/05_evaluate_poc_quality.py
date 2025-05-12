# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from databricks import agents

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC # Load your evaluation set from the previous step

# COMMAND ----------

df = spark.table(EVALUATION_SET_FQN)
eval_df = df.toPandas()
display(eval_df)

# COMMAND ----------

# If you did not collect feedback from your stakeholders, and want to evaluate using a manually curated set of questions, you can use the structure below.

eval_data = [
    {
        ### REQUIRED
        # Question that is asked by the user
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",

        ### OPTIONAL
        # Optional, user specified to identify each row
        "request_id": "your-request-id",
        # Optional: correct response to the question
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_response": "There's no significant difference.",
        # Optional: Which documents should be retrieved.
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/Lineamientos+y+Controles+de+las+Zonas+EDV+Cloud.pdf",
            },
        ],
    }
]

# Uncomment this row to use the above data instead of your evaluation set
#eval_df = pd.DataFrame(eval_data)

# COMMAND ----------

eval_data = [
    {
        "request": "Como realizar una solicitud de creación de un EDV?",
        "request_id": "001",
        "expected_response": "Pasos para seguir por el Power User responsable de la zona EDV 1. PASO 1: Presente su necesidad a su Data Steward explicando cómo le puede ayudar la creación de una zona EDV. Puede hacerlo vía correo o agendando una sesión. En lo posible tenga a la mano datos como:✓ ¿Qué quiere lograr/ probar con su zona EDV? ✓ ¿Qué datos serán parte de su exploración? ✓ ¿Cuánto espacio de almacenamiento necesitará? ✓ ¿Cuánto tiempo durará su ejercicio de exploración? ✓ ¿Cuántos usuarios de su equipo ingresarán a la zona EDV? ✓ ¿Cumple con los lineamientos? ✓ ¿Se realizará validación interna de modelos en la zona EDV?2. PASO 2: Habiendo relevado y evaluado su necesidad, el Data Steward validará diversos aspectos de la misma con los equipos de Arquitectura de Datos y su Analista de Seguridad.3. PASO 3: Deberá completar y entregar a su Data Steward: ✓ Lista de Responsables de la Zona EDV. ✓ Formato de solicitud de zona EDV (para Tipo I o Tipo II según corresponda)✓ Conforme de cada integrante de su equipo que accederá a la zona EDV sobre la aceptación a los lineamientos y norma EDV (lineamientos). Cada integrante debe enviarle un correo con el siguiente contenido: <Acepto haber leído las condiciones estipuladas en la norma 4453.011.01 (Marco de Trabajo sobre la Zona de exploración del Datalake EDV). Con ello, se seguirá con los lineamientos indicados en la norma para los trabajos que se realizarán sobre la Zona EDV.> Deberá recopilar todos los conformes y enviarlos adjuntos en 1 solo correo junto con su conforme sobre el mismo enunciado a su Data StewardRecursos de ayuda para complementar los entregables:✓ Responsables Zona EDV (Template).xlsx ✓ Usuarios EDV Tipo I (Template).xlsx ✓ Usuarios EDV Tipo II (Template).xlsxDatos elaborados por BCP para uso Interno4. PASO 4: Una vez que su requerimiento haya sido validado y haya entregado todo lo listado en el punto 3 se procederá a crear la zona EDV. Se le hará llegar el grupo de red necesario para que usted y su equipo puedan ingresar.5. PASO 5: Una vez creado la zona EDV, el Data Steward debe registrar el grupo de red y el autorizador con colibra.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/3-Revisión Solicitud de creación EDV_Q1_2025.pdf"
            }
        ]
    },
    {
        "request": "Como solicitar acceso a una zona EDV?",
        "request_id": "002",
        "expected_response": "El procedimiento de solicitud de acceso dependerá del tipo de Zona EDV que aperturaron. A continuación una pequeña descripción de los tipos de Zonas EDVs.Zona EDV Tipo I ✓ El acceso a una Zona EDV tipo I es permanente, a través de matriz de roles. Solicite a su Gestor de Matriz de Roles que registre y solicite el acceso de su rol al Grupo de Red correspondiente a la zona EDV. Dado que se trata de un acceso restringido indique a su Gestor de Matriz de Roles que debe ser registrado <Con conformidad de jefatura>. ✓ En este requerimiento necesitará del Grupo de red de la zona EDV, la cual será provista por su Data Steward al Power User responsable de la zona EDV.Zona EDV Tipo II ✓ El acceso a una Zona EDV tipo II debe ser solicitado apenas esta haya sido creada y de preferencia con 1 solo ticket masivo. ✓ En este requerimiento necesitará de la siguiente información, la cual será provista por su Data Steward al Power User responsable de la zona EDV: · Grupo de red de la zona EDV · Autorizador del Grupo de Red · Tiempo permitido para la zona EDV Procedimiento para solicitud de grupo de red PASO 1: Ingrese a la web de Help Desk, opción Accesos. PASO 2: Marcar la opción de <más de un usuario afectado>. PASO 3: Ingrese sus datos: Ubicación, matrícula, piso, anexo. PASO 4: Tipo de Solicitud: BRINDAR ACCESOS. 5. Clic <Agregar a la lista>.PASO 5: En la ventana emergente seleccionar como Aplicación o Servicio <LHCL LAKEHOUSE>. PASO 6: Pulsar el botón <Sí, deseo continuar> PASO 7: Seleccione el acceso: ✓ Zona EDV Tipo II: <Temporal> e ingrese el rango de fechas que corresponda. ✓ Zona EDV Tipo I: <Permanente>. Previamente debe estar registrado el grupo de red en su matriz de roles PASO 8: Ingrese el perfil de acceso <EDV>. PASO 9: Seleccione el ambiente <Producción>. PASO 10: Como sustento de acceso seleccione: ✓ Si el acceso es <Temporal>: Proyecto de usuarios finales multi área (prestados). ✓ Si el acceso es <Permanente>: Mapeo Matriz de Roles PASO 11: Adjunte la conformidad del Autorizador del Grupo de Red. PASO 12: En el Detalle especifique: <Por favor su apoyo para el acceso a la zona EDV del equipo de <nombre de su equipo>, grupo de red: <GRUPO DE RED>. Muchas gracias> PASO 13: Pulse <Aceptar>.PASO 14: En la nueva ventana pulse <Detalle> en la sección de usuarios afectados. Ingrese la información de los usuarios.PASO 15: En Detalle general de la solicitud vuelva a ingresar: <Por favor su apoyo para el acceso a la zona EDV del equipo de <nombre de su equipo>, grupo de red: <GRUPO DE RED>. Muchas gracias> PASO 16: Pulse <Aceptar>",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/4-Revisión Solicitud de acceso EDV_Q1_2025.pdf"
            }
        ]
    },
    {
        "request": "cuales son los controles normativos de los EDV?",
        "request_id": "003",
        "expected_response": "Control normativo 01: Antigüedad de vida de la zona tipo II no mayor a 6 meses.• Descripción: La Zona Tipo II no debe superar 6 meses de antigüedad.• Acción correctiva: La Zona pasará cuarentena x 1 mes para luego ser eliminada.Control normativo 02: Todos los usuarios <USER> sean de la misma Área que el responsable de la zona. Todos los usuarios <UNDVALINT> serán del área de validación interna. • Descripción: o Todos los usuarios con perfil <USER> deben pertenecer al mismo equipo del responsable de la zona.o Todos los usuarios con perfil <VIEWER> pueden o no pertenecer al mismo equipo del responsable de la zona, pero solo para actividades acotadas a validación de la exploración realizada por parte del equipo responsable (USER).o Todos los usuarios con perfil <UNDVALINT> deben pertenecer al equipo de validación interna, y solo se utilizara este perfil para actividades acotadas a validación de modelos realizada por parte del equipo responsable (USER).• Acción Correctiva: Se quitará acceso al usuario que no pertenezca al Área o en su debido caso no pertenezcan al equipo de validación Interna.Control normativo 03: Antigüedad de vida de los objetos no mayor a 6 meses.• Descripción: Los Objetos (archivos y tablas creadas) no deben superar 6 meses de antigüedad.• Acción correctiva: Los objetos serán eliminados.Control normativo 04: No existan exploraciones automáticas• Descripción: Los usuarios no deberían de schedular algún script o workflows.• Acción correctiva: Los workflow deben ser eliminados o retirar el acceso al usuario que tenga workflowControl normativo 05: No exista descarga• Descripción: Los usuarios no descarguen objetos en databricks. • Acción correctiva: Quitar acceso al usuario que tenga acceso a descarga o el usuario debe de retirar el acceso al grupo de red de descargaControl normativo 06: No uso de DAC en la Zona.• Descripción: No está permitido el utilizar DAC en espacios exploratorios.• Acción correctiva: La tabla con DAC será eliminada.Control normativo 07: • Nombre: No existan procesos productivos para la toma de decisión• Descripción: Los usuarios no deben de crear un proceso productivos en la capa exploratoria.• Acción correctiva: Los objetos asociados a los procesos se eliminarán",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/Lineamientos+y+Controles+de+las+Zonas+EDV+Cloud.pdf"
            }
        ]
    },
    {
        "request": "cuales son los lineamientos de los EDV?",
        "request_id": "004",
        "expected_response": "* Los lineamientos a continuación son un resumen de la Norma EDV. Todo usuario de la zona EDV debe haber comprendido, dado su conformidad y debe seguir la norma sobre el uso de EDV.* El incumplimiento de los lineamientos de la norma EDV pueden derivar en sanciones y acciones correctivas sobre la zona EDV.1. Son zonas dedicada exclusivamente a exploración de datos.2. No deben contener DAC.3. No debe alimentar procesos o flujos productivos de negocio.4. No se deben conectar usuarios de aplicación.5. No trasladar datos de las zonas EDV a las zonas productivas RDV, UDV, DDV.6. No debe formar parte de ningún proceso schedulado (programado).7. Solo deben conectarse usuarios del mismo equipo.8. No deben tener archivos o tablas con una antigüedad mayor a 6 meses.9. Las zonas EDV Tipo I son permanentes y están restringidas a actividades continuas y puntuales que no tienen una fecha de fin.10.Las zonas EDV Tipo II tienen un tiempo de vida de 6 meses con una extensión de 6 meses adicionales como máximo.11.Un equipo puede tener varias zonas EDV Tipo II.12.El grupo de red LHCL_DTBR_BCP_EDV_<<UNIDAD>>_UNDVALINT_PROD creado en las zonas EDV Tipo I, sera únicamente utilizado por el equipo de VALIDACIÓN INTERNA, y el acceso otorgado a este grupo bajo ninguna circunstancia podrá ser permanente.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/Lineamientos+y+Controles+de+las+Zonas+EDV+Cloud.pdf"
            }
        ]
    },
    {
        "request": "cuales son los pasos de solicitud de creación de EDV?",
        "request_id": "005",
        "expected_response": "Paso a paso Pasos para seguir por el Power User responsable de la zona EDV 1. PASO 1: Presente su necesidad a su Data Steward explicando cómo le puede ayudar la creación de una zona EDV. Puede hacerlo vía correo o agendando una sesión. En lo posible tenga a la mano datos como:✓ ¿Qué quiere lograr/ probar con su zona EDV? ✓ ¿Qué datos serán parte de su exploración? ✓ ¿Cuánto espacio de almacenamiento necesitará? ✓ ¿Cuánto tiempo durará su ejercicio de exploración? ✓ ¿Cuántos usuarios de su equipo ingresarán a la zona EDV? ✓ ¿Cumple con los lineamientos? ✓ ¿Se realizará validación interna de modelos en la zona EDV?2. PASO 2: Habiendo relevado y evaluado su necesidad, el Data Steward validará diversos aspectos de la misma con los equipos de Arquitectura de Datos y su Analista de Seguridad.3. PASO 3: Deberá completar y entregar a su Data Steward: ✓ Lista de Responsables de la Zona EDV. ✓ Formato de solicitud de zona EDV (para Tipo I o Tipo II según corresponda)✓ Conforme de cada integrante de su equipo que accederá a la zona EDV sobre la aceptación a los lineamientos y norma EDV (lineamientos). Cada integrante debe enviarle un correo con el siguiente contenido: <Acepto haber leído las condiciones estipuladas en la norma 4453.011.01 (Marco de Trabajo sobre la Zona de exploración del Datalake EDV). Con ello, se seguirá con los lineamientos indicados en la norma para los trabajos que se realizarán sobre la Zona EDV.> Deberá recopilar todos los conformes y enviarlos adjuntos en 1 solo correo junto con su conforme sobre el mismo enunciado a su Data Steward Recursos de ayuda para complementar los entregables:✓ Responsables Zona EDV (Template).xlsx ✓ Usuarios EDV Tipo I (Template).xlsx ✓ Usuarios EDV Tipo II (Template).xlsx Datos elaborados por BCP para uso Interno 4. PASO 4: Una vez que su requerimiento haya sido validado y haya entregado todo lo listado en el punto 3 se procederá a crear la zona EDV. Se le hará llegar el grupo de red necesario para que usted y su equipo puedan ingresar. 5. PASO 5: Una vez creado la zona EDV, el Data Steward debe registrar el grupo de red y el autorizador con colibra. ¡NOTA! Como verás para la solicitud de creación del EDV es importante gestionar el recurso de la mano del Data Steward.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/3-Revisión Solicitud de creación EDV_Q1_2025.pdf"
            }
        ]
    },
    {
        "request": "Cuales son los pasos para seguir por el Power User responsable de la zona EDV?",
        "request_id": "006",
        "expected_response": "Paso a paso Pasos para seguir por el Power User responsable de la zona EDV 1. PASO 1: Presente su necesidad a su Data Steward explicando cómo le puede ayudar la creación de una zona EDV. Puede hacerlo vía correo o agendando una sesión. En lo posible tenga a la mano datos como:✓ ¿Qué quiere lograr/ probar con su zona EDV? ✓ ¿Qué datos serán parte de su exploración? ✓ ¿Cuánto espacio de almacenamiento necesitará? ✓ ¿Cuánto tiempo durará su ejercicio de exploración? ✓ ¿Cuántos usuarios de su equipo ingresarán a la zona EDV? ✓ ¿Cumple con los lineamientos? ✓ ¿Se realizará validación interna de modelos en la zona EDV?2. PASO 2: Habiendo relevado y evaluado su necesidad, el Data Steward validará diversos aspectos de la misma con los equipos de Arquitectura de Datos y su Analista de Seguridad.3. PASO 3: Deberá completar y entregar a su Data Steward: ✓ Lista de Responsables de la Zona EDV. ✓ Formato de solicitud de zona EDV (para Tipo I o Tipo II según corresponda)✓ Conforme de cada integrante de su equipo que accederá a la zona EDV sobre la aceptación a los lineamientos y norma EDV (lineamientos). Cada integrante debe enviarle un correo con el siguiente contenido: <Acepto haber leído las condiciones estipuladas en la norma 4453.011.01 (Marco de Trabajo sobre la Zona de exploración del Datalake EDV). Con ello, se seguirá con los lineamientos indicados en la norma para los trabajos que se realizarán sobre la Zona EDV.> Deberá recopilar todos los conformes y enviarlos adjuntos en 1 solo correo junto con su conforme sobre el mismo enunciado a su Data Steward Recursos de ayuda para complementar los entregables:✓ Responsables Zona EDV (Template).xlsx ✓ Usuarios EDV Tipo I (Template).xlsx ✓ Usuarios EDV Tipo II (Template).xlsx Datos elaborados por BCP para uso Interno 4. PASO 4: Una vez que su requerimiento haya sido validado y haya entregado todo lo listado en el punto 3 se procederá a crear la zona EDV. Se le hará llegar el grupo de red necesario para que usted y su equipo puedan ingresar. 5. PASO 5: Una vez creado la zona EDV, el Data Steward debe registrar el grupo de red y el autorizador con colibra. ¡NOTA! Como verás para la solicitud de creación del EDV es importante gestionar el recurso de la mano del Data Steward.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/3-Revisión Solicitud de creación EDV_Q1_2025.pdf"
            }
        ]
    },
    {
        "request": "cuantos controles normativos tienen los EDV y cuales son?",
        "request_id": "007",
        "expected_response": "Son 7 los controles normativos: Control normativo 01: Antigüedad de vida de la zona tipo II no mayor a 6 meses. • Descripción: La Zona Tipo II no debe superar 6 meses de antigüedad. • Acción correctiva: La Zona pasará cuarentena x 1 mes para luego ser eliminada. Control normativo 02: Todos los usuarios <USER> sean de la misma Área que el responsable de la zona. Todos los usuarios <UNDVALINT> serán del área de validación interna. Datos elaborados por BCP para uso Interno • Descripción: o Todos los usuarios con perfil <USER> deben pertenecer al mismo equipo del responsable de la zona. o Todos los usuarios con perfil <VIEWER> pueden o no pertenecer al mismo equipo del responsable de la zona, pero solo para actividades acotadas a validación de la exploración realizada por parte del equipo responsable (USER). o Todos los usuarios con perfil <UNDVALINT> deben pertenecer al equipo de validación interna, y solo se utilizara este perfil para actividades acotadas a validación de modelos realizada por parte del equipo responsable (USER). • Acción Correctiva: Se quitará acceso al usuario que no pertenezca al Área o en su debido caso no pertenezcan al equipo de validación Interna. Control normativo 03: Antigüedad de vida de los objetos no mayor a 6 meses. • Descripción: Los Objetos (archivos y tablas creadas) no deben superar 6 meses de antigüedad. • Acción correctiva: Los objetos serán eliminados. Control normativo 04: No existan exploraciones automáticas • Descripción: Los usuarios no deberían de schedular algún script o workflows. • Acción correctiva: Los workflow deben ser eliminados o retirar el acceso al usuario que tenga workflow Control normativo 05: No exista descarga • Descripción: Los usuarios no descarguen objetos en databricks. • Acción correctiva: Quitar acceso al usuario que tenga acceso a descarga o el usuario debe de retirar el acceso al grupo de red de descarga Control normativo 06: No uso de DAC en la Zona. • Descripción: No está permitido el utilizar DAC en espacios exploratorios. • Acción correctiva: La tabla con DAC será eliminada. Control normativo 07: • Nombre: No existan procesos productivos para la toma de decisión • Descripción: Los usuarios no deben de crear un proceso productivos en la capa exploratoria. • Acción correctiva: Los objetos asociados a los procesos se eliminarán",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/Lineamientos+y+Controles+de+las+Zonas+EDV+Cloud.pdf"
            }
        ]
    },
    {
        "request": "que controles normativos tienen los EDV?",
        "request_id": "008",
        "expected_response": "Control normativo 01: Antigüedad de vida de la zona tipo II no mayor a 6 meses. • Descripción: La Zona Tipo II no debe superar 6 meses de antigüedad. • Acción correctiva: La Zona pasará cuarentena x 1 mes para luego ser eliminada. Control normativo 02: Todos los usuarios <USER> sean de la misma Área que el responsable de la zona. Todos los usuarios <UNDVALINT> serán del área de validación interna. Datos elaborados por BCP para uso Interno • Descripción: o Todos los usuarios con perfil <USER> deben pertenecer al mismo equipo del responsable de la zona. o Todos los usuarios con perfil <VIEWER> pueden o no pertenecer al mismo equipo del responsable de la zona, pero solo para actividades acotadas a validación de la exploración realizada por parte del equipo responsable (USER). o Todos los usuarios con perfil <UNDVALINT> deben pertenecer al equipo de validación interna, y solo se utilizara este perfil para actividades acotadas a validación de modelos realizada por parte del equipo responsable (USER). • Acción Correctiva: Se quitará acceso al usuario que no pertenezca al Área o en su debido caso no pertenezcan al equipo de validación Interna. Control normativo 03: Antigüedad de vida de los objetos no mayor a 6 meses. • Descripción: Los Objetos (archivos y tablas creadas) no deben superar 6 meses de antigüedad. • Acción correctiva: Los objetos serán eliminados. Control normativo 04: No existan exploraciones automáticas • Descripción: Los usuarios no deberían de schedular algún script o workflows. • Acción correctiva: Los workflow deben ser eliminados o retirar el acceso al usuario que tenga workflow Control normativo 05: No exista descarga • Descripción: Los usuarios no descarguen objetos en databricks. • Acción correctiva: Quitar acceso al usuario que tenga acceso a descarga o el usuario debe de retirar el acceso al grupo de red de descarga Control normativo 06: No uso de DAC en la Zona. • Descripción: No está permitido el utilizar DAC en espacios exploratorios. • Acción correctiva: La tabla con DAC será eliminada. Control normativo 07: • Nombre: No existan procesos productivos para la toma de decisión • Descripción: Los usuarios no deben de crear un proceso productivos en la capa exploratoria. • Acción correctiva: Los objetos asociados a los procesos se eliminarán",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/Lineamientos+y+Controles+de+las+Zonas+EDV+Cloud.pdf"
            }
        ]
    },
    {
        "request": "Listar los controles normativos de los EDV",
        "request_id": "009",
        "expected_response": "Control 01: Antigüedad de vida de la zona tipo II no mayor a 6 meses. Control 02: Todos los usuarios <USER> sean de la misma Área que el responsable de la zona. Todos los usuarios <UNDVALINT> serán del área de validación interna. Control 03: Antigüedad de vida de los objetos no mayor a 6 meses. Control 04: No existan exploraciones automáticas. Control 05: No exista descarga. Control 06: No uso de DAC en la Zona. Control 07: No existan procesos productivos para la toma de decisión.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/catalog_demo01/agent_demo_adopcion/volumenrawpdf/Lineamientos+y+Controles+de+las+Zonas+EDV+Cloud.pdf"
            }
        ]
    }
]

#eval_df = pd.DataFrame(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the POC application

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the MLflow run of the POC application 

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], filter_string=f"run_name = '{POC_CHAIN_RUN_NAME}'", output_format="list")

if len(runs) != 1:
    raise ValueError(f"Found {len(runs)} run with name {POC_CHAIN_RUN_NAME}.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the correct Python environment for the POC's app
# MAGIC
# MAGIC TODO: replace this with env_manager=virtualenv once that works

# COMMAND ----------

pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{poc_run.info.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation on the POC app

# COMMAND ----------

with mlflow.start_run(run_id=poc_run.info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{poc_run.info.run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the evaluation results
# MAGIC
# MAGIC You can explore the evaluation results using the above links to the MLflow UI.  If you prefer to use the data directly, see the cells below.

# COMMAND ----------

# Summary metrics across the entire evaluation set
eval_results.metrics

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables['eval_results']

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)
