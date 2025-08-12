import pandas as pd
import numpy as np

import utils as U


def load_raw_excel(path: str) -> pd.DataFrame:
    """Carga el Excel crudo tal como en el notebook."""
    return pd.read_excel(path)


def wrangle_respuestas(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce la lógica de limpieza de limpieza_encuestas.ipynb, incluyendo:
    - drops y columna non_fan
    - relación con el deporte, frecuencia (ord/cat)
    - canales, redes, tipos de contenido (multi → dummies)
    - sigue (equipos/jugadoras), asistencia
    - percepción patrocinio, compra, importancia marcas
    - inversión igual (dummies), actitud marca (dummies)
    - sentimiento campañas, crecimiento, desafíos, motivación
    - apuestas, valores, por qué no ves, conocer ligas, vio contenido
    - percepción general, sabía liga
    """
    df = df_raw.copy()

    # --- Drops y non_fan (Cell 2)
    drops_esp = ['Columna 11', 'Columna 3', '¿Te gustaría participar en la rifa de un PREMIO? (Toma 1 minuto más y tu participación es completamente anónima)',
                 'Ingresa tu código único que incluya:\n- 8 letras (pueden repetirse)\n- 2 números (pueden repetirse)\n- 2 símbolos especiales como: #, @, !, %, &, etc.',
                 '¿Te refirió alguien para contestar esta encuesta? Ingresa su código de referido']
    df = df.drop(columns=[c for c in drops_esp if c in df.columns], errors="ignore")
    col_freq = "¿Con qué frecuencia ves fútbol femenino?"
    if col_freq in df.columns:
        df['non_fan'] = df[col_freq].apply(lambda x: 'No Fan' if x == 'Nunca' else 'Fan')

    # --- Relación con el deporte (Cell 8)
    col_rel = '¿Cuál es tu relación con el deporte? (Elige todas las que apliquen)'
    if col_rel in df.columns:
        df[[f"rel_{c}" for c in U.REL_COLS]] = df[col_rel].apply(U.ohe_relacion)

    # --- Frecuencia ord/cat (Cell 9)
    if col_freq in df.columns:
        df["freq_ord"] = df[col_freq].map(U.FREQ_TO_ORD)
        df["freq_cat"] = pd.Categorical(df[col_freq], categories=U.FREQ_ORDER, ordered=True)

    # --- Canales en vivo (Cell 10)
    col_canales = "¿A través de qué canales sigues los partidos de fútbol femenino en vivo? (Selecciona todos los que apliquen)"
    if col_canales in df.columns:
        canales_feats = df[col_canales].apply(U.featurize_canales)
        df = pd.concat([df, canales_feats], axis=1)

    # --- Redes sociales (Cell 11)
    col_redes = "¿En qué redes sociales sigues contenido de fútbol femenino? (Selecciona todas las que apliquen)"
    if col_redes in df.columns:
        redes_feats = df[col_redes].apply(U.featurize_redes)
        df = pd.concat([df, redes_feats], axis=1)

    # --- Tipos de contenido (Cell 12)
    col_tipo = "¿Qué tipo de contenido consumes más? (Selecciona todas las que apliquen)"
    if col_tipo in df.columns:
        tipos_feats = df[col_tipo].apply(U.featurize_tipos)
        df = pd.concat([df, tipos_feats], axis=1)

    # --- Sigue equipos/jugadoras (Cells 3/4/13)
    col_sigue = "¿Sigues a equipos o jugadoras específicas en el fútbol femenino?"
    if col_sigue in df.columns:
        sigue_feats = df[col_sigue].apply(U.featurize_sigue)
        df = pd.concat([df, sigue_feats], axis=1)

    # --- Asistencia (Cell 14)
    col_asist = "¿Has asistido alguna vez a un partido de fútbol femenino en vivo?"
    if col_asist in df.columns:
        asist_feats = df[col_asist].apply(U.featurize_asistencia)
        df = pd.concat([df, asist_feats], axis=1)

    # --- Percepción patrocinio (Cell 15)
    col_perc = "¿Cómo cambia tu percepción de una marca al verla patrocinando fútbol femenino?"
    if col_perc in df.columns:
        df["percep_patrocinio_ord"] = df[col_perc].apply(U.map_percepcion)

    # --- Compra por patrocinio (Cell 16)
    col_compra = "¿Has comprado un producto o usado un servicio porque patrocinaba un equipo o atleta de deportes femeninos?"
    if col_compra in df.columns:
        df["compra_patrocinio_cat"] = df[col_compra].apply(U.map_compra).astype("category")
        df["compra_influenciada"] = df["compra_patrocinio_cat"].map({"si": 1, "no": 0}).astype("Int64")

    # --- Importancia marcas (Cell 17)
    col_imp = "¿Qué tan importante es para ti que las marcas apoyen el deporte femenino?"
    if col_imp in df.columns:
        imp_num = pd.to_numeric(df[col_imp], errors="coerce").astype("Int64")
        df["importancia_marcas"] = imp_num
        df["importancia_marcas_cat"] = pd.Categorical(imp_num, categories=[1, 2, 3, 4, 5], ordered=True)

    # --- Inversión igual (Cell 18)
    col_inv = "¿Crees que el fútbol femenino debería recibir la misma inversión comercial que el masculino?"
    if col_inv in df.columns:
        inv_feats = df[col_inv].apply(U.map_inversion)
        df = pd.concat([df, inv_feats], axis=1)
        df["inversion_igual_cat"] = df["inversion_igual_cat"].astype("category")
        dummies_inv = pd.get_dummies(df["inversion_igual_cat"], prefix="inversion_igual", dtype="Int64")
        df = pd.concat([df, dummies_inv], axis=1)

    # --- Actitud marcas (Cell 19)
    col_act = "¿Apoyarías o boicotearías una marca según su apoyo al fútbol femenino?"
    if col_act in df.columns:
        act_feats = df[col_act].apply(U.map_actitud)
        df = pd.concat([df, act_feats], axis=1)
        df["actitud_marca_ff_cat"] = df["actitud_marca_ff_cat"].astype("category")
        dummies_act = pd.get_dummies(df["actitud_marca_ff_cat"], prefix="actitud_marca_ff", dtype="Int64")
        df = pd.concat([df, dummies_act], axis=1)

    # --- Sentimiento campañas (Cell 20)
    col_sent = "¿Qué sientes cuando las marcas usan a deportistas femeninas en sus campañas o anuncios?"
    if col_sent in df.columns:
        df["campanas_deportistas_ord"] = df[col_sent].apply(U.map_sentimiento)

    # --- Crecimiento 5y (Cell 21)
    col_crec = "¿Cómo crees que crecerá el fútbol femenino en tu país en los próximos 5 años?"
    if col_crec in df.columns:
        crec_feats = df[col_crec].apply(U.map_crecimiento)
        df = pd.concat([df, crec_feats], axis=1)

    # --- Desafíos (Cell 22)
    col_des = "¿Cuál es el mayor desafío que enfrenta el fútbol femenino en tu país? (Selecciona los 2 principales)"
    if col_des in df.columns:
        desafios_ohe = df[col_des].apply(U.featurize_desafios)
        df = pd.concat([df, desafios_ohe], axis=1)

    # --- Motivación apoyar (Cell 24)
    col_motiv = "¿Te motivaría más apoyar a un equipo si tuviera y apoyara una sección femenina?"
    if col_motiv in df.columns:
        MOTIV_MAP = {"Sí": 1.0, "Tal vez": 0.5, "No": 0.0}
        df["motiv_apoyar_score"] = df[col_motiv].map(MOTIV_MAP)
        df["motiv_apoyar_cat"] = pd.Categorical(df[col_motiv], categories=["No", "Tal vez", "Sí"], ordered=True)

    # --- Apuestas (Cell 25)
    col_apuestas = "En general, incluyendo fútbol masculino y femenino, ¿con qué frecuencia apuestas en eventos deportivos?"
    if col_apuestas in df.columns:
        APUESTAS_MAP = {"Nunca": 0, "Al menos una vez por mes": 1, "Al menos una vez por semana": 5, "Al menos una vez al día": 10}
        df["apuestas_ord"] = df[col_apuestas].map(APUESTAS_MAP).astype("Int64")
        df["apuesta"] = df["apuestas_ord"].gt(0).astype("Int64")

    # --- Valores (Cell 26)
    col_val = "¿Qué valores asocias con el fútbol femenino? (Selecciona al menos 2)"
    if col_val in df.columns:
        valores_ohe = df[col_val].apply(U.featurize_valores)
        df = pd.concat([df, valores_ohe], axis=1)

    # --- Por qué no ves (Cell 27)
    col_no_ves = "¿Por qué no ves fútbol femenino actualmente? (Selecciona todas las opciones que apliquen)"
    if col_no_ves in df.columns:
        no_ves_ohe = df[col_no_ves].apply(U.featurize_no_ves)
        df = pd.concat([df, no_ves_ohe], axis=1)

    # --- Conoces ligas (Cell 28)
    col_ligas = "¿Conoces alguna liga o torneo de fútbol femenino?"
    if col_ligas in df.columns:
        df["conoce_ligas_ord"] = df[col_ligas].map(U.LIGAS_MAP).astype("Int64")

    # --- Vio contenido último año (Cell 29)
    col_contenido = "¿Has visto algún contenido (video, noticia, post) sobre fútbol femenino en el último año?"
    if col_contenido in df.columns:
        df["vio_contenido_ultimo_anio_ord"] = df[col_contenido].map(U.CONTENIDO_MAP).astype("Int64")

    # --- Necesidades (Cell 30)
    col_need = "¿Qué necesitaría el fútbol femenino para que tú te interesaras más en verlo? (Selecciona todas las opciones que apliquen)"
    if col_need in df.columns:
        need_ohe = df[col_need].apply(U.featurize_need)
        df = pd.concat([df, need_ohe], axis=1)

    # --- Percepción general (Cell 31)
    col_perc_gen = "¿Cómo percibes el fútbol femenino actualmente, en general?"
    if col_perc_gen in df.columns:
        df["percepcion_ff_ord"] = df[col_perc_gen].map(U.PERC_MAP).astype("Float64")
        df["percepcion_ff_sin_opinion"] = (df[col_perc_gen] == "No tengo una opinión formada").astype("Int64")

    # --- Sabía que hay liga (Cell 32)
    col_sabia = "¿Sabías que tu país tiene una liga profesional de fútbol femenino?"
    if col_sabia in df.columns:
        df["sabia_liga_cat"] = pd.Categorical(df[col_sabia], categories=["No", "Tal vez", "Sí"], ordered=True)
        df["conoce_liga_pais"] = df[col_sabia].apply(U.flag_conoce_liga).astype("Float64")

    return df


def wrangle_from_excel(path: str) -> pd.DataFrame:
    return wrangle_respuestas(load_raw_excel(path))

