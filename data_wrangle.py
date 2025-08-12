import pandas as pd
import numpy as np
import utils


def load_raw_excel(path: str) -> pd.DataFrame:
    """Carga el Excel crudo tal como en el notebook.

    No altera nombres ni tipos; solo devuelve el DataFrame.
    """
    return pd.read_excel(path)


def wrangle_respuestas(respuestas_esp: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce la limpieza del notebook aplicando las funciones en el mismo orden.
    Devuelve un DataFrame enriquecido listo para EDA/clustering.
    """
    df = respuestas_esp.copy()

    # Cell 2: drops y non_fan
    drops_esp = ['Columna 11', 'Columna 3', '¿Te gustaría participar en la rifa de un PREMIO? (Toma 1 minuto más y tu participación es completamente anónima)',
           'Ingresa tu código único que incluya:\n- 8 letras (pueden repetirse)\n- 2 números (pueden repetirse)\n- 2 símbolos especiales como: #, @, !, %, &, etc.',
           '¿Te refirió alguien para contestar esta encuesta? Ingresa su código de referido']
    df = df.drop(drops_esp, axis=1)
    df['non_fan'] = df['¿Con qué frecuencia ves fútbol femenino?'].apply(lambda x: 'No Fan' if x == 'Nunca' else 'Fan')

    # Cell 10: relación con el deporte -> dummies
    col_rel = '¿Cuál es tu relación con el deporte? (Elige todas las que apliquen)'
    # Usar normalización estricta como en las primeras celdas
    utils._normalize_key = utils._normalize_key_strict
    df[[f"rel_{c}" for c in utils.rel_cols]] = df[col_rel].apply(utils._ohe_relacion)

    # Cell 11: frecuencia ordinal/categórica
    col_freq = "¿Con qué frecuencia ves fútbol femenino?"
    df["freq_ord"] = df[col_freq].map(utils.FREQ_TO_ORD)
    df["freq_cat"] = pd.Categorical(df[col_freq], categories=utils.FREQ_ORDER, ordered=True)

    # Cell 12: canales en vivo
    # A partir de aquí, el notebook redefinió _normalize_key a la versión "soft"
    utils._normalize_key = utils._normalize_key_soft
    col_canales = "¿A través de qué canales sigues los partidos de fútbol femenino en vivo? (Selecciona todos los que apliquen)"
    canales_feats = df[col_canales].apply(utils.featurize_canales)
    df = pd.concat([df, canales_feats], axis=1)

    # Cell 13: redes sociales
    col_redes = "¿En qué redes sociales sigues contenido de fútbol femenino? (Selecciona todas las que apliquen)"
    redes_feats = df[col_redes].apply(utils.featurize_redes)
    df = pd.concat([df, redes_feats], axis=1)

    # Cell 14: tipo de contenido
    col_tipo = "¿Qué tipo de contenido consumes más? (Selecciona todas las que apliquen)"
    tipos_feats = df[col_tipo].apply(utils.featurize_tipos)
    df = pd.concat([df, tipos_feats], axis=1)

    # Cell 16: sigue equipos/jugadoras
    col_sigue = "¿Sigues a equipos o jugadoras específicas en el fútbol femenino?"
    sigue_feats = df[col_sigue].apply(utils.featurize_sigue)
    df = pd.concat([df, sigue_feats], axis=1)

    # Cell 18: asistencia partidos en vivo
    col_asist = "¿Has asistido alguna vez a un partido de fútbol femenino en vivo?"
    asist_feats = df[col_asist].apply(utils.featurize_asistencia)
    df = pd.concat([df, asist_feats], axis=1)

    # Cell 19: percepción patrocinio
    col_perc = "¿Cómo cambia tu percepción de una marca al verla patrocinando fútbol femenino?"
    df["percep_patrocinio_ord"] = df[col_perc].apply(utils.map_percepcion)

    # Cell 21: compra por patrocinio
    col_compra = "¿Has comprado un producto o usado un servicio porque patrocinaba un equipo o atleta de deportes femeninos?"
    df["compra_patrocinio_cat"] = df[col_compra].apply(utils.map_compra).astype("category")
    # Nota: en el notebook hay un map posterior a {"si":1,"no":0} que no coincide con categorías; lo dejamos igual
    df["compra_influenciada"] = df["compra_patrocinio_cat"].map({"si": 1, "no": 0}).astype("Int64")

    # Cell 22: importancia marcas (Likert 1-5)
    col_imp = "¿Qué tan importante es para ti que las marcas apoyen el deporte femenino?"
    imp_num = pd.to_numeric(df[col_imp], errors="coerce").astype("Int64")
    df["importancia_marcas"] = imp_num
    df["importancia_marcas_cat"] = pd.Categorical(imp_num, categories=[1, 2, 3, 4, 5], ordered=True)

    # Cell 23: inversión igual que masculino
    col_inv = "¿Crees que el fútbol femenino debería recibir la misma inversión comercial que el masculino?"
    inv_feats = df[col_inv].apply(utils.map_inversion)
    df = pd.concat([df, inv_feats], axis=1)
    df["inversion_igual_cat"] = df["inversion_igual_cat"].astype("category")
    dummies_inv = pd.get_dummies(df["inversion_igual_cat"], prefix="inversion_igual", dtype="Int64")
    df = pd.concat([df, dummies_inv], axis=1)

    # Cell 24: actitud hacia marcas
    col_act = "¿Apoyarías o boicotearías una marca según su apoyo al fútbol femenino?"
    act_feats = df[col_act].apply(utils.map_actitud)
    df = pd.concat([df, act_feats], axis=1)
    df["actitud_marca_ff_cat"] = df["actitud_marca_ff_cat"].astype("category")
    dummies_act = pd.get_dummies(df["actitud_marca_ff_cat"], prefix="actitud_marca_ff", dtype="Int64")
    df = pd.concat([df, dummies_act], axis=1)

    # Cell 26: sentimiento campañas con deportistas
    col_sent = "¿Qué sientes cuando las marcas usan a deportistas femeninas en sus campañas o anuncios?"
    df["campanas_deportistas_ord"] = df[col_sent].apply(utils.map_sentimiento)

    return df


def wrangle_from_excel(path: str) -> pd.DataFrame:
    """Atajo: carga el excel y aplica `wrangle_respuestas`."""
    return wrangle_respuestas(load_raw_excel(path))

