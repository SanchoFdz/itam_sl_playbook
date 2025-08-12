import re
import unicodedata
import numpy as np
import pandas as pd


# ==========================
# Normalización de texto
# ==========================

def normalize_key_strict(x: str) -> str:
    """
    a) strip espacios (extremos)
    b) elimina acentos
    c) deja solo letras y espacios
    d) lowercase y colapsa espacios

    Args:
        x (str): La cadena a normalizar

    Returns:
        str: La cadena normalizada
    """
    x = str(x).strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"[^a-zA-Z ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


def normalize_key_soft(x: str) -> str:
    """
    Versión más suave usada en celdas posteriores del notebook:
    - Normaliza acentos
    - Solo colapsa espacios y lowercase (mantiene números y signos)
    """
    x = str(x)
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


# ==========================
# Países
# ==========================

MAP_PAISES = {
    "mexico": "México",
    "mexco": "México",
    "mexuco": "México",
    "cdmx": "México",
    "ciudad de mexico": "México",
    "estados unidos": "Estados Unidos",
    "usa": "Estados Unidos",
    "colombia": "Colombia",
    "ecuador": "Ecuador",
    "peru": "Perú",
    "brasil": "Brasil",
    "argentina": "Argentina",
    "costa rica": "Costa Rica",
    "alemania": "Alemania",
    "dinamarca": "Dinamarca",
    "espana": "España",
    "canada": "Canadá",
    "paises bajos": "Países Bajos",
    "puerto rico": "Puerto Rico",
    "el salvador": "El Salvador",
    "guatemala": "Guatemala",
}


def homogeneizar_pais(x: str) -> str:
    """
    Normaliza el nombre de un país para que sea consistente con el diccionario MAP_PAISES.
    """
    if pd.isna(x):
        return x
    key = normalize_key_strict(x)
    return MAP_PAISES.get(key, str(x).strip())


# ==========================
# Edad y género
# ==========================

def transforma_edad(df: pd.DataFrame, age_col: str) -> pd.DataFrame:
    """
    Transforma la columna de edad en una columna ordinal y una columna de punto medio.
    """
    out = df.copy()
    age_label_to_ordinal = {
        "Menor de 18": 0,
        "18-24": 1,
        "25-34": 2,
        "35-44": 3,
        "45-54": 4,
        "55+": 5,
    }
    age_label_to_midpoint = {
        "Menor de 18": 16.5,
        "18-24": 21.0,
        "25-34": 30.0,
        "35-44": 40.0,
        "45-54": 50.0,
        "55+": 60.0,
    }
    out["edad_ordinal"] = out[age_col].map(age_label_to_ordinal).astype("Int64")
    out["edad_midpoint"] = out[age_col].map(age_label_to_midpoint).astype("float64")
    return out


def transforma_genero(df: pd.DataFrame, gender_col: str) -> pd.DataFrame:
    """
    Transforma la columna de género en dummies `genero_*`.
    """
    out = df.copy()
    out["genero_categoria"] = pd.Categorical(df[gender_col], ordered=False)
    dummies_genero = pd.get_dummies(out["genero_categoria"], prefix="genero", dtype="uint8")
    out = pd.concat([out, dummies_genero], axis=1)
    return out


def slug(x: str) -> str:
    """Normaliza nombres a snake simple (ascii)."""
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"[^a-zA-Z0-9]+", "_", x)
    return x.strip("_").lower()


# ==========================
# Relación con el deporte
# ==========================

REL_COLS = ["fanatico", "atleta_amateur", "atleta_profesional", "trabajo_industria", "no_activo"]


def ohe_relacion(txt: str) -> pd.Series:
    """
    Transforma la columna de relación con el deporte en dummies `rel_*`.
    Usa normalización estricta.
    """
    out = {k: 0 for k in REL_COLS}
    if pd.isna(txt):
        return pd.Series(out)
    t = normalize_key_strict(txt)
    if "fanatic" in t:
        out["fanatico"] = 1
    if "amateur" in t:
        out["atleta_amateur"] = 1
    if "profesional" in t:
        out["atleta_profesional"] = 1
    if "industria" in t or "trabajo en la industria" in t:
        out["trabajo_industria"] = 1
    if "no sigo ni trabajo activamente" in t or re.search(r"\bno sigo\b", t):
        out["no_activo"] = 1
    return pd.Series(out)


FREQ_ORDER = [
    "Nunca",
    "Menos de una vez al mes",
    "Al menos dos veces al mes",
    "Semanalmente",
    "Veo cada partido que puedo",
]

FREQ_TO_ORD = {
    "Nunca": 0,
    "Menos de una vez al mes": 1,
    "Al menos dos veces al mes": 2,
    "Semanalmente": 3,
    "Veo cada partido que puedo": 4,
}


# ==========================
# Canales en vivo
# ==========================

CANALES = [
    "tv_abierta",
    "tv_cable",
    "streaming",
    "redes_sociales",
    "radio",
    "estadio",
    "app_deportiva",
    "youtube",
]

ALIAS2CANAL = {
    # TV abierta / cable
    "tv abierta": "tv_abierta",
    "tv por cable": "tv_cable",
    "sky soccer": "tv_cable",
    "sky": "tv_cable",
    # Streaming / plataformas
    "servicio de streaming": "streaming",
    "streaming": "streaming",
    "youtube": "youtube",
    "dazn": "streaming",
    "danz": "streaming",
    # Redes / radio / estadio / app
    "redes sociales": "redes_sociales",
    "radio": "radio",
    "asistencia al estadio": "estadio",
    "aplicacion deportiva": "app_deportiva",
    # IPTV / Smart TV
    "iptv": "tv_cable",
    "samsung tv": "tv_cable",
}

NEGACIONES_CANAL = ["no sigo los juegos en vivo", "no aplica"]
INDETERMINADO_CANAL = ["donde lo pasen"]


def featurize_canales(txt: str) -> pd.Series:
    """Dummies `canal__*` usando normalización suave."""
    out = {f"canal__{c}": 0 for c in CANALES}
    flags = {"canal__no_en_vivo": 0, "canal__indiferente": 0}
    if pd.isna(txt):
        return pd.Series({**out, **flags})
    t = normalize_key_soft(txt)
    if any(neg in t for neg in NEGACIONES_CANAL):
        flags["canal__no_en_vivo"] = 1
        return pd.Series({**out, **flags})
    if any(ind in t for ind in INDETERMINADO_CANAL):
        flags["canal__indiferente"] = 1
    for alias, canal in ALIAS2CANAL.items():
        if alias in t:
            out[f"canal__{canal}"] = 1
    return pd.Series({**out, **flags})


# ==========================
# Redes sociales
# ==========================

REDES = ["instagram", "twitter_x", "facebook", "tiktok", "youtube"]

ALIAS2RED = {
    "instagram": "instagram",
    "twitter/x": "twitter_x",
    "twitter": "twitter_x",
    "facebook": "facebook",
    "tiktok": "tiktok",
    "youtube": "youtube",
    "you tube": "youtube",
}

NEGACIONES_RED = ["no lo sigo en redes sociales", "no aplica"]


def featurize_redes(txt: str) -> pd.Series:
    """Dummies `rs__*` usando normalización suave."""
    out = {f"rs__{r}": 0 for r in REDES}
    flags = {"rs__no_redes": 0, "rs__no_aplica": 0}
    if pd.isna(txt):
        return pd.Series({**out, **flags})
    t = normalize_key_soft(txt)
    if "no aplica" in t:
        flags["rs__no_aplica"] = 1
        return pd.Series({**out, **flags})
    if "no lo sigo en redes sociales" in t:
        flags["rs__no_redes"] = 1
        return pd.Series({**out, **flags})
    for alias, red in ALIAS2RED.items():
        if alias in t:
            out[f"rs__{red}"] = 1
    return pd.Series({**out, **flags})


# ==========================
# Tipos de contenido
# ==========================

TIPOS = [
    "resumenes_highlights",
    "entrevistas",
    "estadisticas_analisis",
    "detras_camaras",
    "contenido_fans",
    "contenido_marca_patrocinado",
    "noticieros_profesionales",
    "noticieros_independientes",
    "contenido_club",
    'contenido_creado_por_usuarios'
]

ALIAS2TIPO = {
    # Resúmenes
    "resumenes": "resumenes_highlights",
    "highlights": "resumenes_highlights",
    # Entrevistas
    "entrevistas a jugadoras": "entrevistas",
    "entrevistas a entrenadores": "entrevistas",
    "entrevistas": "entrevistas",
    # Estadísticas / análisis
    "estadisticas del juego": "estadisticas_analisis",
    "analisis tactico": "estadisticas_analisis",
    "analisis": "estadisticas_analisis",
    # Detrás de cámaras / vida de equipo
    "detras de camaras": "detras_camaras",
    "vida de equipo": "detras_camaras",
    # Fans / creadoras
    "contenido generado por fans": "contenido_creado_por_usuarios",
    "creadoras de contenido": "contenido_creado_por_usuarios",
    # Marca / patrocinado
    "contenido de marca": "contenido_marca_patrocinado",
    "patrocinado": "contenido_marca_patrocinado",
    # Noticieros
    "noticieros profesionales": "noticieros_profesionales",
    "noticieros independientes": "noticieros_independientes",
    # Club
    "contenido del club": "contenido_club",
}

NEGACIONES_TIPO = ["no aplica"]


def featurize_tipos(txt: str) -> pd.Series:
    """Dummies `cont__*` usando normalización suave."""
    out = {f"cont__{t}": 0 for t in TIPOS}
    flags = {"cont__no_aplica": 0}
    if pd.isna(txt):
        return pd.Series({**out, **flags})
    t = normalize_key_soft(txt)
    if any(neg in t for neg in NEGACIONES_TIPO):
        flags["cont__no_aplica"] = 1
        return pd.Series({**out, **flags})
    for alias, tipo in ALIAS2TIPO.items():
        if alias in t:
            # Nota: el notebook mapea estas dos alias a "contenido_creado_por_usuarios",
            # que no está en TIPOS. Respetamos esa lógica (no se creará columna si no existe en TIPOS).
            key = f"cont__{tipo}"
            if key in out:
                out[key] = 1
    return pd.Series({**out, **flags})


# ==========================
# Sigue equipos / jugadoras
# ==========================

def featurize_sigue(txt: str) -> pd.Series:
    """Devuelve `sigue_equipos` y `sigue_jugadoras` (usa normalización suave)."""
    if pd.isna(txt):
        return pd.Series({"sigue_equipos": np.nan, "sigue_jugadoras": np.nan})
    t = normalize_key_soft(txt)
    if "ambos" in t:
        return pd.Series({"sigue_equipos": 1, "sigue_jugadoras": 1})
    if "equipos" in t:
        return pd.Series({"sigue_equipos": 1, "sigue_jugadoras": 0})
    if "jugadoras" in t:
        return pd.Series({"sigue_equipos": 0, "sigue_jugadoras": 1})
    if "no aplica" in t or t == "no":
        return pd.Series({"sigue_equipos": 0, "sigue_jugadoras": 0})
    return pd.Series({"sigue_equipos": np.nan, "sigue_jugadoras": np.nan})


# ==========================
# Asistencia a partidos
# ==========================

def featurize_asistencia(txt: str) -> pd.Series:
    """Devuelve `asist_ord` y `ha_asistido` (usa normalización suave)."""
    if pd.isna(txt):
        return pd.Series({"asist_ord": np.nan, "ha_asistido": np.nan})
    t = normalize_key_soft(txt)
    if "con frecuencia" in t:
        k = 2
    elif "una o dos" in t or "una o dos veces" in t:
        k = 1
    elif "no aplica" in t:
        return pd.Series({"asist_ord": np.nan, "ha_asistido": 0})
    elif "no" in t:
        k = 0
    else:
        k = np.nan
    ha = (k in (1, 2)) * 1 if not pd.isna(k) else np.nan
    return pd.Series({"asist_ord": k, "ha_asistido": ha})


# ==========================
# Percepción patrocinio
# ==========================

PERCEP_MAP_NORM = {
    "Mucho menos favorable": -2,
    "Algo menos favorable": -1,
    "Sin cambio": 0,
    "Algo más favorable": 1,
    "Mucho más favorable": 2,
}


def map_percepcion(x: str) -> int:
    if pd.isna(x):
        return np.nan
    return PERCEP_MAP_NORM.get(x, np.nan)


# ==========================
# Compra por patrocinio
# ==========================

COMPRA_MAP_NORM = {
    "Sí": 1,
    "No": -1,
    "No estoy seguro/a": 0,
    "No aplica": 0,
}


def map_compra(x: str) -> int:
    if pd.isna(x):
        return np.nan
    return COMPRA_MAP_NORM.get(x, np.nan)


# ==========================
# Inversión igual que masculino
# ==========================

def map_inversion(x: str) -> pd.Series:
    if pd.isna(x):
        return pd.Series({"inversion_igual_ord": np.nan, "inversion_igual_cat": np.nan})
    t = normalize_key_soft(x)
    if t.startswith("si"):
        return pd.Series({"inversion_igual_ord": 1, "inversion_igual_cat": "si"})
    if t == "no":
        return pd.Series({"inversion_igual_ord": 0, "inversion_igual_cat": "no"})
    if "no estoy seguro" in t:
        return pd.Series({"inversion_igual_ord": -11, "inversion_igual_cat": "no_seguro"})
    return pd.Series({"inversion_igual_ord": np.nan, "inversion_igual_cat": np.nan})


# ==========================
# Actitud hacia marcas (apoyo/boicot)
# ==========================

def map_actitud(x: str) -> pd.Series:
    if pd.isna(x):
        return pd.Series({"actitud_marca_ff_ord": np.nan, "actitud_marca_ff_cat": np.nan})
    t = normalize_key_soft(x)
    if "boicot" in t:
        return pd.Series({"actitud_marca_ff_ord": -1, "actitud_marca_ff_cat": "boicot"})
    if "no cambiaria" in t or "no cambia" in t:
        return pd.Series({"actitud_marca_ff_ord": 0, "actitud_marca_ff_cat": "no_cambia"})
    if "apoyar" in t or "apoyaria" in t:
        return pd.Series({"actitud_marca_ff_ord": 1, "actitud_marca_ff_cat": "apoyo"})
    return pd.Series({"actitud_marca_ff_ord": np.nan, "actitud_marca_ff_cat": np.nan})


# ==========================
# Sentimiento campañas con deportistas
# ==========================

def map_sentimiento(x: str):
    if pd.isna(x):
        return np.nan
    t = normalize_key_soft(x)
    if "forzado" in t or "superficial" in t:
        return -1
    if "no lo noto" in t:
        return 0
    if "inspira" in t:
        return 2
    if "me gusta" in t or "confianza" in t:
        return 1
    return np.nan


# ==========================
# Crecimiento a 5 años
# ==========================

def map_crecimiento(x: str) -> pd.Series:
    if pd.isna(x):
        return pd.Series({"crecimiento_5y_ord": np.nan, "crecimiento_5y_no_seguro": 0})
    t = normalize_key_soft(x)
    if "se mantendra igual" in t:
        return pd.Series({"crecimiento_5y_ord": 0, "crecimiento_5y_no_seguro": 0})
    if "crecera lentamente" in t:
        return pd.Series({"crecimiento_5y_ord": 1, "crecimiento_5y_no_seguro": 0})
    if "crecera significativamente" in t:
        return pd.Series({"crecimiento_5y_ord": 2, "crecimiento_5y_no_seguro": 0})
    if "no estoy seguro" in t:
        return pd.Series({"crecimiento_5y_ord": np.nan, "crecimiento_5y_no_seguro": 1})
    return pd.Series({"crecimiento_5y_ord": np.nan, "crecimiento_5y_no_seguro": 0})


# ==========================
# Desafíos (multi)
# ==========================

DESAFIOS_MAP = {
    "desafio_estereotipos_genero": ["Estereotipos de género"],
    "desafio_falta_cobertura_mediatica": ["Falta de cobertura mediática"],
    "desafio_poca_independencia": [
        "Poca independencia de las liga/clubes masculinos",
        "En Argentina a nivel selección e interclubes se mueven e intercambian siempre las mismas jugadoras. Es decir",
    ],
    "desafio_pocas_oportunidades_jovenes": [
        "Falta de talleres que impulsen a las niñas a entrenar desde chicas",
        "Pocas oportunidades para niñas y adolescentes",
    ],
    "desafio_estigma_social": ["Estigma social"],
    "desafio_falta_espacios": [
        "Falta de espacios públicos para el deporte",
        "Experiencia del aficionado (Localidad de estadios y experiencia dentro)",
    ],
    "desafio_promocion_debil": ["Promoción débil"],
    "desafio_baja_calidad_juego": ["Baja calidad de juego"],
    "desafio_poca_inversion": ["Poca inversión", "Poco interés de los directivos de fútbol"],
    "desafio_bajos_salarios": [
        "Bajos salarios de jugadoras",
        "Falta de reglas claras para mantener un balance competitivo en la cancha",
    ],
    "desafio_aficion_no_crece": ["La afición no está creciendo"],
}


def _split_multi(s: str) -> set:
    return {p.strip() for p in s.split(",")} if isinstance(s, str) else set()


def featurize_desafios(s: str) -> pd.Series:
    opts = _split_multi(s)
    out = {}
    for col_name, frases in DESAFIOS_MAP.items():
        out[col_name] = 1 if any(frase in opts for frase in frases) else 0
    return pd.Series(out)


# ==========================
# Valores (multi)
# ==========================

VAL_TOKENS = {
    "pasion": "Pasión",
    "liderazgo": ["Liderazgo", "Lealtad"],
    "profesionalismo": [
        "Profesionalismo",
        "En proceso de profesionalismo",
        "Perseverancia y honestidad",
    ],
    "esfuerzo": ["Esfuerzo", "Perseverancia y honestidad"],
    "resiliencia": [
        "Resiliencia",
        "Todo les cuesta el doble con tal de alcanzar lo mismo que el fútbol masculino obtiene por el hecho de existir.",
    ],
    "empoderamiento": "Empoderamiento",
    "igualdad": ["Igualdad", "Inclusión"],
    "superacion": "Superación",
    "trabajo_equipo": "Trabajo en equipo",
}


def featurize_valores(s: str) -> pd.Series:
    opts = _split_multi(s)
    out = {}
    for k, etiquetas in VAL_TOKENS.items():
        if isinstance(etiquetas, str):
            out[f"valor__{k}"] = 1 if etiquetas in opts else 0
        else:
            out[f"valor__{k}"] = 1 if any(e in opts for e in etiquetas) else 0
    return pd.Series(out)


# ==========================
# ¿Por qué no ves? (multi)
# ==========================

NO_VES_TOKENS = {
    "prefiere_masculino": [
        "Prefiero ver fútbol masculino",
        "Prefiero ver fútbol masculino, prefiero ver nauru vs vanuatu sub-15",
    ],
    "no_se_donde_ver": ["No sé dónde verlo / no está disponible"],
    "nivel_competitivo": ["Siento que no tiene suficiente nivel competitivo"],
    "no_identificacion": ["No me identifico con los equipos o jugadoras"],
    "no_interesa_en_general": ["No me interesa el fútbol en general"],
    "no_tiempo": ["No tengo tiempo"],
    "nunca_lo_plantee": ["Nunca me lo he planteado"],
    "otro": ["No veo la tele"],
}


def featurize_no_ves(s: str) -> pd.Series:
    opts = _split_multi(s)
    out = {}
    for k, frases in NO_VES_TOKENS.items():
        out[f"no_ves__{k}"] = int(any(frase in opts for frase in frases))
    return pd.Series(out)


# ==========================
# Necesidades (multi)
# ==========================

NEED_TOKENS = {
    "mas_difusion": ["Más difusión en TV o redes"],
    "jugadoras_conocidas_historias": ["Jugadoras más conocidas o con historias personales"],
    "mejor_nivel": ["Mejor nivel competitivo"],
    "clubes_profesionales": ["Clubes o ligas más profesionales"],
    "mayor_separacion_masculino": ["Mayor separación del fútbol masculino"],
    "mas_contenido_atractivo": ["Más contenido atractivo"],
    "mayor_asociacion_con_marcas": ["Asociación con marcas que me atraen"],
    "recomendacion_influencer": ["Que lo recomiende alguien que sigo"],
    "nada": ["Nada me haría verlo"],
}


def featurize_need(s: str) -> pd.Series:
    opts = _split_multi(s)
    out = {}
    for k, frases in NEED_TOKENS.items():
        out[f"need__{k}"] = int(any(frase in opts for frase in frases))
    return pd.Series(out)


# ==========================
# Mapas simples (ligas, contenido, percepción general, sabía liga)
# ==========================

LIGAS_MAP = {
    "No, ninguna": 0,
    "Me suenan, pero no conozco una en concreto": 1,
    "Sí, una o dos": 2,
    "Sí, varias": 3,
}

CONTENIDO_MAP = {
    "No": -1,
    "Sí, alguna vez": 1,
    "Sí, muchas veces": 2,
    "No lo recuerdo bien": 0,
}

PERC_MAP = {
    "Amateur": -1,
    "Aún en crecimiento": 1,
    "Profesional, pero poco difundido": 2,
    "Igual de valioso que el masculino, pero subestimado": 3,
    "No tengo una opinión formada": 0,
}


def flag_conoce_liga(v: str) -> float:
    if v == "Sí":
        return 1
    if v == "No":
        return -1
    if v == "Tal vez":
        return 0
    return np.nan

