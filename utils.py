import re
import unicodedata
import numpy as np
import pandas as pd


# Nota: En el notebook hay dos versiones de _normalize_key definidas en momentos distintos.
# Para preservar ese comportamiento, exponemos ambas y dejamos _normalize_key apuntando a la versión "estricta" por defecto.
def _normalize_key_strict(x: str) -> str:
    """
    a) strip espacios (extremos)
    b) elimina acentos
    c) deja solo letras y espacios
    d) lowercase y colapsa espacios

    Args:
        x (str): La cadena a normalizar

    Returns:
        str: La cadena normalizada (estricta)
    """
    x = str(x).strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"[^a-zA-Z ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


def _normalize_key_soft(x: str) -> str:
    """
    Versión más suave usada en celdas posteriores del notebook:
    - Normaliza acentos y espacios
    - Mantiene números y signos (solo colapsa espacios)
    """
    x = str(x)
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


# Alias por compatibilidad con el notebook (inicia en modo "estricto")
_normalize_key = _normalize_key_strict


MAP = {
    # Muy casero pero es lo que tenemos y lo que funciona
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
    Normaliza el nombre de un país para que sea consistente con el diccionario MAP.

    Args:
        x (str): El nombre del país a normalizar

    Returns:
        str: El nombre del país normalizado
    """
    if pd.isna(x):
        return x
    key = _normalize_key(x)
    return MAP.get(key, str(x).strip())


def transforma_edad(df: pd.DataFrame, age_col: str) -> pd.DataFrame:
    """
    Transforma la columna de edad en una columna ordinal y una columna de punto medio.

    Args:
        df (pd.DataFrame): El DataFrame con la columna de edad
        age_col (str): El nombre de la columna de edad

    Returns:
        pd.DataFrame: El DataFrame con las columnas de edad ordinal y punto medio
    """
    out = df.copy()
    # Orden lógico para análisis/visualización y uso en modelado
    age_order = ["Menor de 18", "18-24", "25-34", "35-44", "45-54", "55+"]
    age_label_to_ordinal = {
        "Menor de 18": 0,
        "18-24": 1,
        "25-34": 2,
        "35-44": 3,
        "45-54": 4,
        "55+": 5,
    }
    # Punto medio para analísis demográfico
    # En 18 pongo 16.0 porque es el punto medio entre 15 y 18 y está a 5 años de distancia de los otros grupos
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
    Transforma la columna de género en dummies.

    Args:
        df (pd.DataFrame): El DataFrame con la columna de género
        gender_col (str): El nombre de la columna de género

    Returns:
        pd.DataFrame: El DataFrame con las columnas de género dummies
    """
    out = df.copy()
    out["genero_categoria"] = pd.Categorical(gender_col, ordered=False)
    # Transformación en dummies del estilo genero_{genero}
    # OJO: se quedan las 3 categorías, hombre, mujer y otro.
    dummies_genero = pd.get_dummies(out["genero_categoria"], prefix="genero", dtype="uint8")
    out = pd.concat([out, dummies_genero], axis=1)
    return out


def _slug(x: str) -> str:
    """
    Normaliza el nombre de una columna para que sea consistente con el diccionario MAP.

    Args:
        x (str): El nombre de la columna a normalizar

    Returns:
        str: El nombre de la columna normalizado
    """
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"[^a-zA-Z0-9]+", "_", x)
    return x.strip("_").lower()


rel_cols = ["fanatico", "atleta_amateur", "atleta_profesional", "trabajo_industria", "no_activo"]


def _ohe_relacion(txt: str) -> pd.Series:
    """
    Transforma la columna de relación con el deporte en dummies.

    Args:
        txt (str): El texto de la columna de relación con el deporte

    Returns:
        pd.Series: Las columnas de relación con el deporte dummies
    """
    out = {k: 0 for k in rel_cols}
    if pd.isna(txt):
        return pd.Series(out)
    t = _normalize_key(txt)
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
    "danz": "streaming",  # aparece como danz jaja
    # Redes / radio / estadio / app
    "redes sociales": "redes_sociales",
    "radio": "radio",
    "asistencia al estadio": "estadio",
    "aplicacion deportiva": "app_deportiva",
    # IPTV / Smart TV
    "iptv": "tv_cable",
    "samsung tv": "tv_cable",
}


NEGACIONES = ["no sigo los juegos en vivo", "no aplica"]
INDETERMINADO = ["donde lo pasen"]


def featurize_canales(txt: str) -> pd.Series:
    """
    Featuriza las respuestas de la columna de canales en una serie de dummies.

    Args:
        txt (str): La respuesta de la columna de canales

    Returns:
        pd.Series: La serie de dummies
    """
    out = {f"canal__{c}": 0 for c in CANALES}
    flags = {"canal__no_en_vivo": 0, "canal__indiferente": 0}
    if pd.isna(txt):
        return pd.Series({**out, **flags})
    t = _normalize_key(txt)
    if any(neg in t for neg in NEGACIONES):
        flags["canal__no_en_vivo"] = 1
        return pd.Series({**out, **flags})
    if any(ind in t for ind in INDETERMINADO):
        flags["canal__indiferente"] = 1
    for alias, canal in ALIAS2CANAL.items():
        if alias in t:
            out[f"canal__{canal}"] = 1
    return pd.Series({**out, **flags})


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
    """
    Featuriza las respuestas de la columna de redes en una serie de dummies.

    Args:
        txt (str): La respuesta de la columna de redes

    out = {f"rs__{r}": 0 for r in REDES}
    """
    out = {f"rs__{r}": 0 for r in REDES}
    flags = {"rs__no_redes": 0, "rs__no_aplica": 0}
    if pd.isna(txt):
        return pd.Series({**out, **flags})
    t = _normalize_key(txt)
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
    "creadoras_contenido",
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
    """
    Featuriza las respuestas de la columna de tipos en una serie de dummies.

    Args:
        txt (str): La respuesta de la columna de tipos

    Returns:
        pd.Series: La serie de dummies
    """
    out = {f"cont__{t}": 0 for t in TIPOS}
    flags = {"cont__no_aplica": 0}
    if pd.isna(txt):
        return pd.Series({**out, **flags})
    t = _normalize_key(txt)
    if any(neg in t for neg in NEGACIONES_TIPO):
        flags["cont__no_aplica"] = 1
        return pd.Series({**out, **flags})
    for alias, tipo in ALIAS2TIPO.items():
        if alias in t:
            out[f"cont__{tipo}"] = 1
    return pd.Series({**out, **flags})


def featurize_sigue(txt: str) -> pd.Series:
    """
    Featuriza las respuestas de la columna de sigue en una serie de dummies.

    Args:
        txt (str): La respuesta de la columna de sigue

    Returns:
        pd.Series: La serie de dummies
    """
    if pd.isna(txt):
        return pd.Series({"sigue_equipos": np.nan, "sigue_jugadoras": np.nan})
    t = _normalize_key(txt)
    if "ambos" in t:
        return pd.Series({"sigue_equipos": 1, "sigue_jugadoras": 1})
    if "equipos" in t:
        return pd.Series({"sigue_equipos": 1, "sigue_jugadoras": 0})
    if "jugadoras" in t:
        return pd.Series({"sigue_equipos": 0, "sigue_jugadoras": 1})
    if "no aplica" in t or t == "no":
        return pd.Series({"sigue_equipos": 0, "sigue_jugadoras": 0})
    return pd.Series({"sigue_equipos": np.nan, "sigue_jugadoras": np.nan})


def featurize_asistencia(txt: str) -> pd.Series:
    """
    Featuriza las respuestas de la columna de asistencia en una serie de dummies.

    Args:
        txt (str): La respuesta de la columna de asistencia

    Returns:
        pd.Series: La serie con `asist_ord` y `ha_asistido`
    """
    if pd.isna(txt):
        return pd.Series({"asist_ord": np.nan, "ha_asistido": np.nan})
    t = _normalize_key(txt)
    if "con frecuencia" in t:
        k = 2
    elif "una o dos" in t or "una o dos veces" in t:
        k = 1
    elif "no aplica" in t:
        return pd.Series({"asist_ord": np.nan, "ha_asistido": 0})
    elif "no" in t:  # cubre "no, pero me gustaria" y "no, y no me interesa"
        k = 0
    else:
        k = np.nan
    ha = (k in (1, 2)) * 1 if not pd.isna(k) else np.nan
    return pd.Series({"asist_ord": k, "ha_asistido": ha})


PERCEP_MAP_NORM = {
    "Mucho menos favorable": -2,
    "Algo menos favorable": -1,
    "Sin cambio": 0,
    "Algo más favorable": 1,
    "Mucho más favorable": 2,
}


def map_percepcion(x: str) -> int:
    """
    Mapea la percepción de una marca al verla patrocinando fútbol femenino a un valor ordinal.

    Args:
        x (str): La respuesta de la columna de percepción

    Returns:
        int: El valor ordinal de la percepción
    """
    if pd.isna(x):
        return np.nan
    return PERCEP_MAP_NORM.get(x, np.nan)


COMPRA_MAP_NORM = {
    "Sí": 1,
    "No": -1,
    "No estoy seguro/a": 0,
    "No aplica": 0,
}


def map_compra(x: str) -> int:
    """
    Mapea la respuesta de la columna de compra a un valor ordinal.

    Args:
        x (str): La respuesta de la columna de compra

    Returns:
        int: El valor ordinal de la compra
    """
    if pd.isna(x):
        return np.nan
    return COMPRA_MAP_NORM.get(x, np.nan)


def map_inversion(x: str) -> pd.Series:
    """
    Mapea la respuesta de la columna de inversión a un valor ordinal y categoría.

    Args:
        x (str): La respuesta de la columna de inversión

    Returns:
        pd.Series: La serie de dummies
    """
    if pd.isna(x):
        return pd.Series({"inversion_igual_ord": np.nan, "inversion_igual_cat": np.nan})
    t = _normalize_key(x)
    if t.startswith("si"):
        return pd.Series({"inversion_igual_ord": 1, "inversion_igual_cat": "si"})
    if t == "no":
        return pd.Series({"inversion_igual_ord": 0, "inversion_igual_cat": "no"})
    if "no estoy seguro" in t:
        return pd.Series({"inversion_igual_ord": -11, "inversion_igual_cat": "no_seguro"})
    return pd.Series({"inversion_igual_ord": np.nan, "inversion_igual_cat": np.nan})


def map_actitud(x):
    """
    Mapea actitud hacia marcas a ordinal y categoría.
    """
    if pd.isna(x):
        return pd.Series({"actitud_marca_ff_ord": np.nan, "actitud_marca_ff_cat": np.nan})
    t = _normalize_key(x)
    if "boicot" in t:
        return pd.Series({"actitud_marca_ff_ord": -1, "actitud_marca_ff_cat": "boicot"})
    if "no cambiaria" in t or "no cambia" in t:
        return pd.Series({"actitud_marca_ff_ord": 0, "actitud_marca_ff_cat": "no_cambia"})
    if "apoyar" in t or "apoyaria" in t:
        return pd.Series({"actitud_marca_ff_ord": 1, "actitud_marca_ff_cat": "apoyo"})
    return pd.Series({"actitud_marca_ff_ord": np.nan, "actitud_marca_ff_cat": np.nan})


def map_sentimiento(x):
    """
    Mapea sentimiento respecto a campañas con deportistas a escala ordinal.
    """
    if pd.isna(x):
        return np.nan
    t = _normalize_key(x)
    if "forzado" in t or "superficial" in t:
        return -1
    if "no lo noto" in t:
        return 0
    if "inspira" in t:
        return 2
    if "me gusta" in t or "confianza" in t:
        return 1
    return np.nan

