# -*- coding: utf-8 -*-
"""
AI-powered 48-column unifier for the waterproject repo.

Inputs:
  - LONG CSV (GitHub): https://raw.githubusercontent.com/AIforimpact22/waterproject/main/all_sheets_first10_long.csv
    (You can override with env: LONG_CSV=<path or URL>)

Outputs (created in ./unified):
  - unified_48cols.csv        → the 48-column wide table
  - mapping_used.json         → final raw→canonical header mapping
  - conflicts_log.csv         → decisions kept for traceability
  - ai_validation.md          → short AI audit (if OPENAI_API_KEY is available)

Environment (optional):
  - OPENAI_API_KEY  → enables GPT assistance (header mapping & final audit)
  - OPENAI_MODEL    → default: gpt-4o-mini (fallbacks handled)
  - AI_MODE         → ambiguous_only | always | disabled (default: ambiguous_only)
  - AI_TIMEOUT_SEC  → per AI chunk (default: 12)
  - AI_CHUNK_SIZE   → raw headers per AI chunk (default: 60)
  - AI_MAX_CHUNKS   → cap number of chunks (default: 10)
"""

import os, re, json, time, math, warnings
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
import pandas as pd

# ---------------------- Config ----------------------
LONG_CSV = os.environ.get(
    "LONG_CSV",
    "https://raw.githubusercontent.com/AIforimpact22/waterproject/main/all_sheets_first10_long.csv"
)

OUT_DIR       = os.environ.get("OUT_DIR", "./unified")
OUT_UNIFIED   = os.path.join(OUT_DIR, "unified_48cols.csv")
OUT_MAP       = os.path.join(OUT_DIR, "mapping_used.json")
OUT_CONFLICTS = os.path.join(OUT_DIR, "conflicts_log.csv")
OUT_AI_MD     = os.path.join(OUT_DIR, "ai_validation.md")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY")

AI_TIMEOUT_SEC = int(os.environ.get("AI_TIMEOUT_SEC", "12"))
AI_MAX_CHUNKS  = int(os.environ.get("AI_MAX_CHUNKS", "10"))
AI_CHUNK_SIZE  = int(os.environ.get("AI_CHUNK_SIZE", "60"))
AI_MODE        = os.environ.get("AI_MODE", "ambiguous_only")  # ambiguous_only | always | disabled

# ------------------ Canonical Schema -----------------
CANONICAL_48 = [
    "vifir_kod","vor_kod","obj_tipus","objszam","objhely",
    "csoporthely","csoportszam","objcsop_kod","gw_kod","vizig",
    "telep_nev","telepules","irsz","cim","helyrajzi_szam","eov_x","eov_y","tszf_m",
    "ev","ho","nap","datum",
    "letesites_eve","viztipus_rt","reteg_teteje_m","reteg_alja_m","talp_m",
    "szuro_teteje_m","szuro_alja_m","szuro_db","szuro_hossz_m",
    "nyugalmi_vizszint_m","uzemi_vizszint_m","homerseklet_c","hozam_lperc","termeles_ezer_m3ev",
    "engedely_szam","uzem_engedely_szam","vizikonyv_szam","ervenyes_tol","ervenyes_ig",
    "enged_termeles_m3nap","lekotes_ezer_m3ev","lekotes_m3nap","vkj_kategoria",
    "vizhasznalat","vizhasznalat_alkat","megjegyzes"
]

# Description (kept minimal; used for AI hints)
CANON_DESC = {
    "vifir_kod":"VIFIR object code",
    "vor_kod":"VOR code",
    "obj_tipus":"object type (k/e/h/t/R/T)",
    "objszam":"object serial number (K-28 → 28)",
    "objhely":"object place code",
    "csoporthely":"group place code",
    "csoportszam":"group number",
    "objcsop_kod":"object group code",
    "gw_kod":"GW code",
    "vizig":"VIZIG id",
    "telep_nev":"site name",
    "telepules":"settlement",
    "irsz":"postal code",
    "cim":"address",
    "helyrajzi_szam":"land registry/cadastral no.",
    "eov_x":"EOV X","eov_y":"EOV Y","tszf_m":"TSZF (m)",
    "ev":"year","ho":"month","nap":"day","datum":"ISO date",
    "letesites_eve":"construction year",
    "viztipus_rt":"water type (R/T/R+T)",
    "reteg_teteje_m":"layer top (m)","reteg_alja_m":"layer bottom (m)","talp_m":"well bottom (m)",
    "szuro_teteje_m":"screen top (m)","szuro_alja_m":"screen bottom (m)","szuro_db":"screen count","szuro_hossz_m":"screen length (m)",
    "nyugalmi_vizszint_m":"static WL (m)","uzemi_vizszint_m":"pumping WL (m)","homerseklet_c":"temp (°C)",
    "hozam_lperc":"yield (L/min)","termeles_ezer_m3ev":"production (k m³/yr)",
    "engedely_szam":"permit no.","uzem_engedely_szam":"operating permit no.","vizikonyv_szam":"water-book no.",
    "ervenyes_tol":"valid from","ervenyes_ig":"valid to",
    "enged_termeles_m3nap":"permitted prod (m³/day)","lekotes_ezer_m3ev":"commitment (k m³/yr)","lekotes_m3nap":"commitment (m³/day)",
    "vkj_kategoria":"VKJ category","vizhasznalat":"use (main)","vizhasznalat_alkat":"use (sub)","megjegyzes":"note"
}

# Hard/priority mapping & numeric tolerances
HARD_LOCK_MAP: Dict[str, str] = {
    "SZURO_H": "szuro_hossz_m",
    "SZURO_A": "szuro_teteje_m",
    "SZURO_F": "szuro_alja_m",
    "RETEG_F": "reteg_teteje_m",
    "RETEG_A": "reteg_alja_m",
    "TALP":    "talp_m",
    "MAX_TALP":"talp_m",
    "NYUGALMI":"nyugalmi_vizszint_m",
    "UZEMI":   "uzemi_vizszint_m",
    "HOMERS":  "homerseklet_c",
    "HOZAM":   "hozam_lperc",
    "EOVX":    "eov_x",
    "EOVY":    "eov_y",
    "TSZF":    "tszf_m",
    "HELYI NÉV": "telep_nev",
    "HELYI_NÉV": "telep_nev",
    "HELYI_NEV": "telep_nev",
    "HELYI_NEV (Telep neve)": "telep_nev",
    "Telep": "telep_nev",
    "Telep település": "telepules",
    "TELEPÜLÉS": "telepules",
    "TELEPULES": "telepules",
    "Település": "telepules",
    "TELEPULES_KAT": "telepules",
    "TELEPÜLÉS_TELEP ": "telepules",
}
RESTRICTED_ALLOWED_CANON: Dict[str, List[str]] = {
    "SZURO_H": ["szuro_hossz_m"],
    "SZURO_A": ["szuro_teteje_m"],
    "SZURO_F": ["szuro_alja_m"],
    "RETEG_F": ["reteg_teteje_m"],
    "RETEG_A": ["reteg_alja_m"],
    "TALP":    ["talp_m"],
    "MAX_TALP":["talp_m"],
    "VGT1_ObjAz": [],  # safety: never map to vor_kod
}
SOURCE_PREF_NUM = {
    "szuro_hossz_m": ["SZURO_H"],
    "szuro_teteje_m": ["SZURO_A"],
    "szuro_alja_m": ["SZURO_F"],
    "reteg_teteje_m": ["RETEG_F"],
    "reteg_alja_m": ["RETEG_A"],
    "talp_m": ["TALP","MAX_TALP"],
    "eov_x": ["EOVX"],
    "eov_y": ["EOVY"],
    "tszf_m": ["TSZF"],
    "nyugalmi_vizszint_m": ["NYUGALMI"],
    "uzemi_vizszint_m": ["UZEMI"],
    "homerseklet_c": ["HOMERS"],
    "hozam_lperc": ["HOZAM"],
}
SOURCE_PREF_TEXT = {
    "telep_nev": ["HELYI NÉV","HELYI_NÉV","HELYI_NEV","HELYI_NEV (Telep neve)","Telep"],
    "telepules": ["TELEPÜLÉS","TELEPULES","Település","Telep település","TELEPULES_KAT","TELEPÜLÉS_TELEP "],
    "csoporthely": ["CSOPORTHELY\nmodósítás kék betűvel és cellaszínnel.","CSOPORTHELY","CSOPHELY","OBJHELY"],
    "objcsop_kod": ["OBJCSOPKOD","OBJCSOP","OBJCSOP2018","OBJCSOP2017","OBJCSOP2016","OBJCSOP2015","OBJCSOP2014","OBJCSOP2013"],
    "vor_kod": ["VOR"],
    "vizikonyv_szam": ["VK szám","VK","Vizikönyvi szám","VK szám.1","VK_G","vk"],
    "uzem_engedely_szam": ["ÜZ.ENG.","UZENG","UZENGSZAM","UZENG_SZAM_2015","UZENG_TELEP","UZENGSZAM (telep)"],
    "engedely_szam": ["Engedélyszám","Engedélyszám_Gkód","ÜZENG"],
}
NUM_TOL = {
    "szuro_hossz_m": (1.0, 0.10),
    "szuro_teteje_m": (0.5, 0.05),
    "szuro_alja_m": (0.5, 0.05),
    "reteg_teteje_m": (1.0, 0.10),
    "reteg_alja_m": (1.0, 0.10),
    "talp_m": (1.0, 0.02),
    "eov_x": (1.0, 1e-5),
    "eov_y": (1.0, 1e-5),
    "tszf_m": (0.3, 0.01),
    "homerseklet_c": (0.5, 0.05),
    "hozam_lperc": (5.0, 0.10),
    "nyugalmi_vizszint_m": (0.2, 0.02),
    "uzemi_vizszint_m": (0.5, 0.05),
}

# ---------------------- Utilities --------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def normalize_header(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).replace("\u00A0", " ")
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # handle "('CSOPHELY',)" formatting from earlier profiling
    m = re.match(r"^\(\s*'([^']+)'\s*,\s*\)$", s)
    if m:
        s = m.group(1)
    return s

def normalize_value_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).replace("\u00A0", " ").strip()

def looks_numeric_token(s: str) -> bool:
    if not s:
        return False
    s1 = re.sub(r"[ €$£%]", "", s.replace(" ", ""))
    if re.search(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]", s1):
        return False
    return bool(re.search(r"\d", s1))

def parse_number_hu_safe(s) -> Optional[float]:
    # robust for pd.NA/NaN/None
    if s is None:
        return None
    try:
        if pd.isna(s):
            return None
    except Exception:
        pass
    s0 = str(s).strip()
    # e.g. "K-28" → 28
    m = re.match(r"^[A-Za-z]{1,3}[-/ ]+(\d+)$", s0)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    if not looks_numeric_token(s0):
        return None
    s1 = s0.replace(" ", "").replace("\u00A0", "")
    s1 = re.sub(r"[€$£%]", "", s1)
    if "," in s1 and "." in s1:
        if s1.rfind(",") > s1.rfind("."):
            s1 = s1.replace(".", "").replace(",", ".")
        else:
            s1 = s1.replace(",", "")
    elif "," in s1:
        s1 = s1.replace(",", ".")
    s1 = re.sub(r"[^0-9.\-+eE]", "", s1)
    if s1 in {"", "-", "+", ".", "e", "E"}:
        return None
    try:
        v = float(s1)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _unify_sep_drop_trailing_dot_series(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"[./]", "-", regex=True)
              .str.replace(r"\.$", "", regex=True)
              .str.strip())

def parse_date_quiet(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace("\u00A0", " ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(_unify_sep_drop_trailing_dot_series(s), errors="coerce", dayfirst=True)

def is_alpha_dominant(text: str) -> bool:
    t = normalize_value_str(text)
    return bool(re.search(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]", t))

def is_vor_code(s: str) -> bool:
    t = normalize_value_str(s)
    return bool(re.match(r"^[A-Z]{2,3}\d{3}$", t)) or bool(re.match(r"^[kehtr]\d{9}$", t, flags=re.IGNORECASE))

def split_obj_code(s: str) -> Tuple[Optional[str], Optional[float]]:
    t = normalize_value_str(s)
    if not t:
        return None, None
    m = re.match(r"^[A-Za-z]{1,3}[-/ ]+(\d+)$", t)
    if m:
        tp = t.strip()[0].lower()
        return tp, float(m.group(1))
    m2 = re.match(r"^(\d+)$", t)
    if m2:
        return None, float(m2.group(1))
    return None, None

# ---------------- Heuristic header mapping -------------
REGEX_RULES = [
    (r"^VIFIR.*$|^VIFIRKOD.*$|^VIFIRKód.*$", "vifir_kod"),
    (r"^VOR(_.*)?$|^kút VOR$", "vor_kod"),
    (r"^OBJKOD$|^OBJ(TIPUS|_TIPUS)$|^OBJ_TIPUS$", "obj_tipus"),
    (r"^OBJSZAM|^OBJDB$|^kút száma$|^vízmű száma$", "objszam"),
    (r"^OBJHELY$", "objhely"),
    (r"^CSOP(HELY|ORTHELY|_HELY_SZÁM)(\.?\d+)?$", "csoporthely"),
    (r"^CSOP(SZAM|ORTSZÁM)(\.?\d+)?$", "csoportszam"),
    (r"^OBJCSOP(KOD)?(20\d{2})?.*$|^OBJ_CSOP$", "objcsop_kod"),
    (r"^GW_?KÓD$|^GW_?KOD$|^GWKOD$|^gw_kod$|^GW_KOD$", "gw_kod"),
    (r"^VIZIG(KOD)?$|^KULDO_VIZIG$|^ILLETEKES_VIZIG$", "vizig"),
    (r"^HELYI.?NÉV.*$|^HELYI_NEV.*$|^HELYINEV.*$|^HELYI NÉV KAT\.$", "telep_nev"),
    (r"^TELEPÜLÉS.*$|^TELEPULES.*$|^Település.*$|^Telep település$", "telepules"),
    (r"^IRSZ$", "irsz"),
    (r"^Cím$", "cim"),
    (r"^HRSZ$|^kataszteri szám$", "helyrajzi_szam"),
    (r"^EOVX$", "eov_x"),
    (r"^EOVY$", "eov_y"),
    (r"^TSZF$", "tszf_m"),
    (r"^EV$", "ev"),
    (r"^HO$|^HÓ_?TELEP$", "ho"),
    (r"^NAP(_?TELEP)?$", "nap"),
    (r"^DATUM$", "datum"),
    (r"^LETESITESEVE$|^LETESITES$", "letesites_eve"),
    (r"^(VÍZ|VIZ)TÍPUS.*$|^VIZTIPUS$|^TIPUS(\.1)?$", "viztipus_rt"),
    (r"^RETEG_F$", "reteg_teteje_m"),
    (r"^RETEG_A$", "reteg_alja_m"),
    (r"^TALP$|^MAX_TALP$", "talp_m"),
    (r"^SZURO_A$", "szuro_teteje_m"),
    (r"^SZURO_F$", "szuro_alja_m"),
    (r"^SZURO_H$", "szuro_hossz_m"),
    (r"^SZURO_DB$", "szuro_db"),
    (r"^NYUGALMI$", "nyugalmi_vizszint_m"),
    (r"^UZEMI$", "uzemi_vizszint_m"),
    (r"^HOMERS(\.1)?$", "homerseklet_c"),
    (r"^HOZAM( \(l/perc\))?$|^HOZAM$", "hozam_lperc"),
    (r"^TERMELES.*(ezer m3/év|ezerm3/év).*|^Termelés \[em3\]$", "termeles_ezer_m3ev"),
    (r"^Engedélyszám.*$", "engedely_szam"),
    (r"^(ÜZ\.?ENG\.?|UZENG.*|UZENGSZAM.*)$", "uzem_engedely_szam"),
    (r"^(VK|VK szám|Vizikönyvi szám|vk).*$", "vizikonyv_szam"),
    (r"^Hatályba lép$", "ervenyes_tol"),
    (r"^(Érvényes.*|érvényes.*)$", "ervenyes_ig"),
    (r"^ENG.*\(m3/nap\).*|^ENGMENNY_m3_nap$|^ENGTERM$", "enged_termeles_m3nap"),
    (r"^Lekötés \[m3\]$", "lekotes_ezer_m3ev"),
    (r"^VKJ .*\(m3/nap\).*|^vkj lek.*$", "lekotes_m3nap"),
    (r"^VKJ kategória.*$", "vkj_kategoria"),
    (r"^(VÍZHASZNÁLAT|Vizhasznalat|VIZHASZN(_\d{4})?)$", "vizhasznalat"),
    (r"^(VizhasznAlkat|VizhasznFokat.*)$", "vizhasznalat_alkat"),
    (r"^Megjegyzés.*$|^VIZIG megjegyzés$", "megjegyzes"),
]
def regex_map_header(h: str) -> Optional[str]:
    if h in HARD_LOCK_MAP:
        return HARD_LOCK_MAP[h]
    for pattern, target in REGEX_RULES:
        if re.match(pattern, h, flags=re.IGNORECASE):
            return target
    return None

# ---------------- OpenAI wrapper (dual SDK) -------------
_client_new = None
_client_old = None
def _init_openai_clients():
    global _client_new, _client_old
    if not OPENAI_KEY:
        return
    try:
        # New SDK (>=1.0)
        from openai import OpenAI
        _client_new = OpenAI(api_key=OPENAI_KEY)
    except Exception:
        try:
            # Legacy SDK (<1.0)
            import openai
            openai.api_key = OPENAI_KEY
            _client_old = openai
        except Exception:
            _client_new = None
            _client_old = None

def _ai_call_messages(messages: List[dict], response_format_json: bool = True) -> str:
    if _client_new:
        kwargs = {"model": OPENAI_MODEL, "temperature": 0.0, "messages": messages}
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _client_new.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    elif _client_old:
        # legacy API does not support response_format; return raw text
        resp = _client_old.ChatCompletion.create(model=OPENAI_MODEL, temperature=0.0, messages=messages)
        return resp["choices"][0]["message"]["content"]
    else:
        raise RuntimeError("OpenAI client not initialized")

# -------------- AI mapping of ambiguous headers ----------
def build_profile_hints_from_long(long_df: pd.DataFrame) -> pd.DataFrame:
    # small hint table for AI: per raw header, inferred type and sample values
    parts = []
    for h, g in long_df.groupby("column_name"):
        vals = g["value"].astype("string")
        # naive inferred type
        n_num = vals.map(parse_number_hu_safe).notna().sum()
        n_text = (vals.map(lambda x: not looks_numeric_token(str(x))) ).sum()
        inferred = "integer/float" if n_num > n_text else "string"
        top_vals = "; ".join([f"{k}:{v}" for k, v in vals.value_counts(dropna=False).head(5).items()])
        parts.append({"column_name": h, "inferred_type": inferred, "files_count": g["file"].nunique(),
                      "sheets_count": g["sheet"].nunique(), "top_values": top_vals})
    return pd.DataFrame(parts, dtype="string")

def ai_make_mapping_ambiguous(raw_headers: List[str], hints_df: pd.DataFrame,
                              heur_map: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    # choose ambiguous only unless AI_MODE=always
    ambiguous = []
    for h in raw_headers:
        hm = heur_map.get(h)
        if hm is None or hm not in CANONICAL_48:
            ambiguous.append(h)

    if AI_MODE.lower() == "disabled" or not OPENAI_KEY:
        print("⚙️  AI mapping disabled or no OPENAI_API_KEY; using heuristics only.")
        return {}

    to_query = raw_headers if AI_MODE.lower() == "always" else ambiguous
    if not to_query:
        print("🧭 No ambiguous headers; heuristics fully covered.")
        return {}

    _init_openai_clients()
    if not (_client_new or _client_old):
        print("⚠️  OpenAI SDK not available; fallback to heuristics only.")
        return {}

    canonical_list = [{"name": c, "desc": CANON_DESC.get(c, "")} for c in CANONICAL_48]
    ai_map: Dict[str, Optional[str]] = {}
    chunks = [to_query[i:i+AI_CHUNK_SIZE] for i in range(0, len(to_query), AI_CHUNK_SIZE)]
    chunks = chunks[:AI_MAX_CHUNKS]
    total = len(chunks)

    with ThreadPoolExecutor(max_workers=1) as pool:
        for idx, chunk in enumerate(chunks, 1):
            ctx = hints_df[hints_df["column_name"].isin(chunk)].copy()
            hints = []
            for _, r in ctx.iterrows():
                hints.append({
                    "column_name": r.get("column_name",""),
                    "type": r.get("inferred_type",""),
                    "files": r.get("files_count",""),
                    "sheets": r.get("sheets_count",""),
                    "examples": str(r.get("top_values",""))[:200]
                })
            system_msg = (
                "You are a data integration expert. Map each RAW header to EXACTLY one of the 48 canonical fields "
                "(snake_case list provided). Return ONLY JSON: {\"map\": {\"raw\": \"canonical\"|null}}. "
                "Safety rules: HELYI_NEV → telep_nev; TELEPULES → telepules; never map TELEPULES to telep_nev; "
                "RETEG_F=top, RETEG_A=bottom; SZURO_A=top, SZURO_F=bottom, SZURO_H=length; "
                "Never map VGT1_ObjAz to vor_kod. If unsure, return null."
            )
            user_payload = {"canonical": canonical_list, "raw_headers": chunk, "profile_hints": hints}
            messages = [{"role":"system","content": system_msg},
                        {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}]

            print(f"🤖 AI mapping chunk {idx}/{total} (size={len(chunk)}) …", flush=True)
            future = pool.submit(_ai_call_messages, messages, True)
            try:
                content = future.result(timeout=AI_TIMEOUT_SEC)
                try:
                    data = json.loads(content)
                    mp = data.get("map", {})
                    for k, v in mp.items():
                        ai_map[k] = v
                except Exception:
                    print(f"⚠️  AI returned non-JSON for chunk {idx}; skipping.", flush=True)
            except FuturesTimeout:
                print(f"⏱️  AI mapping chunk {idx} timed out after {AI_TIMEOUT_SEC}s; using heuristics.", flush=True)
                future.cancel()
            except Exception as e:
                print(f"⚠️  AI mapping error on chunk {idx}: {e}; continuing.", flush=True)
            time.sleep(0.15)
    return ai_map

# ---------------- Coercion / Casting -------------------
def coerce_to_canonical(col: str, series: pd.Series) -> pd.Series:
    s = series
    if col in {"datum","ervenyes_tol","ervenyes_ig"}:
        return parse_date_quiet(series.astype("string")).dt.date.astype("string")

    numeric_cols = {
        "eov_x","eov_y","tszf_m","reteg_teteje_m","reteg_alja_m","talp_m",
        "szuro_teteje_m","szuro_alja_m","szuro_hossz_m","szuro_db",
        "nyugalmi_vizszint_m","uzemi_vizszint_m","homerseklet_c",
        "hozam_lperc","enged_termeles_m3nap","lekotes_ezer_m3ev","lekotes_m3nap",
        "ev","ho","nap","vizig","objszam","letesites_eve","csoportszam"
    }
    if col in numeric_cols or col == "termeles_ezer_m3ev":
        arr = s.astype("string").map(parse_number_hu_safe).astype("float64")
        if col == "termeles_ezer_m3ev":
            non_null = arr.dropna()
            if len(non_null) >= 5 and non_null.quantile(0.90) > 10000:
                arr = arr / 1000.0  # auto-fix if someone gave m3/yr instead of k m3/yr
        if col == "lekotes_ezer_m3ev":
            non_null = arr.dropna()
            if len(non_null) and non_null.quantile(0.90) > 5000:
                arr = arr / 1000.0
        return arr
    return s.astype("string").replace({"": pd.NA})

def cast_integral_in_place(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            mask = v.notna()
            if mask.any():
                v.loc[mask] = np.round(v.loc[mask])
            df[c] = v.astype("Int64")

# --------------- Conflict selection -------------------
def almost_equal(a: float, b: float, tol: Tuple[float, float]) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
    abs_tol, rel_tol = tol
    if abs(a - b) <= abs_tol:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= rel_tol

def select_by_source_priority(cands: List[Tuple[str, str]], priority: List[str]) -> Optional[str]:
    if not cands:
        return None
    for src in priority:
        for s, v in cands:
            vv = normalize_value_str(v)
            if s == src and vv:
                return vv
    for _, v in cands:
        vv = normalize_value_str(v)
        if vv:
            return vv
    return None

def choose_numeric_safe(canonical: str, candidates: List[Tuple[str, Optional[float]]],
                        file: str, sheet: str, rid: str, conflicts: List[dict]) -> Optional[float]:
    cand = [(src, v) for src, v in candidates if v is not None]
    if not cand:
        return None
    pref = SOURCE_PREF_NUM.get(canonical, [])
    for src in pref:
        vals = [v for s, v in cand if s == src]
        if vals:
            v0 = vals[0]
            tol = NUM_TOL.get(canonical, (0.0, 0.0))
            bad = [(s, v) for s, v in cand if not almost_equal(v0, v, tol)]
            if len(cand) > 1 and bad:
                conflicts.append({
                    "file": file, "sheet": sheet, "row_index": rid, "canonical": canonical,
                    "note": "USED_PRIORITY_SOURCE",
                    "raw_columns": "; ".join([s for s,_ in cand]),
                    "values_raw": "; ".join([str(v) for _,v in cand]),
                    "reason_code": "GENERIC_NUMERIC_CONFLICT", "reason_detail": "Priority source chosen."
                })
            return float(v0)
    tol = NUM_TOL.get(canonical, (0.0, 0.0))
    base = cand[0][1]
    if all(almost_equal(base, v, tol) for _, v in cand):
        return float(np.nanmean([v for _, v in cand]))
    cand_sorted = sorted(cand, key=lambda x: x[0])
    v0 = cand_sorted[0][1]
    conflicts.append({
        "file": file, "sheet": sheet, "row_index": rid, "canonical": canonical,
        "note": "USED_ALPHABETIC_TIEBREAK",
        "raw_columns": "; ".join([s for s,_ in cand]),
        "values_raw": "; ".join([str(v) for _,v in cand]),
        "reason_code": "GENERIC_NUMERIC_CONFLICT", "reason_detail": "Alphabetic source tie-break."
    })
    return float(v0)

def choose_text_prefer_alpha(canonical: str, candidates: List[Tuple[str, str]],
                             file: str, sheet: str, rid: str, conflicts: List[dict],
                             priority: Optional[List[str]] = None) -> Optional[str]:
    cands = [(s, normalize_value_str(v)) for s, v in candidates if normalize_value_str(v)]
    if not cands:
        return None
    if priority:
        pick = select_by_source_priority(cands, priority)
        if pick:
            vals = [v for _, v in cands]
            uniq_vals = list(pd.Series(vals, dtype="string").dropna().unique())  # avoid FutureWarning
            if len(uniq_vals) > 1:
                conflicts.append({
                    "file": file, "sheet": sheet, "row_index": rid, "canonical": canonical,
                    "note": "USED_PRIORITY_SOURCE",
                    "raw_columns": "; ".join([s for s,_ in cands]),
                    "values_raw": "; ".join(uniq_vals[:10]),
                    "reason_code": "TRULY_DIFFERENT_STRINGS", "reason_detail": "Priority textual source."
                })
            return pick
    alpha = [v for _, v in cands if is_alpha_dominant(v)]
    if alpha:
        return alpha[0]
    return cands[0][1]

# -------------- Mapping: locks → heuristics → AI -------
def build_final_mapping(all_headers: List[str], long_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    heur_map: Dict[str, Optional[str]] = {h: regex_map_header(h) for h in all_headers}
    # enforce restricted
    for h, allowed in RESTRICTED_ALLOWED_CANON.items():
        if h in heur_map and allowed and heur_map[h] not in allowed:
            heur_map[h] = None

    # reuse cached mapping if present
    cached: Dict[str, Optional[str]] = {}
    if os.path.exists(OUT_MAP):
        try:
            with open(OUT_MAP, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except Exception:
            cached = {}

    for h in all_headers:
        if h in cached and cached[h] in CANONICAL_48:
            heur_map[h] = cached[h]

    hints_df = build_profile_hints_from_long(long_df)
    ai_map = ai_make_mapping_ambiguous(all_headers, hints_df, heur_map)

    final_map: Dict[str, Optional[str]] = {}
    rejected_ai = 0
    for h in all_headers:
        if h in HARD_LOCK_MAP:
            final_map[h] = HARD_LOCK_MAP[h]
            continue
        ai_c = ai_map.get(h, None)
        heur_c = heur_map.get(h, None)

        allowed = RESTRICTED_ALLOWED_CANON.get(h, None)
        if allowed is not None:
            if ai_c not in (allowed or [None]):
                ai_c = None
                rejected_ai += 1
            if heur_c not in (allowed or [None]):
                heur_c = None

        if ai_c in CANONICAL_48:
            final_map[h] = ai_c
        elif heur_c in CANONICAL_48:
            final_map[h] = heur_c
        else:
            final_map[h] = None

    if rejected_ai:
        print(f"🔒 Rejected {rejected_ai} unsafe AI suggestions (protected headers).")

    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump(final_map, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved mapping: {OUT_MAP}")

    return final_map

# --------------------------- Main ---------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_outdir(OUT_DIR)

    # 1) Load long CSV (URL or local file)
    print(f"📥 Loading long CSV from: {LONG_CSV}", flush=True)
    long_df = pd.read_csv(
        LONG_CSV,
        dtype={"file":"string","sheet":"string","method":"string","row_index":"string",
               "column_name":"string","value":"string"},
        encoding="utf-8", na_filter=False, low_memory=False
    )
    long_df["column_name"] = long_df["column_name"].map(normalize_header)

    all_headers = sorted(pd.unique(long_df["column_name"]).tolist())
    print(f"🔎 Unique raw headers detected: {len(all_headers)}", flush=True)

    # 2) Build final raw→canonical mapping
    final_map = build_final_mapping(all_headers, long_df)

    # 3) Keep only mapped rows
    long_df["canonical"] = long_df["column_name"].map(final_map).astype("string")
    kept = long_df[long_df["canonical"].notna()].copy()
    print(f"📦 Rows mapped to canonicals: {len(kept):,} / {len(long_df):,}", flush=True)

    # 4) Row-wise unification
    group_keys = ["file","sheet","row_index"]
    conflicts: List[dict] = []
    out_rows: List[dict] = []

    for (f, sh, rid), sub in kept.groupby(group_keys, sort=False):
        row: Dict[str, Optional[str]] = {"file": f, "sheet": sh, "row_index": rid}
        cands_by_canon: Dict[str, List[Tuple[str, str]]] = {cn: [] for cn in CANONICAL_48}
        for _, r in sub.iterrows():
            c = r["canonical"]
            cands_by_canon[c].append((r["column_name"], r["value"]))

        # preferred text
        row["telep_nev"]   = choose_text_prefer_alpha("telep_nev",   cands_by_canon["telep_nev"],   f, sh, rid, conflicts, SOURCE_PREF_TEXT.get("telep_nev"))
        row["telepules"]   = choose_text_prefer_alpha("telepules",   cands_by_canon["telepules"],   f, sh, rid, conflicts, SOURCE_PREF_TEXT.get("telepules"))
        row["vizikonyv_szam"] = choose_text_prefer_alpha("vizikonyv_szam", cands_by_canon["vizikonyv_szam"], f, sh, rid, conflicts, SOURCE_PREF_TEXT.get("vizikonyv_szam"))
        row["uzem_engedely_szam"] = choose_text_prefer_alpha("uzem_engedely_szam", cands_by_canon["uzem_engedely_szam"], f, sh, rid, conflicts, SOURCE_PREF_TEXT.get("uzem_engedely_szam"))
        row["engedely_szam"] = choose_text_prefer_alpha("engedely_szam", cands_by_canon["engedely_szam"], f, sh, rid, conflicts, SOURCE_PREF_TEXT.get("engedely_szam"))

        row["csoporthely"] = select_by_source_priority(cands_by_canon["csoporthely"], SOURCE_PREF_TEXT.get("csoporthely", []))
        row["objcsop_kod"] = select_by_source_priority(cands_by_canon["objcsop_kod"], SOURCE_PREF_TEXT.get("objcsop_kod", []))

        # vor_kod (never from VGT1_ObjAz)
        vor_cands = [(s, v) for s, v in cands_by_canon["vor_kod"] if s != "VGT1_ObjAz"]
        vor_codes = [normalize_value_str(v) for _, v in vor_cands if is_vor_code(v)]
        if vor_codes:
            row["vor_kod"] = vor_codes[0]
        elif vor_cands:
            row["vor_kod"] = choose_text_prefer_alpha("vor_kod", vor_cands, f, sh, rid, conflicts, SOURCE_PREF_TEXT.get("vor_kod"))
        else:
            row["vor_kod"] = None

        # objszám + obj_tipus
        obj_tp = None
        obj_num = None
        for src, val in cands_by_canon["objszam"]:
            if src.upper().startswith("OBJSZAM"):
                tp, num = split_obj_code(val)
                if num is not None:
                    obj_num = num
                if tp:
                    obj_tp = tp
                break
        if not obj_tp:
            for src, val in cands_by_canon.get("obj_tipus", []):
                v = normalize_value_str(val)
                if v:
                    obj_tp = v.lower()[0]
                    break
        if obj_num is None:
            for src, val in cands_by_canon["objszam"]:
                if src.upper() == "OBJDB":
                    num = parse_number_hu_safe(val)
                    if num is not None:
                        obj_num = num
                        break
        row["objszam"] = obj_num
        row["obj_tipus"] = obj_tp

        # numeric canonicals
        for cn in ["eov_x","eov_y","tszf_m","reteg_teteje_m","reteg_alja_m","talp_m",
                   "szuro_teteje_m","szuro_alja_m","szuro_hossz_m","szuro_db",
                   "nyugalmi_vizszint_m","uzemi_vizszint_m","homerseklet_c",
                   "hozam_lperc","termeles_ezer_m3ev","enged_termeles_m3nap",
                   "lekotes_ezer_m3ev","lekotes_m3nap","ev","ho","nap","vizig","letesites_eve","csoportszam"]:
            cands = cands_by_canon.get(cn, [])
            nums = [(src, parse_number_hu_safe(val)) for src, val in cands]
            row[cn] = choose_numeric_safe(cn, nums, f, sh, rid, conflicts)

        # dates (first valid)
        for cn in ["datum","ervenyes_tol","ervenyes_ig"]:
            vals = [normalize_value_str(v) for _, v in cands_by_canon.get(cn, []) if normalize_value_str(v)]
            if not vals:
                row[cn] = None
            else:
                parsed = parse_date_quiet(pd.Series(vals))
                row[cn] = str(parsed.dropna().iloc[0].date()) if parsed.notna().any() else None

        # remaining text
        for cn in ["vifir_kod","objhely","gw_kod","irsz","cim","helyrajzi_szam",
                   "viztipus_rt","vkj_kategoria","vizhasznalat","vizhasznalat_alkat","megjegyzes"]:
            if cn not in row:
                row[cn] = choose_text_prefer_alpha(cn, cands_by_canon.get(cn, []), f, sh, rid, conflicts,
                                                   SOURCE_PREF_TEXT.get(cn, []))

        # swap top/bottom if inverted
        for top_key, bot_key in [("reteg_teteje_m","reteg_alja_m"), ("szuro_teteje_m","szuro_alja_m")]:
            top = row.get(top_key); bot = row.get(bot_key)
            if isinstance(top, float) and isinstance(bot, float):
                if (top is not None and bot is not None and math.isfinite(top) and math.isfinite(bot)) and top > bot:
                    row[top_key], row[bot_key] = bot, top

        # derive length if missing
        sh_len, st, sa = row.get("szuro_hossz_m"), row.get("szuro_teteje_m"), row.get("szuro_alja_m")
        if (sh_len is None or (isinstance(sh_len, float) and np.isnan(sh_len))) and \
           isinstance(st, float) and isinstance(sa, float) and \
           (st is not None and sa is not None and math.isfinite(st) and math.isfinite(sa)):
            d = sa - st
            if 0 <= d <= 1000:
                row["szuro_hossz_m"] = d

        out_rows.append(row)

    # 5) DataFrame + casting
    wide = pd.DataFrame.from_records(out_rows)
    for cn in CANONICAL_48:
        if cn not in wide.columns:
            wide[cn] = pd.NA
        wide[cn] = coerce_to_canonical(cn, wide[cn])
    cast_integral_in_place(wide, ["objszam","ev","ho","nap","vizig","szuro_db","csoportszam","letesites_eve"])

    cols = ["file","sheet","row_index"] + CANONICAL_48
    wide = wide[cols]

    # 6) Persist
    if conflicts:
        pd.DataFrame(conflicts).to_csv(OUT_CONFLICTS, index=False, encoding="utf-8")
        print(f"⚠️  Wrote conflicts: {OUT_CONFLICTS}  ({len(conflicts):,} rows)", flush=True)
    else:
        print("✅ No conflicts recorded (after safe resolution).", flush=True)

    wide.to_csv(OUT_UNIFIED, index=False, encoding="utf-8")
    print(f"✅ Wrote unified table: {OUT_UNIFIED}  ({len(wide):,} rows, {len(wide.columns)} cols)", flush=True)

    # 7) Short AI validation note (optional)
    try:
        if not OPENAI_KEY or AI_MODE.lower() == "disabled":
            raise RuntimeError("AI disabled or no OPENAI_API_KEY")
        _init_openai_clients()
        if not (_client_new or _client_old):
            raise RuntimeError("OpenAI SDK unavailable")

        prof2 = []
        for cn in CANONICAL_48:
            ser = wide[cn]
            notna = int(ser.notna().sum())
            uniq = int(pd.Series(ser.dropna().astype(str)).nunique())
            sample = "; ".join([str(x) for x in ser.dropna().astype(str).head(5).tolist()])
            prof2.append({"name": cn, "non_null": notna, "unique": uniq, "sample": sample})
        msg = {
            "summary": {"rows": len(wide), "conflicts": len(conflicts)},
            "columns": prof2,
            "policy": "Priority text sources; OBJSZAM parsed (K-28→28); vor_kod pattern-checked; top/bottom swap; screen length from bounds."
        }
        messages = [
            {"role":"system","content":"Audit the 48-column unification. Produce a short markdown note with coverage & caveats."},
            {"role":"user","content": json.dumps(msg, ensure_ascii=False)}
        ]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_ai_call_messages, messages, False)
            try:
                ai_text = future.result(timeout=max(6, AI_TIMEOUT_SEC//2))
            except Exception:
                ai_text = "_AI validation timed out or unavailable._"
        with open(OUT_AI_MD, "w", encoding="utf-8") as f:
            f.write(ai_text)
        print(f"✅ Wrote AI validation notes: {OUT_AI_MD}", flush=True)
    except Exception as e:
        with open(OUT_AI_MD, "w", encoding="utf-8") as f:
            f.write(f"_AI validation unavailable: {e}_")
        print(f"⚠️  AI validation unavailable: {e}", flush=True)

if __name__ == "__main__":
    main()
