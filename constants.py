import os
import re
from collections import OrderedDict
from pathlib import Path

from join_ae_recall.constants import AE_UID_KEY
from data_pull_pipeline.config import ML_DATA_ROOT  # single source for ML data root

ALT_DATA_ROOT= "./"
ML_DATA_ROOT: Path = Path(os.getenv("ML_DATA_ROOT", ALT_DATA_ROOT / "ml_data")).expanduser().resolve()


# Canonical device identifier in joined datasets
DEVICE_ID_COLUMN = "serona_device_id"

# ───────────────────────────────────────────────
# 1.  CONFIG & LOGGING
# ───────────────────────────────────────────────
COUNT_COLS = [
    # "ae_count_prior",
    "ae_count_30d",
    "ae_count_60d",
    "ae_count_90d",
    "ae_count_180d",
]
HIST_COLS = ["nlp_hist", "dev_hist", "cat_hist"]

# Use pipeline-configured ML root (external SSD). Can still be overridden via env.
_THIS_DIR = Path(__file__).parent
DATA_ROOT = Path(os.environ.get("SERONA_DATA_ROOT", ML_DATA_ROOT)).expanduser().resolve()
CACHE_DIR = Path(os.environ.get("SERONA_CACHE_DIR", DATA_ROOT / "ml_training" / "cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LMDB_DIR = Path(os.environ.get("SERONA_LMDB_DIR", CACHE_DIR / "lmdb")).resolve()
LMDB_DIR.mkdir(parents=True, exist_ok=True)
LMDB_MAP_SIZE = int(
    os.environ.get("SERONA_LMDB_MAP_SIZE_BYTES", 2**39)
)  # default 512 GiB
LMDB_META_PREFIX = "__meta__"
LMDB_FEATURE_SETS: "OrderedDict[str, dict]" = OrderedDict(
    [
        ("counts", {"columns": tuple(COUNT_COLS)}),
        ("nlp_hist", {"column": "nlp_hist"}),
        ("dev_hist", {"column": "dev_hist"}),
        ("cat_hist", {"column": "cat_hist"}),
    ]
)

# Training/test split years for evaluation.
TRAIN_YEARS = [2023, 2024]
TEST_YEARS = [2025]

_DEFAULT_JOINED_DATA_ROOT = (
    Path(__file__).parent.parent / "join_ae_recall" / "250526_ae_recall_labelled_output"
)
_MONTH_PARTITION_RE = re.compile(r"^month=(\d{4})-\d{2}$")
_CACHE_PARQUET_RE = re.compile(r"^final_dataset_proc_(\d{4})$")
_LMDB_RE = re.compile(r"^hist_features_(\d{4})$")


def _parse_years_env(raw: str | None) -> list[int]:
    if not raw:
        return []
    # Accept comma/space-separated lists (e.g. "2023,2024,2025").
    tokens = re.split(r"[,\s]+", raw.strip())
    years = []
    for t in tokens:
        if not t:
            continue
        if t.isdigit() and len(t) == 4:
            years.append(int(t))
    return sorted(set(years))


def get_cache_years(cache_dir: Path | None = None, lmdb_dir: Path | None = None) -> list[int]:
    cache_dir = cache_dir or CACHE_DIR
    lmdb_dir = lmdb_dir or LMDB_DIR
    years: set[int] = set()

    if cache_dir.exists():
        for p in cache_dir.glob("final_dataset_proc_*.parquet"):
            m = _CACHE_PARQUET_RE.match(p.stem)
            if m:
                years.add(int(m.group(1)))

    if lmdb_dir.exists():
        for p in lmdb_dir.glob("hist_features_*"):
            m = _LMDB_RE.match(p.name)
            if m:
                years.add(int(m.group(1)))

    return sorted(years)


def get_joined_data_years(data_root: Path) -> list[int]:
    if not data_root.exists():
        return []
    years: set[int] = set()
    for p in data_root.glob("month=????-??"):
        m = _MONTH_PARTITION_RE.match(p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def get_available_years(joined_data_root: Path | None = None) -> list[int]:
    env_years = _parse_years_env(os.environ.get("SERONA_DATA_YEARS"))
    if env_years:
        return env_years

    cache_years = get_cache_years()
    if cache_years:
        return cache_years

    if joined_data_root is not None:
        joined_years = get_joined_data_years(joined_data_root)
        if joined_years:
            return joined_years

    # Fall back to the evaluation split years.
    return sorted(set(TRAIN_YEARS + TEST_YEARS))


# All available data years (auto-discovered). Used for data prep, production training, and inference.
YEARS = get_available_years(joined_data_root=_DEFAULT_JOINED_DATA_ROOT)

# PRODUCTION MODE: Train on ALL data for maximum performance
# Note: For production inference (step 11), the last 120 days are held out
# based on unified_ae_date, not by year

# Inference holdout configuration (in days)
INFERENCE_HOLDOUT_DAYS = 120


DEVICE_GROUP_COLUMN = "device_openfda_device_name"

TIME_FEAT_METADATA_COLUMNS = [
    AE_UID_KEY,
    DEVICE_ID_COLUMN,
    "unified_ae_date",
    "days_to_recall",
    DEVICE_GROUP_COLUMN,
]

FEATURE_COLS = ["days_to_recall", "unified_ae_date", AE_UID_KEY, DEVICE_GROUP_COLUMN]

LOG_DIR = Path(os.environ.get("SERONA_LOG_DIR", DATA_ROOT / "ml_training" / "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Shared analysis/output directory for ML training artifacts
ANALYSIS_DIR = Path(
    os.environ.get("SERONA_ANALYSIS_DIR", DATA_ROOT / "ml_training" / "feature_analysis")
).resolve()
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


DEVICE_NAME_FEATURES = [
    "device_brand_name",
    "device_generic_name",
    "device_manufacturer_d_name",
    "event_type",
    "device_device_operator",
]
OTHER_NLP = [
    "product_problems",
    "patient_patient_problems",
    #"patient_sequence_number_treatment", -> leakage, need to treat carefully
    # "patient_sequence_number_outcome", #i think this was causing leakage
    # "mdr_text_text",-> leakage, need to treat carefully
    "source_type",

]
CATEGORICAL_RAW = [
    "device_openfda_device_class",
    "device_openfda_medical_specialty_description",
    # "device_openfda_regulation_number", -> revisit adding this later, there is a structure here that isn't categorical
    "health_professional",
    "device_device_availability",
    # "device_implant_flag",
    # "patient_patient_ethnicity",
   # "patient_patient_race",
    # "mdr_text_text_type_code", -> leakage, need to treat carefully
    # "type_of_report",
    "report_source_code",
    "product_problem_flag",
    "adverse_event_flag",
    # "combination_product_flag", # want to include this, but not available in all years
    "reprocessed_and_reused_flag",
    "single_use_flag",
    # "manufacturer_state" #wrong field name
]
DERIVED_CATEGORICAL = [
    "has_removal_correction_number",
]
CATEGORICAL = CATEGORICAL_RAW + DERIVED_CATEGORICAL
NUMERIC = [
    "patient_patient_age",
    "patient_patient_sex",
    "patient_patient_weight",
    "number_devices_in_event",
    "number_patients_in_event",
]
ESSENTIAL_COLS = [
    "unified_ae_date",
    "unified_recall_date",
    DEVICE_ID_COLUMN,
    'merge_confidence',
    'NON_MERGE_REASON',
] + DEVICE_NAME_FEATURES + OTHER_NLP + CATEGORICAL_RAW + NUMERIC + ["days_since_last_recall", "removal_correction_number", AE_UID_KEY, DEVICE_GROUP_COLUMN]
#TODO add lag features like (date_manufacturer_received - date_of_event)
FEATURISER_PATH = CACHE_DIR / "featurisers.pkl"

# ───────────────────────────────────────────────
# 2.  TRAINING FEATURE SETS
# ───────────────────────────────────────────────
# Primary feature set used by `ml_training/train_simple_ml.py`
FEATURES_TO_TRAIN_ON: list[str] = [
    "adverse_event_flag_max",
    "adverse_event_flag_min",
    "adverse_event_flag_mode",
    "adverse_event_flag_unique",
    "ae_count_180d",
    "ae_count_30d",
    "ae_count_60d",
    "ae_count_90d",
    "ae_count_prior",
    "days_since_last_recall",
    "device_brand_name_embedding_m3",
    "device_brand_name_embedding_mean",
    "device_brand_name_embedding_var",
    "device_device_operator_embedding_m3",
    "device_device_operator_embedding_mean",
    "device_device_operator_embedding_var",
    "device_generic_name_embedding_m3",
    "device_generic_name_embedding_mean",
    "device_generic_name_embedding_var",
    "device_manufacturer_d_name_embedding_m3",
    "device_manufacturer_d_name_embedding_mean",
    "device_manufacturer_d_name_embedding_var",
    "device_openfda_device_class_max",
    "device_openfda_device_class_min",
    "device_openfda_device_class_mode",
    "device_openfda_device_class_unique",
    "device_openfda_medical_specialty_description_max",
    "device_openfda_medical_specialty_description_min",
    "device_openfda_medical_specialty_description_mode",
    "device_openfda_medical_specialty_description_unique",
    "event_type_embedding_m3",
    "event_type_embedding_mean",
    "event_type_embedding_var",
    "health_professional_max",
    "health_professional_min",
    "health_professional_mode",
    "health_professional_unique",
    "has_removal_correction_number_max",
    "has_removal_correction_number_min",
    "has_removal_correction_number_mode",
    "has_removal_correction_number_unique",
    "device_device_availability_max",
    "device_device_availability_min",
    "device_device_availability_mode",
    "device_device_availability_unique",
    "patient_patient_problems_embedding_m3",
    "patient_patient_problems_embedding_mean",
    "patient_patient_problems_embedding_var",
    "product_problem_flag_max",
    "product_problem_flag_min",
    "product_problem_flag_mode",
    "product_problem_flag_unique",
    "product_problems_embedding_m3",
    "product_problems_embedding_mean",
    "product_problems_embedding_var",
    "report_source_code_max",
    "report_source_code_min",
    "report_source_code_mode",
    "report_source_code_unique",
    "reprocessed_and_reused_flag_max",
    "reprocessed_and_reused_flag_min",
    "reprocessed_and_reused_flag_mode",
    "reprocessed_and_reused_flag_unique",
    "single_use_flag_max",
    "single_use_flag_min",
    "single_use_flag_mode",
    "single_use_flag_unique",
    "source_type_embedding_m3",
    "source_type_embedding_mean",
    "source_type_embedding_var",
]

# Features found empirically to be non-useful; used for optional filtering.
EMPIRICALLY_NON_USEFUL_FEATURES: list[str] = [
    "device_manufacturer_d_name_embedding_var",
    "device_openfda_medical_specialty_description_max",
    "device_device_operator_embedding_var",
    "health_professional_min",
    "single_use_flag_unique",
    "report_source_code_min",
    "single_use_flag_max",
    "adverse_event_flag_max",
    "adverse_event_flag_min",
    "adverse_event_flag_mode",
    "adverse_event_flag_unique",
    "days_since_last_recall",
    "device_openfda_device_class_unique",
    "device_openfda_medical_specialty_description_unique",
    "health_professional_max",
    "health_professional_mode",
    "health_professional_unique",
    "product_problem_flag_max",
    "product_problem_flag_min",
    "product_problem_flag_mode",
    "product_problem_flag_unique",
    "report_source_code_max",
    "report_source_code_mode",
    "report_source_code_unique",
    "reprocessed_and_reused_flag_max",
    "reprocessed_and_reused_flag_min",
    "reprocessed_and_reused_flag_mode",
    "reprocessed_and_reused_flag_unique",
    "single_use_flag_min",
    "single_use_flag_mode",
]
