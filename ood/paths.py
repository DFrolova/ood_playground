from .utils import choose_root

# ################################################ DATA PATHS ################################################

LIDC_DATA_PATH = None
LITS_DATA_PATH = None
MEDSEG9_DATA_PATH = None
CANCER500_DATA_PATH = None

ENSEMBLE_MODELS_PATH_LIDC = choose_root(
    '/shared/experiments/ood_playground/lidc/lidc/',
    '/',  # TODO: avoiding `FileNotFoundError`
)
