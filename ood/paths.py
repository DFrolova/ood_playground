from .utils import choose_root

# ################################################ DATA PATHS ################################################

LIDC_DATA_PATH = None
LITS_DATA_PATH = None
MEDSEG9_DATA_PATH = None
MIDRC_DATA_PATH = None
CANCER500_DATA_PATH = None
CT_ICH_DATA_PATH = None
VSSEG_DATA_PATH = '/shared/data/VS-SEG/'
CC359_DATA_PATH = None
CROSSMODA_DATA_PATH = None
NSCLC_DATA_PATH = None
EGD_DATA_PATH = '/shared/data/EGD'

# LIDC_AUGM_DATA_PATH = '/shared/experiments/ood_playground/lidc_augm_data'
# VSSEG_AUGM_DATA_PATH = '/shared/experiments/ood_playground/vsseg_augm_data'

# LIDC_ARTIFACTS_DATA_PATH = '/shared/experiments/ood_playground/artifacts_lidc'

# svd benchmark
LIVER_DATASET_DATA_PATH = '/shared/experiments/ood_playground/svd_paper/liver_data/'
MSD_DATASET_DATA_PATH = '/shared/experiments/ood_playground/svd_paper/MSD/'

# MOOD
MOOD_DATA_PATH = '/shared/data/MOOD/'

ENSEMBLE_MODELS_PATH_LIDC = choose_root(
    '/shared/experiments/ood_playground/lidc/lidc/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

ENSEMBLE_MODELS_PATH_VSSEG = choose_root(
    '/shared/experiments/ood_playground/vsseg/vsseg/',
    '/',  # TODO: avoiding `FileNotFoundError`
)
