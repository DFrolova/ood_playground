from .utils import choose_root

# ################################################ DATA PATHS ################################################


CC359_DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/cc359',
    '/gpfs/data/gpfs0/b.shirokikh/data/cc359',
    '/shared/data/cc359',
    '/',  # TODO: avoiding `FileNotFoundError`
)

LIDC_DATA_PATH = None
LITS_DATA_PATH = None

WMH_DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/wmh_ants',
    '/gpfs/data/gpfs0/b.shirokikh/data/wmh_ants',
    '/',  # TODO: avoiding `FileNotFoundError`
)

GAMMA_KNIFE_MET_T1_T1C_PATH = choose_root(
    '/nmnt/x4-hdd/experiments/gamma-knife-met-t1-t1c',
    '/shared/data/gamma-knife-met-t1-t1c/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

GAMMA_KNIFE_BRAIN_PATH = choose_root(
    '/shared/data/gamma-knife-brain/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

TASK4_HIPPO_PATH = choose_root(
    '/shared/data/Task04_Hippocampus/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

HEART_DATA_PATH = choose_root(
    '/shared/data/Task02_Heart/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

HARP_DATA_PATH = choose_root(
    '/shared/data/HarP/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

LUNGS_BBOXES_PATH = choose_root(
    '/shared/experiments/ood_playground/bounding_boxes/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

ENSEMBLE_MODELS_PATH_CC359 = choose_root(
    '/shared/experiments/ood_playground/cc359/brain_segm/cc359/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

ENSEMBLE_MODELS_PATH_LIDC = choose_root(
    '/shared/experiments/ood_playground/luna_no_crop/lidc/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

ENSEMBLE_MODELS_PATH_MIDRC = choose_root(
    '/shared/experiments/ood_playground/midrc_no_crop/midrc/',
    '/',  # TODO: avoiding `FileNotFoundError`
)
