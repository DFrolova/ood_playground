from .utils import choose_root


# ################################################ DATA PATHS ################################################


CC359_DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/cc359',
    '/gpfs/data/gpfs0/b.shirokikh/data/cc359',
    '/shared/data/cc359',
    '/',  # TODO: avoiding `FileNotFoundError`
)

LIDC_SUPPLEMENTARY_PATH = '/data/lidc_supplementary/'

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
