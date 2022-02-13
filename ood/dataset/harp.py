import numpy as np

import nibabel as nib
from dpipe.dataset.segmentation import MultichannelSegmentationFromCSV


class HarP(MultichannelSegmentationFromCSV):
    def __init__(self, data_path, modalities=('MRI', ), targets=('hippo_mask_L', 'hippo_mask_R'), 
                 metadata_rpath='meta.csv', index_col='id'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         targets=targets,
                         metadata_rpath=metadata_rpath,
                         index_col=index_col)

    def load_image(self, i):
        image = nib.load(self.df['MRI'].loc[i]).get_fdata()
        image = np.swapaxes(image, 0, 2) # reshape image
        return np.float32(image)

    def load_segm(self, i):
        masks = super().load_segm(i)
        return np.float32(np.logical_or(masks[0] > 0.5, masks[1] > 0.5))  # left and right hippocampus

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing