import numpy as np

from dpipe.dataset.segmentation import MultichannelSegmentationFromCSV


class Heart(MultichannelSegmentationFromCSV):
    def __init__(self, data_path, modalities=('MRI', ), target='mask', metadata_rpath='meta_heart.csv', index_col='id'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         targets=(target, ),
                         metadata_rpath=metadata_rpath,
                         index_col=index_col)

    def load_image(self, i):
        return np.float32(super().load_image(i)[0])  # 4D -> 3D

    def load_segm(self, i):
        return np.float32(super().load_segm(i)[0])  # 4D -> 3D

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing