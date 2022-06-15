import numpy as np

from dpipe.dataset.segmentation import MultichannelSegmentationFromCSV


class CC359(MultichannelSegmentationFromCSV):
    def __init__(self, data_path, modalities=('MRI',), target='brain_mask', metadata_rpath='meta.csv', index_col='id'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         targets=(target,),
                         metadata_rpath=metadata_rpath,
                         index_col=index_col)
        self.n_domains = len(self.df['fold'].unique())

    def load_image(self, i):
        return np.float32(super().load_image(i)[0])  # 4D -> 3D

    def load_segm(self, i):
        return np.float32(super().load_segm(i)[0])  # 4D -> 3D

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing

    def load_domain_label(self, i):
        domain_id = self.df['fold'].loc[i]
        return np.eye(self.n_domains)[domain_id]  # one-hot-encoded domain

    def load_domain_label_number(self, i):
        return self.df['fold'].loc[i]
