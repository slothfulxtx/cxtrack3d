import torch
from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self, split_type, cfg, log):
        super().__init__()
        assert split_type in ['train', 'val', 'test']
        self.split_type = split_type
        self.tracklet_annotations = []
        self.tracklet_st_frame_id = []
        self.tracklet_ed_frame_id = []
        self.cfg = cfg
        self.log = log

    @abstractmethod
    def get_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def num_tracklets(self):
        raise NotImplementedError

    @abstractmethod
    def num_frames(self):
        raise NotImplementedError

    @abstractmethod
    def num_tracklet_frames(self, tracklet_id):
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, tracklet_id, frame_id):
        raise NotImplementedError

    @abstractmethod
    def get_comp_template_pcd(self, tracklet_id):
        raise NotImplementedError

    @abstractmethod
    def get_tracklet_frame_id(self, idx):
        raise NotImplementedError


class EvalDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        return self.dataset.num_tracklets()

    def __getitem__(self, idx):
        tracklet_anno = self.dataset.tracklet_annotations[idx]
        frames = []
        for i in range(len(tracklet_anno)):
            frames.append(self.dataset.get_frame(idx, i))

        return frames
