# a modification of 'MapDataset' in detectron2.data.common
import copy
import itertools
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied. A list of dicts.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func, copy_paste_mix=0):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))
        self._copy_paste_mix = copy_paste_mix

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)
        while True:
            if self._copy_paste_mix:
                # generate a list of random ids to load multiple images
                extra_inds = np.random.randint(np.random.randint(self.__len__(), size=1),
                                               size=np.random.randint(1, self._copy_paste_mix + 1, size=1))
                data = self._map_func([self._dataset[cur_idx]] + [self._dataset[i] for i in extra_inds])
            else:
                data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )