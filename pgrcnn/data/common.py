# a modification of 'MapDataset' in detectron2.data.common
import logging
from typing import Dict, Tuple, List
import random
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

class MapAugDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied. A list of dicts.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self,
                 dataset,
                 map_func,
                 copy_paste_mix=0,
                 applicable_dict: Dict[int, Tuple[int, int]] = None,
                 applicable_inds: Dict[int, List[int]] = None
                 ):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))
        self._copy_paste_mix = copy_paste_mix
        self._applicable_dict = applicable_dict
        self._applicable_inds = applicable_inds

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)
        while True:
            applicable, dataset_idx = self._applicable_dict[cur_idx]
            if self._copy_paste_mix and applicable:
                # generate a list of random ids to load multiple images
                mix_size = self._rng.randint(0, self._copy_paste_mix)
                # with replacement (good for when dataset smaller than mix_size)
                extra_inds = self._rng.choices(self._applicable_inds[dataset_idx], k=mix_size)
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