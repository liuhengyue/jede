import itertools
from typing import Any, Dict, List, Tuple, Union
import torch
from detectron2.structures.instances import Instances
from itertools import compress

class CustomizedInstances(Instances):
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/Get a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``,

    Modified for taking list as fields and support list indexing
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        super().__init__(image_size, **kwargs)

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields) and 'digit' not in name:
            # we allow the instance to have different length
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    # Tensor-like methods
    # add support for list of tensors
    def to(self, device: str) -> "CustomizedInstances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = CustomizedInstances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            if type(v) == list: # and it is a list of tensors
                v = [x.to(device) for x in v]
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "CustomizedInstances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = CustomizedInstances(self._image_size)
        for k, v in self._fields.items():
            # item can be mask or index
            if type(v) == list:
                if item.dtype == torch.bool: # bool tensor
                    ret.set(k, list(compress(v, item.tolist())))
                elif item.dtype == torch.Tensor: # index tensor
                    ret.set(k, [v[idx] for idx in item.tolist()])
                else: # int index
                    ret.set(k, v[item])

            else:
                ret.set(k, v[item])
        return ret

    @staticmethod
    def cat(instance_lists: List["CustomizedInstances"]) -> "CustomizedInstances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
            extended cat for other type like boxes and keypoints
        """
        assert all(isinstance(i, CustomizedInstances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = CustomizedInstances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            elif type(v0).__name__ == "Keypoints":
                values = type(v0)(torch.cat([v.tensor for v in values], dim=0))
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret