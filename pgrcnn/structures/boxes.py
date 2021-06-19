import torch
from detectron2.structures.boxes import Boxes as detectron_boxes

class Boxes(detectron_boxes):
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)



    def get_scales(self):
        widths = self.tensor[:, 2] - self.tensor[:, 0]
        heights = self.tensor[:, 3] - self.tensor[:, 1]
        return torch.stack((widths, heights), dim=1)

    def remove_duplicates(self, labels):
        # remove duplicate boxes associated with labels, and return the filtered labels
        labels = torch.cat((self.tensor, labels[...,None]), dim=-1)
        output = torch.unique(labels, dim=0)
        self.tensor = output[..., :4]
        return output[..., 4].long()

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)