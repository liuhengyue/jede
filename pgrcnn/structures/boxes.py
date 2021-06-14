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