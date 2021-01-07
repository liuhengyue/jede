from detectron2.structures.boxes import *

class DigitBoxes(Boxes):
    """
    Here, the digit boxes for each instance is in shape of
    (1, M, 4) or (B, N, M, 4). So for each person, it has M digits.
    """
    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a B x N x 2 x 4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 0, 4, dtype=torch.float32, device=device)
        assert tensor.dim() > 2 and tensor.dim() < 5 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def to(self, device: str) -> "Boxes":
        return DigitBoxes(self.tensor.to(device))

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[..., 0::2] *= scale_x
        self.tensor[..., 1::2] *= scale_y

    def clip(self, box_size: BoxSizeType) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        self.tensor[..., 0].clamp_(min=0, max=w)
        self.tensor[..., 1].clamp_(min=0, max=h)
        self.tensor[..., 2].clamp_(min=0, max=w)
        self.tensor[..., 3].clamp_(min=0, max=h)

    def flat(self) -> "Boxes":
        """
        Reshape DigitBoxes has shape (N,2,4) to Boxes of shape (2*N, 4)
        """
        return Boxes(self.tensor.reshape((-1, 4)))

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[..., 2] - box[..., 0]
        heights = box[..., 3] - box[..., 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
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
            return DigitBoxes(self.tensor[item].view(1, -1, 4))
        b = self.tensor[item]
        assert b.dim() > 2 and b.dim() < 5, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return DigitBoxes(b)