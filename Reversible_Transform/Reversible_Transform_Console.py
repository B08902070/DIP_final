from .Reversible_Rotate import Reversible_Rotate
from .Reversible_Resize import Reversible_Resize
from .Reversible_Swirl import Reversible_Swirl
from .Reversible_Wavy import Reversible_Wavy

class Reversible_Type:
    Rotate = Reversible_Rotate
    Swirl = Reversible_Swirl
    Resize = Reversible_Resize
    Wavy = Reversible_Wavy


class Reversible_Transform_Console:
    def __init__(self):
        self.transform_fns = []

    def load_transform_ops(self, transform_fn_name, transform_kwargs):
        transform_fn = getattr(Reversible_Type, transform_fn_name)
        self.transform_fns.append(transform_fn(transform_kwargs))

    def forward(self, image):
        for fn in self.transform_fns:
            image = fn.forward(image)
        return image

    def backward(self, image):
        N = len(self.transform_fns)
        for i in range(1, N+1):
            image = self.transform_fns[N-i].backward(image)
        return image


    
