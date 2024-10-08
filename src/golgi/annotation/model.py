from abc import ABC
import abc

import numpy as np

class Model(ABC):
    
    @abc.abstractmethod
    def infer(self, img):
        ...

class LocalModel(Model):

    def __init__(self, model):
        self.model = model

    def infer(self, img):
        results = self.model(img, max_det=1, verbose=False)

        mask = []
        
        if not results[0].masks:
            return None

        for r in results:
            for x, y in r.masks.xy[0]:
                mask.append(int(x))
                mask.append(int(y))

        ctr = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)

        return ctr
