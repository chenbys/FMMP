import cv2
import numpy as np
from mmpose.datasets.builder import PIPELINES
import kornia.augmentation as KA


@PIPELINES.register_module()
class ColorAug:
    def __init__(self, roughness=(0.1, 0.7), intensity=(0.0, 0.7), jiggle=(0.1, 0.1, 0.1, 0.1),
                 pb=1.0, pc=1.0, pj=1.0):
        self.roughness = roughness
        self.intensity = intensity
        self.jiggle = jiggle
        self.pb = pb
        self.pc = pc
        self.pj = pj

    def __call__(self, results):
        img_aug = results['img'][None]
        # img_aug = KA.RandomPlasmaBrightness(roughness=self.roughness, intensity=self.intensity, p=self.pb)(img_aug)
        # img_aug = KA.RandomPlasmaContrast(roughness=self.roughness, p=self.pc)(img_aug)
        img_aug = KA.ColorJiggle(self.jiggle[0], self.jiggle[1], self.jiggle[2], self.jiggle[3], p=self.pj)(img_aug)
        results['img'] = img_aug[0]
        return results
