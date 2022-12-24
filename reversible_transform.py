import cv2
import numpy as np


class Reversible_Transform:
    def __init__(self, config):
        self.op_list=[]
