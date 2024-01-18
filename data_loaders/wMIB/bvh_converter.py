import numpy as np
import pandas as pd

from data_loaders.wMIB.pymo.pymo.parsers import BVHParser


class BVHConverter:
    def __init__(self, path):
        self.parser = BVHParser()
        # print('Reading data from: ', path)
        self.parser.parse(path)
        self.data = self.parser.data
        self.nframe = self.parser.data.values.shape[0]
        self.skeleton = self.parser._skeleton

    def get_frames(self, start, end):
        """
            return mocap data between frame[start] and frame[end]
        """
        if start >= 0 and end <= self.nframe:
            return self.data.values[start: end]
        elif end > self.nframe and start < self.nframe:
            # 有的标注帧数大于实际帧数，来骗来偷袭我69岁的老同志
            return self.data.values[start: self.nframe]

    def get_skeleton(self):
        return self.skeleton


def main():
    bvh_path = './bvh/Shanghai/SOP/20230524_SOP/input/120347224712913/Male_Basic01_ROM.bvh'
    data = BVHConverter(bvh_path)

    print(data.get_frames(1, 10))
    print(data.data.channel_names)
    print(len(data.data.channel_names))

if __name__ == "__main__":
    main()