import numpy as np
import pandas as pd

from data_loaders.wMIB.pymo.pymo.parsers import BVHParser


class BVHConverter:
    def __init__(self, path):
        self.parser = BVHParser()

        self.parser.parse(path)
        self.data = self.parser.data
        self.nframe = self.parser.data.values.shape[0]
        self.skeleton = self.parser._skeleton

    def get_frames(self, start, end):
        """
            return mocap data between frame[start] and frame[end]
        """
        if start >= 0 and end < self.nframe:
            return self.data.values[start: end]

    def get_skeleton(self):
        return self.skeleton


def main():
    bvh_path = '../bvh/test2/Shanghai_SOP_20230308_SOP/c1_5_001.bvh'
    data = BVHConverter(bvh_path)

    # print(data.get_frames(1, 10))
    print(data.data.channel_names)
    print(len(data.data.channel_names))

if __name__ == "__main__":
    main()