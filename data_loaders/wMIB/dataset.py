"""
    This file is used for extracting motion and its annotations from given annotation file.

    Including:
        class Action: get start time, end time, and other attributes of an action
        class MotionAnno: one line in annotation file, get motion from a bvh,
                          and extract its annotation from .json anno file
        class ActionLoader: preprocess whole annotation file

"""
import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from os.path import join as pjoin

import numpy as np
import pandas as pd
import random

from pathlib import Path
from data_loaders.wMIB.bvh_converter import BVHConverter
from utils.rotation_conversions import euler_angles_to_matrix, matrix_to_rotation_6d
from utils.rotation_conversions import matrix_to_euler_angles
from data_loaders.wMIB.nlp_consts import fix_spell
import spacy
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

nlp = spacy.load('en_core_web_sm')


# Tokenizer according to https://github.com/EricGuo5513/HumanML3D/blob/main/text_process.py
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


class Action:
    def __init__(self, prev_action_anno, cur_action_anno):
        if prev_action_anno is None:
            self.start_frame = 0
        else:
            self.start_frame = int(prev_action_anno["tag"][0])
        self.end_frame = int(cur_action_anno["tag"][0])
        self.length = self.end_frame - self.start_frame
        self.action = tuple(cur_action_anno["tag"][1].split("+"))
        self.caption = random.choice(self.action)
        self.speed = cur_action_anno["tag"][2]
        self.orientation = cur_action_anno["tag"][3]
        self.turning = cur_action_anno["tag"][4]

        # preprocess description:
        # After this, description will look like ['stoop', 'pick up', '中', '右前', '无转向'],
        # where the first and second one stand for action, other words stand for action labels
        self.description = []
        self.word_enc = []
        for word in self.action:
            self.description.append(fix_spell(word))
        self.description.append(fix_spell(self.speed))
        self.description.append(fix_spell(self.orientation))
        self.description.append(fix_spell(self.turning))

        def get_tokens(caption):
            word_list, pose_list = process_text(caption)
            return ['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))]

        self.tokens_data = []
        for k, text in enumerate(self.description):
            self.tokens_data.extend(get_tokens(text))

        # print(self.tokens_data)


class MotionAnno:
    def __init__(self, anno):
        self.anno = anno

        name = anno['datasetItemContent']['file_name']
        path = anno['datasetItemContent']['file_path']
        self.id = anno['detailId']

        self.file_path = Path(path.replace('mp4_2x', 'bvh')) / Path(name).with_suffix('')

        self.objects = anno["detailLabel"]["objects"]
        self.tags = anno["detailLabel"]["tags"]

        self.type, self.speed, self.orientation, self.trans_orientation, self.abnormal, self.cut_anno = \
            anno["detailLabel"]["tags"][0], anno["detailLabel"]["tags"][1], anno["detailLabel"]["tags"][2], \
                anno["detailLabel"]["tags"][3], anno["detailLabel"]["tags"][4], anno["detailLabel"]["tags"][5]

        if len(self.abnormal["value"]):
            raise Exception
        self.total_length = int(self.cut_anno["value"][0]["tag"][0])

        """
            这个数据集的标注是从最后一个动作开始的，并且只给了每个动作的结束帧，因此需要特别处理一下
            calc_anno_statistics.py 里搞反了
        """
        self.action_list = []
        for i in range(len(self.cut_anno["value"])):
            if i == len(self.cut_anno["value"]) - 1:
                self.action_list.append(Action(None, self.cut_anno["value"][i]))
            else:
                self.action_list.append(Action(self.cut_anno["value"][i + 1], self.cut_anno["value"][i]))


def movement_to_rot6d(motion_data):
    """
        该函数的过程和 movement_save_init() 一致，注释将被放在下一函数中
        Transform Euler angle in motion data into rot6d representation
        bvh motion structure: [root position x/y/z, root rotation x/y/z, child rotation x/y/z * n]

        Also, we do a preprocessing progress in this step,
        where the initial position of each movement will be placed at (0, 0, 0) and
        all motion will face x- direction initially.

        Param:
            motion_data: dataframe, shape [frame, root_xyz + rotation * 3 * joints]

        Return:
            motion_data_rot6d: [frame, root_xyz + rotation * 6 * joints] tensor.
    """
    motions = np.array(motion_data)
    motion_rot6d = np.empty((motions.shape[0], 0))
    num_cols = motions.shape[1]

    # Path
    # Attention: xyz in bvh -> xzy in houdini (maybe, but the order is not the same)
    init_euler = euler_angles_to_matrix(torch.from_numpy(motions[0, 3: 6] * 3.1415926 / 180.), 'XYZ')
    rot_vec = torch.matmul(init_euler, torch.from_numpy(np.array([0., 0., 1.])).reshape((3, 1)))
    rot_e = -np.arccos(rot_vec[0] / np.sqrt(rot_vec[0] ** 2 + rot_vec[2] ** 2))
    rot_e = float(rot_e)

    joint_rot = np.array([
        np.cos(rot_e), 0, np.sin(rot_e),
        0, 1, 0,
        -np.sin(rot_e), 0, np.cos(rot_e)
    ])

    rot_mat = torch.from_numpy(joint_rot).reshape((3, 3))
    path_rotation_mat = torch.matmul(rot_mat, torch.from_numpy(motions[:, 0: 3]).transpose(1, 0))
    path_rotation_mat = path_rotation_mat.transpose(1, 0)
    # print(path_rotation_mat - motions[:, 0:3])

    new_init_pos = path_rotation_mat.numpy()[0, :]
    new_root_pos = path_rotation_mat.numpy() - new_init_pos

    # Euler Angle
    rot_mat = torch.from_numpy(joint_rot).reshape((3, 3))
    joint_euler_mat = euler_angles_to_matrix(torch.from_numpy(motions[:, 3: 6] * 3.1415926 / 180.), 'XYZ')
    joint_rot_mat = torch.matmul(rot_mat.reshape(1, 3, 3), joint_euler_mat)
    new_euler = matrix_to_euler_angles(joint_rot_mat, 'XYZ')
    # print(new_euler)
    for i in range(0, num_cols, 3):
        sub_arr = motions[:, i: i + 3]
        if i == 0:
            # initial motion will be moved to (0, 0, 0)
            motion_rot6d = np.hstack((motion_rot6d, new_root_pos))
        else:
            rot_6d = matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(sub_arr), 'XYZ'))
            if i == 3:  # root rotations
                rot_6d = matrix_to_rotation_6d(euler_angles_to_matrix(new_euler, 'XYZ'))
            motion_rot6d = np.hstack((motion_rot6d, rot_6d))

    return motion_rot6d


def movement_save_init(motion_data, filename):
    """
        Helper function to validate the fragment we sliced from MotionAnno.action_list given below
        It works the same as the function movement_to_rot6d() above, but the output is in euler angle form
        we'll put this output into Houdini to check if the fragment we got is correct
    """

    """
        motions 是 bvh 文件中的 frames 部分，每一行代表一帧内骨骼的位置，一行数据的形式如下：
        (root x/y/z 绝对坐标, root x/y/z 欧拉角(内旋，绝对坐标), joint[0] x/y/z 欧拉角(内旋，相对坐标), joint[1] x/y/z 欧拉角
                , ... , joint[k] x/y/z 欧拉角)
        我们做的归一化就是把所有动作的初始位置放在原点，初始面朝的方向一致
        由于 bvh 文件中除 root 之外的部分均为相对坐标，故归一化的过程仅需改变 motions[:, 0:6] 的值
    """
    motions = np.array(motion_data)
    init = motions[0, 0: 3] # root_x, root_y, root_z
    motion_rot6d = np.empty((motions.shape[0], 0))
    num_cols = motions.shape[1]
    num_rows = motions.shape[0]

    """ 
        rotate all action, letting all motion face x- axis initially,
        including:
            0. Rotation:    We figure out rotation by checking initial euler angle: [rot_x_0, rot_y_0, rot_z_0]
                            which is motions[0, 3: 6], and get the rotation matrix from it.
                            ATTENTION: euler angles in motions[] array are in degree, so must be multiplied by
                            pi/180 before transforming them into rotation matrices.
            1. Path:        The motion should be placed at (0, 0, 0) initially.
                            This step is simple, just apply the rotation matrix above to all root positions
                            Then get the initial root position (motions[0, 0:3]), minus it from
                            all root position.
            2. Euler Angle: Rotations and positions of joints in .bvh files are all relative coordinates, so
                            we just need to modify the root.
                            Transform root rotation into matrices, do matmul(rot_matrix, root_rotation) then
                            transform the result back to euler angles, where rot_matrix is the rotation matrix we get
                            in step 0.
    """

    # 1. Path， 旋转+移动回原点
    # Attention: xyz in bvh -> xzy in houdini (maybe, I'm mot sure about that, but the order is not the same)
    init_euler = euler_angles_to_matrix(torch.from_numpy(motions[0, 3: 6] * 3.1415926 / 180.), 'XYZ')
    rot_vec = torch.matmul(init_euler, torch.from_numpy(np.array([0., 0., 1.])).reshape((3, 1)))
    rot_e = -np.arccos(rot_vec[0] / np.sqrt(rot_vec[0] ** 2 + rot_vec[2] ** 2))
    rot_e = float(rot_e)

    joint_rot = np.array([
        np.cos(rot_e), 0, np.sin(rot_e),
        0, 1, 0,
        -np.sin(rot_e), 0, np.cos(rot_e)
    ])

    rot_mat = torch.from_numpy(joint_rot).reshape((3, 3))
    path_rotation_mat = torch.matmul(rot_mat, torch.from_numpy(motions[:, 0: 3]).transpose(1, 0))
    path_rotation_mat = path_rotation_mat.transpose(1, 0)
    # print(path_rotation_mat - motions[:, 0:3])

    new_init_pos = path_rotation_mat.numpy()[0, :]
    new_root_pos = path_rotation_mat.numpy() - new_init_pos

    # 2. Euler Angle，欧拉角转来转去
    rot_mat = torch.from_numpy(joint_rot).reshape((3, 3))

    """
        弧度制注意 ！！
    """
    joint_euler_mat = euler_angles_to_matrix(torch.from_numpy(motions[:, 3: 6] * 3.1415926 / 180.), 'XYZ')
    joint_rot_mat = torch.matmul(rot_mat.reshape(1, 3, 3), joint_euler_mat)
    new_euler = matrix_to_euler_angles(joint_rot_mat, 'XYZ')

    for i in range(0, num_cols, 3):
        sub_arr = motions[:, i: i + 3]
        if i == 0:
            # initial motion will be moved to (0, 0, 0)
            motion_rot6d = np.hstack((motion_rot6d, new_root_pos))
        elif i == 3:
            # 记得把弧度转回来
            motion_rot6d = np.hstack((motion_rot6d, new_euler * 180./3.1415926))
        else:
            motion_rot6d = np.hstack((motion_rot6d, sub_arr))

    np.savetxt(Path('./motion_segment') / Path(filename + '.bvh'), motion_rot6d, '%s', ' ', '\n')

class ActionLoader:
    def __init__(self, datapath: str):

        data_dict = {}
        new_name_list = []
        filelen = sum([1 for i in open(Path(datapath))])

        with open(Path(datapath), 'r') as f:
            for idx, line in tqdm(enumerate(f), 'Loading dataset', total=filelen):  # .readlines():
                # get annotation from json file
                anno = json.loads(line)
                if anno['detailStatus'] == "已标注":
                    try:
                        # preprocessing of file path according to Readme
                        motions = MotionAnno(anno)

                        converter = BVHConverter(Path('./data_loaders/wMIB/') / motions.file_path)
                        for action in motions.action_list:
                            start = action.start_frame
                            end = action.end_frame
                            # print(action.start_frame, action.end_frame, action.action)

                            action_data = converter.get_frames(start, end)

                            action_data_rot6d = movement_to_rot6d(action_data)

                            # 统计数据里貌似有>20个动作的序列(虽然不多)，因此用两个随机字母让名字不至于用完
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_'\
                                       + random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + str(motions.id)
                            while new_name in new_name_list:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_'\
                                           + random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + str(motions.id)
                            # print(new_name)

                            data_dict[new_name] = {
                                'motion': action_data_rot6d,
                                'length': len(action_data_rot6d),
                                'text': action.tokens_data,
                                'caption': action.caption
                            }

                            # this if branch is just for test
                            # if data_dict[new_name]['length'] == 231:
                            #     movement_save_init(action_data, new_name)
                            # print(data_dict[new_name]['text'])

                            new_name_list.append(new_name)

                    except:
                        pass

        """
            data_dict contains the separated motion data in bvh, 
            length of each motion, and labels about each motion
            
            new_name_list is a helper array to get items from data_dict
        """
        self.data_dict = data_dict
        self.new_name_list = new_name_list
        # print(data_dict[new_name_list[0]]['text'])

"""
class wMIB(Dataset):
    dataname = 'wMIB'

    def __init__(self, datapath: str = './annotated_samples.json',
                 mode: str = 'train',
                 downsample=True, **kwargs):

        self.downsample = downsample
        self.action_loader = ActionLoader(datapath)

        super().__init__()

    def __getitem__(self, index, mode='train'):

        key = self.action_loader.new_name_list[index]
        data = self.action_loader.data_dict[key]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        element = {
            'motion': motion,
            'length': m_length,
            'text': text_list
        }

        return element

    def __len__(self):
        return len(self.action_loader.new_name_list)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"
"""

class wMIB_Text2MotionDataset(Dataset):
    def __init__(self, mean, std, opt, w_vectorizer, datapath, **kwargs):

        self.dataname = 'wMIB'
        self.action_loader = ActionLoader(datapath)
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 200
        self.mean = np.zeros([189], dtype=np.float32)  # data is already normalized
        self.std = np.ones([189], dtype=np.float32)  # data is already normalized
        self.opt = opt

    def __len__(self):
        return len(self.action_loader.new_name_list)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, index, mode='train'):
        keyid = self.action_loader.new_name_list[index]
        batch = self.action_loader.data_dict[keyid]

        # Randomly choose a motion from batch
        tokens = batch['text']
        motion = batch['motion']
        m_length = batch['length']
        caption = batch['caption']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        idx = random.randint(0, abs(len(motion) - m_length))

        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length <= self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class wMIB(Dataset):
    def __init__(self, opt, split="train", **kwargs):

        self.split = split
        self.datapath = './data_loaders/wMIB/annotated_samples.json'
        abs_base_path = f'.'

        if opt is None:
            device = None  # torch.device('cuda:4') # This param is not in use in this context
            opt = get_opt('./dataset/humanml_opt.txt', device)
            opt.data_root = pjoin('dataset', 'babel')
            opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
            opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
            opt.text_dir = pjoin(abs_base_path, opt.text_dir)
            opt.model_dir = None
            opt.checkpoints_dir = '.'
            opt.data_root = pjoin(abs_base_path, opt.data_root)
            opt.save_root = pjoin(abs_base_path, opt.save_root)
            opt.meta_dir = './dataset'
            opt.dim_pose = 189
            opt.foot_contact_entries = 0
            opt.dataset_name = 'babel'
            opt.decomp_name = 'Decomp_SP001_SM001_H512_babel_2700epoch'
            opt.meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'motion1', 'meta')
            opt.min_motion_length = 0 # must be at least window size
            opt.max_motion_length = 200
        self.opt = opt

        print('Loading dataset %s ...' % opt.dataset_name)

        self.dataset_name = opt.dataset_name
        self.dataname = opt.dataset_name

        self.mean = np.zeros([opt.dim_pose], dtype=np.float32)  # data is already normalized
        self.std = np.ones([opt.dim_pose], dtype=np.float32)  # data is already normalized

        DATA = wMIB_Text2MotionDataset

        self.w_vectorizer = WordVectorizer('./glove', 'our_vab')
        self.t2m_dataset = DATA(
            opt=self.opt,
            mean=self.mean, std=self.std, w_vectorizer=self.w_vectorizer, datapath=self.datapath,
        )
        self.num_actions = 1  # dummy placeholder
        print(len(self.t2m_dataset))
        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


if __name__ == "__main__":
    wMIB(datapath='../wMIB/annotated_samples.json', opt=None)
