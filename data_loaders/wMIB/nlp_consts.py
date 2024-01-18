# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# aspell list < your_file.txt > mispell2
from re import L


SPELL_CORRECTOR = {
                    "快": "Fast",
                    "中": "Medium speed",
                    "慢": "Slowly",
                    "前": "Forward",
                    "后": "Back",
                    "左": "Left",
                    "右": "Right",
                    "向左转": "Turn left",
                    "向右转": "Turn right",
                    "无转向": "No turning",
                    "左前": "Forward left",
                    "右前": "Forward right",
                    "左后": "Backward left",
                    "右后": "Backward right",
                    "上": "Up",
                    "下": "Down"
}

def fix_spell(words):
    l_words = words.strip().split()
    for i, x in enumerate(l_words):
        if x in SPELL_CORRECTOR:
            l_words[i] = SPELL_CORRECTOR[x]
        else:
            l_words[i] = x
    return ' '.join(l_words)
