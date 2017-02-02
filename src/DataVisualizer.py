#!/usr/bin/python3
'''
Visual-Doom-AI: Driver.py
Authors: Rafael Zamora, Lauren An, William Steele, Joshua Hidayat
Last Updated: 2/2/17
CHANGE-LOG:

'''

"""
Script used as to visualize data produced by models.

"""

from moviepy.editor import VideoClip
import numpy as np

def make_mp4(data):

    data = np.concatenate((data, data, data), axis=0)
    data = data.reshape(data.shape[0], 120, 160, 30)
    
    def make_frame(t):
        data[t]
        return mplfig_to_npimage() # (Height x Width x 3) Numpy array

    animation = VideoClip(make_frame, duration= len(data)) # 3-second clip

    # For the export, many options/formats/optimizations are supported
    animation.write_videofile("../doc/my_animation.mp4", fps=24) # export as video
    #animation.write_gif("my_animation.gif", fps=24) # export as GIF (slow)
