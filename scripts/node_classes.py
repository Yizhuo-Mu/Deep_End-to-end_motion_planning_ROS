import os
#from pylearn2.utils import serial
import re
#import numpy as N
#from pylearn2.config import yaml_parse
import string
#from theano import function1
from math import sqrt,pi,atan2

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import CNN


class Model:

    def __init__(self):

        #self.filename = model['filename']
        self.laser = tf.placeholder(shape=[1,1080,1],dtype=tf.float32)
        self.goal = tf.placeholder(shape=[1,4],dtype=tf.float32)
        self.logits = CNN.inference(self.laser, self.goal)
        #loss = CNN.loss(logits, cmd)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            CNN.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state('train7')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

    def run(self,laserdata,goaldata):
        out = None
        if laserdata is not None and goaldata is not None:
            out = self.sess.run(self.logits,feed_dict={self.laser : laserdata.reshape(1,1080,1), self.goal: goaldata.reshape(1,4)})
        return out


class SequenceInput:

    def __init__(self,length,mean=0,std=1,sequence=1,normalise=0):
        self.data = np.zeros((1,length))
        self.sequence = sequence
        self.normalise = normalise
        self.length = length
        self.mean = mean
        self.std = std

    def update(self,new_data):

        data = new_data.copy()

        if len(data.shape) < 2:
            data = data.reshape(1,data.shape[0])

        if self.normalise != 0:
            data -= self.mean
            data /= self.std

        self.data = np.concatenate((self.data[:,int(self.length/self.sequence):],data),axis=1)

class GoalManager:

    def __init__(self):
        self.current_goal = np.zeros((2))
        self.current_pose = np.zeros((3))
        self.relative_goal = np.zeros((2))
        try:
            self.tolerance = float(1.0)
        except:
            self.tolerance = 2.0
        #try:
        #    self.aim_next = settings.goals['aim_next'] == 'True'
        #except:
        self.aim_next = False
        self.first_goal = False
        self.reset()

    def reset(self):
        self.last_distance = self.tolerance
        self.within_goal = False
        self.closest = False
        self.next_goal_flag = False
        self.reorient = self.aim_next

    def update_pose(self,pose):
        if type(pose) == type([]):
            pose = np.asarray(pose)
        self.current_pose = pose

    def update_relative(self):


        ang = math.atan2(2 * self.current_pose[4] * self.current_pose[5],
                         1 - 2 * self.current_pose[4] * self.current_pose[4])
        # print("ang", ang / math.pi * 180, self.Robo)
        a = self.current_goal[0] - self.current_pose[0]
        b = self.current_goal[1] - self.current_pose[1]
        beta = math.atan2(b, a)
        goal_x = np.sqrt(np.square(a) + np.square(b)) * math.cos(ang - beta)
        goal_y = -np.sqrt(np.square(a) + np.square(b)) * math.sin(ang - beta)
        goal_r = np.sqrt(np.square(a) + np.square(b))
        goal_angle = beta - ang
        self.goaldata = np.array([goal_x, goal_y, goal_r, goal_angle])

        distance = goal_r
        self.relative_goal = np.asarray((distance, goal_angle))

        if distance < self.tolerance:
            self.within_goal = True
        else:
            self.within_goal = False

        if self.within_goal and (distance > self.last_distance or distance < 0.5*self.tolerance):
            self.closest = True

        if self.within_goal and self.closest:
            self.next_goal_flag = True

        self.last_distance = distance if distance < self.last_distance else self.last_distance

        if self.reorient and abs(angle) < np.deg2rad(5).item():
            self.reorient = False


    def update_goal(self,goal):

        if type(goal) == type([]):
            goal = np.asarray(goal)
        self.current_goal = goal
        if not self.first_goal:
            self.first_goal = True
        self.reset()
