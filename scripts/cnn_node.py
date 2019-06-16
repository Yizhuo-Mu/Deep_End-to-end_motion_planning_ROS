#!/usr/bin/env python
import sys, argparse, rospy
import numpy as N
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseActionGoal
from rospy.numpy_msg import numpy_msg
import time
from std_msgs.msg import String,Bool,Float64MultiArray
from tf.transformations import euler_from_quaternion as efq
from node_classes import Model, SequenceInput, GoalManager
import untangle
from scipy.ndimage.filters import gaussian_filter1d
from math import tanh

class CNN_node:
    def __init__(self):

        print("loading model...")

        self.cnn = Model()


        self.new_goal = False
        self.message_sent = False

        self.goals = GoalManager()

        self.laser = None
        self.goal = None
        self.command = None
        self.pose = None

        print("...done.")

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.goal_pub = rospy.Publisher("/goal_reached", String, queue_size=10)
        self.acc_pub = rospy.Publisher("/accuracy",String, queue_size=10)
        self.data_pub = rospy.Publisher('/data', Float64MultiArray,queue_size=10)

        self.scan_sub = rospy.Subscriber("/base_scan", numpy_msg(LaserScan), self.laser_callback)
        self.goal_sub = rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, self.goal_callback)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        self.stall_sub = rospy.Subscriber("/stall", Bool, self.stall_callback)

        self.twist_msg = Twist()
        self.string_msg = String()
        self.data_msg = Float64MultiArray()

        print("\nAwaiting first goal!")

        time.sleep(2)

        self.string_msg.data = 's'
        self.goal_pub.publish(self.string_msg)
        self.string_msg.data = ''

    def stall_callback(self,stall):

        if stall.data:
            self.string_msg.data = str(self.goals.last_distance)
            self.acc_pub.publish(self.string_msg)
            self.string_msg.data = 'c'
            self.goal_pub.publish(self.string_msg)
            print 'Collision!'
            rospy.signal_shutdown('Collision')
            sys.exit("Collision")

    def pose_callback(self,pose):

        self.pose = N.asarray([pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, pose.pose.pose.orientation.z, pose.pose.pose.orientation.w])

        self.goals.update_pose(self.pose)

        if self.goals.next_goal_flag:

            self.string_msg.data = str(self.goals.last_distance)
            self.acc_pub.publish(self.string_msg)

            self.string_msg.data = str(self.goals.current_goal)
            self.goal_pub.publish(self.string_msg)

    def laser_callback(self,laser):

        if self.goal is not None:
            if not self.goals.first_goal:
                return
                # Shift goal sequence and add current relative goal
            self.goals.update_relative()

        las = laser.ranges
        self.laserdata = N.asarray(las)
        if laser is not None and self.goal is not None:
            out = self.cnn.run(self.laserdata, self.goals.goaldata)



            self.twist_msg.linear.x = out[0][0]
            self.twist_msg.angular.z = out[0][1]
            # Publish command
            if not self.goals.next_goal_flag:
                self.cmd_pub.publish(self.twist_msg)

            # Print information
            sys.stdout.write('                                                                                                                                         \r')
            #sys.stdout.write('Goal:\tx = %2.1f, y = %2.1f\t\tTo Goal:\tr = %2.1f, theta = %2.1f\tUsing %s\r\r' % (self.goals.current_goal[0],self.goals.current_goal[1],self.goals.relative_goal[0],self.goals.relative_goal[1],factor))
            sys.stdout.flush()

    def goal_callback(self,goalPose):


        # receive new goal
        self.goals.update_goal(N.asarray([goalPose.goal.target_pose.pose.position.x, goalPose.goal.target_pose.pose.position.y,
                                 goalPose.goal.target_pose.pose.orientation.x,
                                 goalPose.goal.target_pose.pose.orientation.y,
                                 goalPose.goal.target_pose.pose.orientation.z,
                                 goalPose.goal.target_pose.pose.orientation.w]))
        self.goal = goalPose
        # dodgy screen update
        sys.stdout.write('\r')
        sys.stdout.flush()

def make_argument_parser():

    parser = argparse.ArgumentParser(description="Launch a prediction from a pkl file")

    parser.add_argument('xml', help='Specifies the xml file containing settings')
    parser.add_argument('-t', dest='topic',default='/base_scan', help='Supply the appropriate laser sensor topic')

    return parser

def main():
    rospy.init_node('CNN_node', anonymous=True)
    CNNNode = CNN_node()
    rospy.spin()

if __name__ == '__main__':
    main()