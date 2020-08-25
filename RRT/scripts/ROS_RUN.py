#!/usr/bin/env python3

import rospy  # Import the Python library for ROS
from std_msgs.msg import Float64
#from time import time ,sleep
import numpy as np
from gazebo_msgs.srv import *
from sensor_msgs.msg import JointState


class ROS_master():

    def __init__(self,HZ):
        rospy.init_node('ROS_arm', anonymous=True)  # Initiate a Node named 'topic_publisher'
        self.pub=[]
        for j in [1,2]:
            for i in range(1,7):
                self.pub.append(rospy.Publisher('/robot{}/joint{}_position/command/'.format(j,i), Float64, queue_size=10))
        self.var = Float64()
        self.service_name = '/gazebo/get_joint_properties'
        self.joints = ['joint1', 'joint2', 'joint3', 'joint4','joint5','joint6']
        self.model_name=['robot1','robot2']
        self.rate = rospy.Rate(HZ)
        # Create a Publisher object, that will publish on the /counter topic
        # messages of type Int32

    def readstat(self):
        joints_list1 = [self.model_name[0] + '::' + joint for joint in self.joints]
        joints_list2 = [self.model_name[1] + '::' + joint for joint in self.joints]
        rospy.wait_for_service(self.service_name)
        service = rospy.ServiceProxy(self.service_name, GetJointProperties)
        req1 = GetJointPropertiesRequest()
        req2 = GetJointPropertiesRequest()
        while not rospy.is_shutdown():
            state1 = JointState()
            state2 = JointState()
            for i in range(len(joints_list1)):
                req1.joint_name = joints_list1[i]
                req2.joint_name = joints_list2[i]
                try:
                    response1 = service(req1)
                    response2 = service(req2)
                    if response1.success:
                        state1.name.append(joints_list1[i].split('::')[-1])
                        state1.position.append(response1.position[0])
                        state1.velocity.append(response1.rate[0])
                    else:
                        pass
                    if response2.success:
                        state2.name.append(joints_list2[i].split('::')[-1])
                        state2.position.append(response2.position[0])
                        state2.velocity.append(response2.rate[0])
                    else:
                        pass
                except Exception as e:
                    rospy.logerr("Exception in Joint State Publisher.")
                    print(type(e))
                    print(e)
                    return e
            return np.array([np.append(state1.position,state2.position,axis=0)])*180/np.pi

    def send_to_arm(self,allpoint):
        #allpoint=allpoint*np.pi/180
        while not rospy.is_shutdown():  # Create a loop that will go until someone stops the program execution
            for j in range(len(allpoint)):
                for i in range(12):
                    self.var.data = allpoint[j,i]
                    self.pub[i].publish(self.var)
                    self.rate.sleep()
            break

