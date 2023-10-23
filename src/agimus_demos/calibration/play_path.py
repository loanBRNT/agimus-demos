#!/usr/bin/env python
# Copyright 2021 CNRS - Airbus SAS
# Author: Florent Lamiraux
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from csv import writer
import subprocess
import time, roslib
import rospy
import math
import pinocchio
import yaml
import tf2_ros
import cv2 as cv
import eigenpy, numpy as np
from sensor_msgs.msg import Image, CameraInfo
from dynamic_graph_bridge_msgs.msg import Vector
import geometry_msgs.msg
from sensor_msgs.msg import JointState
from smach_msgs.msg import SmachContainerStatus
from std_msgs.msg import Bool, UInt32
from hpp.corbaserver import Client as HppClient

cameraInfoString='''<?xml version="1.0"?>
<root>
  <!--This file stores intrinsic camera parameters used
   in the vpCameraParameters Class of ViSP available
   at https://visp.inria.fr/download/ .
   It is constructed by reading the /camera_info topic.
   WARNING: distortion coefficients are not reported. -->
  <camera>
    <!--Name of the camera-->
    <name>Camera</name>
    <!--Size of the image on which camera calibration was performed-->
    <image_width>{width}</image_width>
    <image_height>{height}</image_height>
    <!--Intrinsic camera parameters computed for each projection model-->
    <model>
      <!--Projection model type-->
      <type>perspectiveProjWithoutDistortion</type>
      <!--Pixel ratio-->
      <px>{px}</px>
      <py>{py}</py>
      <!--Principal point-->
      <u0>{u0}</u0>
      <v0>{v0}</v0>
    </model>
    <model>
      <!--Projection model type-->
      <type>perspectiveProjWithDistortion</type>
      <!--Pixel ratio-->
      <px>{px}</px>
      <py>{py}</py>
      <!--Principal point-->
      <u0>{u0}</u0>
      <v0>{v0}</v0>
      <!--Distorsion-->
      <kud>0</kud>
      <kdu>0</kdu>
    </model>
  </camera>
</root>
'''
kinematicParamsString="""# Pose of {0} in {1}
camera:
  x: {2}
  y: {3}
  z: {4}
  roll: {5}
  pitch: {6}
  yaw: {7}
"""
# Write an image as read from a ROS topic to a file in png format
def writeImage(image, filename):
    count = 0
    a = np.zeros(3*image.width*image.height, dtype=np.uint8)
    a = a.reshape(image.height, image.width, 3)
    for y in range(image.height):
        for x in range(image.width):
            for c in range(3):
                a[y,x,c] = image.data[count]
                count +=1
    cv.imwrite(filename, a)

## Control calibration motions on UR10 robot
#
#  iteratively
#    - trigger motions planned by HPP by publishing in "/agimus/start_path"
#      topic,
#    - wait for end of motion by listening "/agimus/status/running" topic,
#    - record tf transformation between world and camera frames
#    - record image
class CalibrationControl (object):
    directory = "measurements"
    configDir = "config"
    endEffectorFrame = "ref_camera_link"
    origin = "world"
    maxDelay = rospy.Duration (1,0)
    squareSize = 0.025
    joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    def __init__ (self) :
        self.running = False
        self.sotJointStates = None
        self.rosJointStates = None
        self.jointNames = None
        self.pathId = 0
        self.mountFrame = None
        self.cameraFrame = "camera_color_optical_frame"
        if not rospy.core.is_initialized():
            rospy.init_node ('calibration_control')
        self.tfBuffer = tf2_ros.Buffer(rospy.Duration (1,0))
        self.tf2Listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pubStartPath = rospy.Publisher ("/agimus/start_path", UInt32,
                                             queue_size=1)
        self.subRunning = rospy.Subscriber ("/agimus/status/running", Bool,
                                            self.runningCallback)
        self.subStatus = rospy.Subscriber \
                         ("/agimus/agimus/smach/container_status",
                          SmachContainerStatus, self.statusCallback)
        self.subCameraInfo = rospy.Subscriber("/camera/color/camera_info",
                               CameraInfo, self.cameraInfoCallback)
        self.hppClient = HppClient ()
        self.count = 0
        self.measurements = list ()
        # set a default timeout
        self.timeout = 100.

    def playPath (self, pathId, collect_data = True):
        nbPaths = self.hppClient.problem.numberPaths ()
        if pathId >= nbPaths:
            raise RuntimeError ("pathId ({}) is bigger than number paths {}"
                                .format (pathId, nbPaths - 1))
        self.errorOccured = False
        self.pubStartPath.publish (pathId)
        self.waitForEndOfMotion ()
        if not self.errorOccured and collect_data:
            # wait to be sure that the robot is static
            rospy.sleep(1.)
            print("Collect data.")
            self.collectData ()

    def collectData (self):
        self.rosJointStates = None
        self.subRosJointState = rospy.Subscriber ("/joint_states", JointState,
                                                  self.rosJointStateCallback)
        self.image = None
        self.subImage = rospy.Subscriber ("/camera/color/image_raw", Image,
                                                   self.imageCallback)
        measurement = dict ()
        # record position of camera
        try:
            self.wMc = self.tfBuffer.lookup_transform\
                (self.origin, self.endEffectorFrame, rospy.Time(0))
            measurement["wMc"] = self.wMc
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            self.wMc = None
            rospy.loginfo ("No camera pose in tf")
        now = rospy.Time.now ()
        t1 = now
        # Get image.
        while not self.image or not self.rosJointStates:
            rospy.sleep(1e-3)
            t1 = rospy.Time.now ()
            if t1 - now > rospy.Duration(self.timeout):
                raise RuntimeError("Could not acquire data before timeout.")
        t = self.image.header.stamp
        # Check that data is recent enough
        if abs (now - t) < self.maxDelay:
            measurement["image"] = self.image
        else:
            rospy.loginfo ("time latest image from now: {}".
                           format ((now - t).secs + 1e-9*(now - t).nsecs))
        # Get joint values.
        if self.rosJointStates:
            measurement ["joint_states"] = self.rosJointStates
        self.measurements.append (measurement)

    def save(self):
        # write camera.xml from camera_info topic
        px = self.cameraInfo.K[0 * 3 + 0];
        py = self.cameraInfo.K[1 * 3 + 1];
        u0 = self.cameraInfo.K[0 * 3 + 2];
        v0 = self.cameraInfo.K[1 * 3 + 2];
        width = self.cameraInfo.width
        height= self.cameraInfo.height
        with open(self.directory + '/camera.xml', 'w') as f:
            f.write(cameraInfoString.format(width=width, height=height,
                                            px=px, py=py, u0=u0, v0=v0))
        # write urdf description of robot
        robotString = rospy.get_param('/robot_description')
        with open(self.directory + '/robot.urdf', 'w') as f:
            f.write(robotString)
        count=0
        for measurement in self.measurements:
            if not "image" in measurement.keys() or \
               not "wMc" in measurement.keys():
                continue
            count+=1
            writeImage(measurement['image'], self.directory + "/image-{}.png".\
                       format(count))
            with open(self.directory + "/pose_fPe_{}.yaml".format(count), 'w') as f:
                tf = measurement['wMc']
                rot = eigenpy.Quaternion(tf.transform.rotation.w,
                                         tf.transform.rotation.x,
                                         tf.transform.rotation.y,
                                         tf.transform.rotation.z)
                aa = eigenpy.AngleAxis(rot)
                utheta = aa.angle*aa.axis
                f.write("rows: 6\n")
                f.write("cols: 1\n")
                f.write("data:\n")
                f.write("- [{}]\n".format(tf.transform.translation.x))
                f.write("- [{}]\n".format(tf.transform.translation.y))
                f.write("- [{}]\n".format(tf.transform.translation.z))
                f.write("- [{}]\n".format(utheta[0]))
                f.write("- [{}]\n".format(utheta[1]))
                f.write("- [{}]\n".format(utheta[2]))
            if "joint_states" in measurement:
                with open(self.directory + f"/configuration_{count}", 'w') as f:
                    line = ""
                    q = len(measurement ["joint_states"])* [None]
                    for jn, jv in zip(self.jointNames,
                                      measurement ["joint_states"]):
                        q[self.joints.index(jn)] = jv
                    for jv in q:
                        line += f"{jv},"
                    line = line [:-1] + "\n"
                    f.write (line)


    def waitForEndOfMotion (self):
        rate = rospy.Rate(2) # 2hz
        # wait for motion to start
        rospy.loginfo ("wait for motion to start")
        while not self.running:
            rate.sleep ()
            if rospy.is_shutdown (): raise RuntimeError ("rospy shutdown.")
        # wait for motion to end
        rospy.loginfo ("wait for motion to end")
        while self.running:
            rate.sleep ()
            if rospy.is_shutdown (): raise RuntimeError ("rospy shutdown.")
        # wait for one more second
        rospy.loginfo ("motion stopped")
        for i in range (2):
            rate.sleep ()

    def runningCallback (self, msg):
        self.running = msg.data

    def statusCallback (self, msg):
        if msg.active_states [0] == 'Error':
            self.errorOccured = True
            rospy.loginfo ('Error occured.')

    def imageCallback(self, msg):
        self.image = msg
        self.subImage.unregister()
        self.subImage = None

    def cameraInfoCallback(self, msg):
        self.cameraInfo = msg
        self.subCameraInfo.unregister()
        self.subCameraInfo = None

    def sotJointStateCallback (self, msg):
        self.sotJointStates = msg.data

    def rosJointStateCallback (self, msg):
        self.rosJointStates = msg.position
        if not self.jointNames:
            self.jointNames = msg.name
        self.subRosJointState.unregister()
        self.subRosJointState = None

    def computeHandEyeCalibration(self):
        res = subprocess.run(["visp-compute-chessboard-poses",
                              "--square_size", f"{self.squareSize}",
                              "--input", f"{self.directory}/image-%d.png",
                              "--intrinsic", f"{self.directory}/camera.xml",
                              "--output", f"{self.directory}/pose_cPo_%d.yaml",
                              "--no_interactive"])
        if res.returncode != 0:
            raise RuntimeError("program compute-chessboard-poses failed.")
        res = subprocess.run(["visp-compute-hand-eye-calibration",
                              "--data-path", f"{self.directory}",
                              "--fPe", "pose_fPe_%d.yaml", "--cPo",
                              "pose_cPo_%d.yaml", "--output", "eMc.yaml"])
        if res.returncode != 0:
            raise RuntimeError("program compute-hand-eye-calibration failed.")

    def computeCameraPose(self):
        # We denote
        #  - m panda2_hand,
        #  - c camera_color_optical_frame,
        #  - e ref_camera_link (end effector).
        # we wish to compute a new position mMe_new of ref_camera_link in
        # panda2_hand in such a way that
        #  - eMc remains the same (we assume the camera is well calibrated),
        #  - mMc = mMe_new * eMc = mMe * eMc_measured
        # Thus
        #  mMe_new = mMe*eMc_measured*eMc.inverse()
        timeout = 5.
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        # Get pose of end effector frame in mount frame.
        mMe = None
        try:
            mMe_ros = tfBuffer.lookup_transform(self.mountFrame,
                self.endEffectorFrame, rospy.Time(), rospy.Duration(timeout))
            mMe_ros = mMe_ros.transform
            x = mMe_ros.rotation.x
            y = mMe_ros.rotation.y
            z = mMe_ros.rotation.z
            w = mMe_ros.rotation.w
            n = math.sqrt(x*x+y*y+z*z+w*w)
            mMe = pinocchio.XYZQUATToSE3(
                [mMe_ros.translation.x, mMe_ros.translation.y,
                 mMe_ros.translation.z, x/n, y/n, z/n, w/n])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            raise RuntimeError(str(e))

        # Get pose of optical frame in end effector frame.
        eMc = None
        try:
            eMc_ros = tfBuffer.lookup_transform(self.endEffectorFrame,
                self.cameraFrame, rospy.Time(), rospy.Duration(timeout))
            eMc_ros = eMc_ros.transform
            x = eMc_ros.rotation.x
            y = eMc_ros.rotation.y
            z = eMc_ros.rotation.z
            w = eMc_ros.rotation.w
            n = math.sqrt(x*x+y*y+z*z+w*w)
            eMc = pinocchio.XYZQUATToSE3(
                [eMc_ros.translation.x, eMc_ros.translation.y,
                 eMc_ros.translation.z, x/n, y/n, z/n, w/n])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            raise RuntimeError(str(e))

        eMc_measured = None
        with open(self.directory + "/eMc.yaml", "r") as f:
            d = yaml.safe_load(f)
            v = np.array(list(zip(*d['data']))[0])
            eMc_measured = pinocchio.SE3(translation=v[0:3],
                rotation = pinocchio.exp3(v[3:6]))
        return mMe*eMc_measured*eMc.inverse()

    def writeCameraParameters(self, mMe_new):
        xyz = mMe_new.translation
        rpy = pinocchio.rpy.matrixToRpy(mMe_new.rotation)
        with open(self.configDir + "/calibrated-params.yaml", "w") as f:
            f.write(kinematicParamsString.format(
            self.endEffectorFrame, self.mountFrame,
                xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]))

def playAllPaths (startIndex):
    i = startIndex
    while i < nbPaths:
        cc.playPath (i)
        if not cc.errorOccured:
            print("Ran {}".format(i))
            i+=1
        #rospy.sleep (1)

if __name__ == '__main__':
    cc = CalibrationControl ()
    nbPaths = cc.hppClient.problem.numberPaths ()
    #playAllPaths (0)
