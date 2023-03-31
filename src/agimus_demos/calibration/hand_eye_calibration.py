# Copyright 2020 CNRS - Airbus SAS
# Author: Florent Lamiraux, Joseph Mirabel, Alexis Nicolin
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

# This script selectively does one of the following
#
#  1. generate n configurations where the camera looks at the left wrist
#     options --N=n --arm=left,
#  2. generate n configurations where the camera looks at the right wrist
#     options --N=n --arm=right,
#  3. reads configurations from file './data/all-configurations.csv')
#
#  Then, it computes a path going through all these configurations.

import os
from csv import reader, writer
import yaml
import argparse, numpy as np
from hpp import Transform
from hpp.corbaserver import wrap_delete
from hpp.corbaserver.manipulation import Constraints, SecurityMargins
import pinocchio
from pinocchio import SE3, exp3, RobotWrapper
from ..tools_hpp import RosInterface
from .. import InStatePlanner

# get indices of closest configs to config [i]
def getClosest(dist,i,n):
    d = list()
    for j in range(dist.shape[1]):
        if j!=i:
            d.append((j,dist[i,j]))
    d.sort(key=lambda x:x[1])
    return list(zip(*d))[0][:n]

def connectedComponent(ps, q):
    for i in range(ps.numberConnectedComponents()):
        if q in ps.nodesConnectedComponent(i):
            return i
    raise RuntimeError\
        ('Configuration {} does not seem to belong to the roadmap.'.format\
         (q))

class Calibration(object):
    """
    Various methods to perform hand-eye calibration
    """
    transition = "Loop | f"
    nbConnect = 100
    camera_frame = None
    chessboard_name = "part"
    chessboard_frame = 'part/base_link'
    chessboard_joint = 'part/root_joint'
    # Position of the chessboard center in the part frame
    chessboard_center = (0,0,0)
    # For security margins
    robot_name = None
    security_distance_robot_robot = 0
    security_distance_robot_chessboard = 0.015
    security_distance_chessboard_universe = float("-inf")
    security_distance_robot_universe = .01
    default_security_distance = .01

    def wd(self, o):
        return wrap_delete(o, self.ps.client.basic)

    def __init__(self, ps, graph, factory):
        self.ps = ps
        self.graph = graph
        self.factory = factory
        self.euclideanDistance = np.zeros(0).reshape(0,0)
        self.cproblem = self.wd(ps.client.basic.problem.getProblem())
        self.cgraph = self.wd(self.cproblem.getConstraintGraph())
        self.cedge = dict()
        self.pathValidation = dict()

    # Store edge and path validation instances in a dictionary of key edge names
    def getPathValidation(self):
        if not self.transition in self.pathValidation:
            self.cedge[self.transition] = self.wd(self.cgraph.get(
                self.graph.edges[self.transition]))
            self.pathValidation[self.transition] = self.wd(
                self.cedge[self.transition].getPathValidation())
        return self.pathValidation[self.transition]


    def addStateToConstraintGraph(self):
        if 'look-at-cb' in self.graph.nodes.keys():
            return
        if self.robot_name is None:
            raise RuntimeError("You should specify a robot name in member " +
                "robot_name for setting security margins.")
        ps = self.ps; graph = self.graph
        hpp_camera_frame = self.robot_name + "/" + self.camera_frame
        ps.createPositionConstraint\
            ('look-at-cb', hpp_camera_frame, self.chessboard_frame,
             (0, 0, 0), self.chessboard_center,
             (True, True, False))

        ps.createLockedJoint(f"place_{self.chessboard_name}/complement",
            self.chessboard_joint, [0, 0, 0, 0, 0, 0, 1])
        ps.setConstantRightHandSide(f"place_{self.chessboard_name}/complement",
            False)

        graph.createNode(['look-at-cb'])
        graph.createEdge('free', 'look-at-cb', 'go-look-at-cb', 1,
                         'free')
        graph.createEdge('look-at-cb', 'free', 'stop-looking-at-cb', 1,
                         'free')

        graph.addConstraints(node='look-at-cb',
            constraints = Constraints(numConstraints=['look-at-cb']))
        graph.addConstraints(edge='go-look-at-cb',
            constraints = Constraints(numConstraints=
                [f"place_{self.chessboard_name}/complement"]))
        graph.addConstraints(edge='stop-looking-at-cb',
            constraints = Constraints(numConstraints=
                [f"place_{self.chessboard_name}/complement"]))

        self.factory.generate()
        self.sm = SecurityMargins(ps, self.factory, [self.robot_name,
            self.chessboard_name])
        self.sm.setSecurityMarginBetween(self.robot_name, self.robot_name,
                                    self.security_distance_robot_robot)
        self.sm.setSecurityMarginBetween(self.robot_name, self.chessboard_name,
                                    self.security_distance_robot_chessboard)
        self.sm.setSecurityMarginBetween(self.robot_name, "universe",
            self.security_distance_robot_universe)
        # deactivate collision checking between chessboard and environment
        self.sm.setSecurityMarginBetween(self.chessboard_name, "universe",
            self.security_distance_chessboard_universe)
        self.sm.defaultMargin = self.default_security_distance
        self.sm.apply()
        graph.initialize()
        self.transition = 'go-look-at-cb'

    ## Generate calibration configurations
    #
    # \param q0 configuration of the robot that specified the pose of the
    #           chessboard
    # \param n number of configurations to generate,
    # \param m, M minimal and maximal distances between the camera and the
    #             chessboard
    def generateValidConfigs(self, q0, n, m, M):
        robot = self.ps.robot
        graph = self.graph
        hpp_camera_frame = self.robot_name + "/" + self.camera_frame
        if not hpp_camera_frame in robot.getAllJointNames():
            raise RuntimeError("Specify the camera frame with member " +
                f"camera_frame. Current value {self.camera_frame} prefixed " +
                f"with {self.robot_name}/ is not in robot.getAllJointNames()")
        if not self.chessboard_frame in robot.getAllJointNames():
            raise RuntimeError("Specify the chessboard frame with member " +
                f"chessboard_frame. Current value {self.chessboard_frame} is " +
                               "not in robot.getAllJointNames()")
        result = list()
        i = 0
        while i < n:
            q = self.shootRandomConfigs(q0, 1)[0]
            robot.setCurrentConfig(q)
            hpp_camera_frame = self.robot_name + "/" + self.camera_frame
            wMc = Transform(robot.getLinkPosition(hpp_camera_frame))
            wMp = Transform(robot.getLinkPosition(self.chessboard_frame))
            # Position of chessboard center in world frame
            c = wMp.transform(np.array(self.chessboard_center))
            # Position of chessboard center in camera frame
            c1 = wMc.inverse().transform(c)
            if not (m < c1[2] and c1[2] < M): continue
            result.append(q)
            i += 1
        return result

    # Generate calibration configs and integrate them in a roadmap
    #
    #   After building a roadmap with those configurations, if all of
    #   them are not in the same connected component, add random
    #   configurations to the roadmap and add edges between those configurations
    #   and previously existing configurations, until half of the nbConfigs
    #   configurations lie in the same connected component of the roadmap
    def generateConfigurationsAndPaths(self, q_init, nbConfigs,
                                       filename = None):
        transition = self.transition
        self.transition = 'go-look-at-cb'
        ri = RosInterface(self.ps.robot)
        if filename:
            self.calibConfigs = self.readConfigsInFile(filename)
        else:
            self.calibConfigs = self.generateValidConfigs\
                (q_init, nbConfigs, .3, .5)
        finished = False
        configs = [q_init] + self.calibConfigs
        # save transition
        self.transition = "Loop | f"
        for q in configs:
            res, msg = self.getPathValidation().validateConfiguration(q)
            if not res:
                print (f"{q} is not valid")
                return
            res, err = self.graph.getConfigErrorForEdgeLeaf("Loop | f", q_init,
                                                            q)
            if not res:
                print(f"{q} is not reachable from q_init")
                return

        while not finished:
            self.buildRoadmap(configs)
            # find connected component of q_init
            for i in range(self.ps.numberConnectedComponents()):
                cc = self.ps.nodesConnectedComponent(i)
                if not q_init in cc:
                    continue
                print(f'number of nodes in q_init connected component: {len(cc)}')
                # From here on, q_init is in cc
                finished = True
                count = 0
                for q in self.calibConfigs:
                    if q in cc:
                        count +=1
            if 2*count < len(self.calibConfigs): finished = False
            # if half of the configurations are in the same connected component,
            # get out of the while loop
            if finished: break
            # Otherwise generate another batch of random configurations
            configs += self.shootRandomConfigs(q_init, 10)
            self.buildRoadmap(configs, len(configs) - 10)
        self.transition = transition
        configs = self.orderConfigurations([q_init] + self.calibConfigs)
        self.visitConfigurations([q_init] + self.calibConfigs)

    # Write configurations in a file in CSV format
    def writeConfigsInFile(self, filename, configs):
        with open(filename, "w") as f:
            w = writer(f)
            for q in configs:
                w.writerow(q)

    # Read configurations in a file in CSV format
    def readConfigsInFile(self, filename):
        with open(filename, "r") as f:
            configurations = list()
            r = reader(f)
            for line in r:
                configurations.append(list(map(float,line)))
        return configurations

    # distance between configurations
    def distance(self, q0, q1) :
        ''' Distance between two configurations of the box'''
        assert(len(q0) == self.ps.robot.getConfigSize())
        assert(len(q1) == self.ps.robot.getConfigSize())
        d = self.ps.hppcorba.problem.getDistance()
        return d.call(q0, q1)

    def shootRandomConfigs(self, q0, n):
        robot = self.ps.robot
        configs = list()
        i = 0
        while i < n:
            q = robot.shootRandomConfig()
            res, q1, err = self.graph.generateTargetConfig\
               (self.transition, q0, q)
            if not res: continue
            res, msg = self.getPathValidation().validateConfiguration(q1)
            if res:
                configs.append(q1)
                i += 1
        return configs

    def buildEuclideanDistanceMatrix (self, configs, start_at=0):
        assert(self.euclideanDistance.shape[0] >= start_at and \
               self.euclideanDistance.shape[1] >= start_at)
        # Build matrix of distances between configurations
        N = len (configs)
        # Copy previous matrix in bigger matrix
        dist = np.array (np.zeros (N*N).reshape (N,N))
        dist[0:start_at,0:start_at] = self.euclideanDistance[0:start_at,
                                                             0:start_at]
        for i in range (N):
            for j in range (max(i+1, start_at),N):
                dist [i,j] = self.distance (configs [i], configs [j])
                dist [j,i] = dist [i,j]
        self.euclideanDistance = dist

    def buildRoadmapDistanceMatrix(self, configs):
        N = len(configs)
        # Build matrix of distances between box poses
        dist = np.matrix(np.zeros(N*N).reshape(N,N))
        # Initialize matrix with 1e8 between all different configurations
        for i in range(N):
            for j in range(i+1,N):
                dist[i,j] = dist[j,i] = 1e8
        # Set value for configurations that are linked by an edge in the roadmap
        for k in range(self.ps.numberEdges()):
            e = self.ps.edge(k)
            if e[0] in configs and e[1] in configs:
                i = configs.index(e[0])
                j = configs.index(e[1])
                dist[i,j] = dist[j,i] = self.distance(e[0],e[1])
        return dist

    def orderConfigurations(self, configs):
        N = len(configs)
        # Order configurations according to naive solution to traveler
        # salesman problem
        notVisited = list(range(1,N))
        visited = [0]
        dist = self.buildRoadmapDistanceMatrix(configs)
        while len(notVisited) > 0:
            # rank of current configuration in visited
            i = visited [-1]
            # find closest not visited configuration
            m = 1e20
            closest = None
            for j in notVisited:
                if dist [i,j] < m:
                    m = dist [i,j]
                    closest = j
            notVisited.remove(closest)
            visited.append(closest)
        orderedConfigs = list()
        for i in visited:
            orderedConfigs.append(configs [i])
        return orderedConfigs

    # Build a roadmap with the input configurations
    #
    #   The configurations are inserted as nodes
    #   Edges are built between the closest configurations
    def buildRoadmap(self, configs, start_at = 0):
        if len(configs) - start_at ==0: return
        self.buildEuclideanDistanceMatrix(configs, start_at)
        for q in configs[start_at:]:
            self.ps.addConfigToRoadmap(q)
        for i, q in enumerate(configs[start_at:]):
            closest = getClosest(self.euclideanDistance, i, self.nbConnect)
            for j in closest:
                if self.euclideanDistance[i,j] != 0 and j<i:
                    qi=configs[i]
                    qj=configs[j]
                    res, pid, msg = self.ps.directPath(qi,qj,True)
                    if res:
                        self.ps.addEdgeToRoadmap(qi,qj,pid,True)
        # clear paths
        for i in range(self.ps.numberPaths(),0,-1):
            self.ps.erasePath(i-1)

    def visitConfigurations(self, configs):
        nOptimizers = len(self.ps.getSelected("PathOptimizer"))
        for q_init, q_goal in zip(configs, configs [1:]):
            if connectedComponent(self.ps, q_goal) == \
               connectedComponent(self.ps, q_init):
                self.ps.resetGoalConfigs()
                self.ps.setInitialConfig(q_init)
                self.ps.addGoalConfig(q_goal)
                self.ps.solve()
                for i in range(nOptimizers):
                    # remove non optimized paths
                    pid = self.ps.numberPaths() - 2
                    self.ps.erasePath(pid)
            else:
                break
    # Generate a csv file with the following data for each configuration:
    #  x, y, z, phix, phiy, phiz,
    #  ur10e/shoulder_pan_joint, ur10e/shoulder_lift_joint, ur10e/elbow_joint,
    #  ur10e/wrist_1_joint, ur10e/wrist_2_joint, ur10e/wrist_3_joint
    #  corresponding to the pose of the camera in the world frame (orientation
    #  in roll, pitch, yaw), and the joints angles of the robot.
    #
    #  maxIndex is the maximal index of the input files:
    #    configuration_{i}, i<=maxIndex,
    #    pose_cPo_{i}.yaml, i<=maxIndex.
    def generateDataForFigaroh(self, input_directory, output_file, maxIndex):
        robot = self.ps.robot
        # Use reference pose of chessboard. Note that this is the pose of the
        # frame extracted from pose computation top left part of the chessboard
        wMo = SE3(translation=np.array([1.28, 0.108, 1.4775]),
                  rotation = np.array([[0,0,1],[-1,0,0],[0,-1,0]]))
        # create a pinocchio model from the Robot.urdfString
        with open('/tmp/robot.urdf', 'w') as f:
            f.write(robot.urdfString)
        pinRob = RobotWrapper()
        pinRob.initFromURDF('/tmp/robot.urdf')
        # get camera frame id
        camera_frame_id = pinRob.model.getFrameId(self.camera_frame)
        with open(output_file, 'w') as f:
            # write header line in output file
            header = "x1,y1,z1,phix1,phiy1,phiz1,"
            for n in pinRob.model.names[1:]:
                header += n + ","
            f.write(header[:-1])
            f.write("\n")
            count = 1
            while count <= maxIndex:
                poseFile = os.path.join(input_directory, f'pose_cPo_{count}.yaml')
                configFile = os.path.join(input_directory, f'configuration_{count}')
                try:
                    f1 = open(poseFile, 'r')
                    d = yaml.safe_load(f1)
                    f1.close()
                    f1 = open(configFile, 'r')
                    config = f1.readline()
                    f1.close()
                    cMo_tuple = list(zip(*d['data']))[0]
                    trans = np.array(cMo_tuple[:3])
                    rot = exp3(np.array(cMo_tuple[3:]))
                    cMo = SE3(translation = trans, rotation = rot)
                    oMc = cMo.inverse()
                    wMc = wMo * oMc
                    line = ""
                    # write camera pose
                    for i in range(3):
                        line += f'{wMc.translation[i]},'
                    rpy = pinocchio.rpy.matrixToRpy(wMc.rotation)
                    for i in range(3):
                        line += f'{rpy[i]},'
                    # write configuration
                    line += config
                    f.write(line)
                except FileNotFoundError as exc:
                    print(f'{poseFile} does not exist')
                count+=1
