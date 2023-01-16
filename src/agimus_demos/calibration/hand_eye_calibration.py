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

from csv import reader, writer
import argparse, numpy as np
from hpp.corbaserver import wrap_delete
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

    def wd(self, o):
        return wrap_delete(o, self.ps.client.basic)

    def __init__(self, ps, graph):
        self.ps = ps
        self.graph = graph
        self.euclideanDistance = np.zeros(0).reshape(0,0)
        self.setTransition(self.transition)

    # Set transition
    #  \param transition name of the transition
    # Recover transition path validation object to validate configurations
    def setTransition(self, transition):
        self.transition = transition
        cproblem = self.wd(self.ps.client.basic.problem.getProblem())
        cgraph = cproblem.getConstraintGraph()
        cedge = self.wd(cgraph.get(self.graph.edges[transition]))
        self.pathValidation = self.wd(cedge.getPathValidation())

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
            res, msg = self.pathValidation.validateConfiguration(q1)
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
