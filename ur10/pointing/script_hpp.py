import sys, argparse, numpy as np, time, rospy
from math import pi
from hpp.corbaserver import loadServerPlugin
from hpp.corbaserver.manipulation import Robot, loadServerPlugin, \
    createContext, newProblem, ProblemSolver, ConstraintGraph, \
    ConstraintGraphFactory, Rule, Constraints, CorbaClient
from hpp.gepetto.manipulation import ViewerFactory
from tools_hpp import ConfigGenerator

class PartP72:
    urdfFilename = "package://agimus_demos/urdf/P72-with-table.urdf"
    srdfFilename = "package://agimus_demos/srdf/P72.srdf"
    rootJointType = "freeflyer"

# parse arguments
defaultContext = "corbaserver"
p = argparse.ArgumentParser (description=
                             'Initialize demo of UR10 pointing')
p.add_argument ('--context', type=str, metavar='context',
                default=defaultContext,
                help="identifier of ProblemSolver instance")
args = p.parse_args ()

try:
    import rospy
    Robot.urdfString = rospy.get_param('robot_description')
    print("reading URDF from ROS param")
except:
    print("reading generic URDF")
    from hpp.rostools import process_xacro, retrieve_resource
    Robot.urdfString = process_xacro\
      ("package://agimus_demos/urdf/ur10_robot.urdf.xacro",
       "transmission_hw_interface:=hardware_interface/PositionJointInterface")
Robot.srdfString = ""

loadServerPlugin (args.context, "manipulation-corba.so")
newProblem()
client = CorbaClient(context=args.context)
client.basic._tools.deleteAllServants()
def wd(o):
    from hpp.corbaserver import wrap_delete
    return wrap_delete(o, client.basic._tools)
client.manipulation.problem.selectProblem (args.context)

robot = Robot("robot", "ur10", rootJointType="anchor", client=client)
ps = ProblemSolver(robot)
ps.loadPlugin("manipulation-spline-gradient-based.so")
vf = ViewerFactory(ps)

## Shrink joint bounds of UR-10
#
jointBounds = dict()
jointBounds["default"] = [ (jn, robot.getJointBounds(jn)) \
                           if not jn.startswith('ur10/') else
                           (jn, [-pi, pi]) for jn in robot.jointNames]

## Remove some collision pairs
#
ur10JointNames = filter(lambda j: j.startswith("ur10/"), robot.jointNames)
ur10LinkNames = [ robot.getLinkNames(j) for j in ur10JointNames ]

## Load P72
#
vf.loadRobotModel (PartP72, "part")
robot.setJointBounds('part/root_joint', [-2, 2, -2, 2, -2, 2])

robot.client.manipulation.robot.insertRobotSRDFModel\
    ("ur10", "package://agimus_demos/srdf/ur10_robot.srdf")


## Define initial configuration
q0 = robot.getCurrentConfig()
q0[:3] = [0, -pi/2, pi/2]
r = robot.rankInConfiguration['part/root_joint']
q0[r:r+3] = [0., 1., 0.8]

## Build constraint graph
all_handles = ps.getAvailable('handle')
part_handles = filter(lambda x: x.startswith("part/"), all_handles)

graph = ConstraintGraph(robot, 'graph')
factory = ConstraintGraphFactory(graph)
factory.setGrippers(["ur10/gripper",])
factory.setObjects(["part",], [part_handles], [[]])
factory.generate()
graph.initialize()
