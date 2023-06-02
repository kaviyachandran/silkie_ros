import rospy
import math

import silkie

from threading import Lock

from geometry_msgs.msg import PoseStamped
from mujoco_msgs.msg import ObjectStateArray

# TODO: 1. Subscribe to object states?
# 2. Get object poses
# 3. function for predicates that require some computation E.g : near?
# 4. And do the minor things like if
# 5. To create facts before sending to build theory
# >>> t = silkie.readTerm(t2)
# >>> facts[t[0]] = silkie.PFact(t[0])
# >>> facts
# {'contains': <silkie.PFact object at 0x7fce402519a0>}
# >>> facts[t[0]].addFact(t[1], t[2], 'DEFEASIBLE')

strict_fact = 'STRICT'
defeasible_fact = 'DEFEASIBLE'

class PouringFacts:

    def __init__(self, source : str = None, dest : str = None, src_dim : set = {}) -> None:
        #if src_dim is None: 
        self.source_dim = frozenset({0.0651, 0.0651, 0.06})  ## l, d, h
        self.pred_near=False
        if source is None:
            self.source="sync_cup2"
        if dest is None:
            self.dest="sync_bowl"


    def create_facts(self, fact_type:str, s:str, p:str ,o:str=None) -> dict:
        facts = {}
        facts[p] = silkie.PFact(p)
        facts[p].addFact(s, o, fact_type)
        return facts


class SimChannel:

    def __init__(self) -> None:
        print("aallll good")
        self.sim_subscriber = rospy.Subscriber("/mujoco/object_states", ObjectStateArray, self.pose_listener)
        self.pose_update_duration = rospy.Duration(5.0)
        self.pf = PouringFacts()
        self.src_pose = PoseStamped()
        self.dest_pose = PoseStamped()

    def pose_listener(self, req):
        # print("time ", (rospy.Time.now() - self.src_pose.header.stamp) >=  self.pose_update_duration)
        # print(req.object_states)
        
        if req.object_states and  (self.src_pose.header.stamp.secs == 0 or ((rospy.Time.now() - self.src_pose.header.stamp) >=  self.pose_update_duration)):
            print("I am in")
            for obj in req.object_states:
                if obj.name == self.pf.source:
                    print("source")
                    self.src_pose.header.frame_id = self.pf.source
                    self.src_pose.header.stamp = rospy.Time.now()
                    self.src_pose.pose = obj.pose

                if obj.name == self.pf.dest:
                    print("dests")
                    self.dest_pose.header.frame_id = self.pf.dest
                    self.dest_pose.header.stamp = self.src_pose.header.stamp
                    self.dest_pose.pose = obj.pose

            distance = math.dist((self.src_pose.pose.position.x, self.src_pose.pose.position.y, self.src_pose.pose.position.z),
                                  (self.dest_pose.pose.position.x, self.dest_pose.pose.position.y, self.dest_pose.pose.position.z))
            if distance <= max(self.pf.source_dim):
                self.pf.pred_near=True
                print("near true")
            else:
                print("near false")

        
        




if __name__ == '__main__':
    rospy.init_node('reasoner_sim_node')
    SimChannel()
    # rospy.Subscriber("/mujoco/object_states", ObjectState, sim.pose_listener)
    rospy.spin()
