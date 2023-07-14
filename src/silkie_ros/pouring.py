import rospy
import math

import silkie

from geometry_msgs.msg import PoseStamped, Point
from mujoco_msgs.msg import ObjectStateArray


# TODO: 
# create facts before sending to build theory - 
# Check if the value of the predicate changed and update the facts
# publish perform conclusions (to be used by giskard)


# Qns: How to keep the facts up to date?
# What is the right time to build theory? Anytime we update facts?
# Publish when the facts changed

strict_fact = 1
defeasible_fact = 2

class PouringFacts:

    def __init__(self, source : str = None, dest : str = None, src_dim : set = {}) -> None:

        self.source_dim = (0.0651, 0.0651, 0.08)  ## l, d, h
        self.dest_dim = (0.1, 0.1, 0.05)
        self.total_particles = 200
        self.pred_near = False
        self.pred_spilling = False

        self.facts = {}

        if source is None:
            self.source="sync_cup2"
        if dest is None:
            self.dest="sync_bowl"

    def create_facts(self,  s: str, p: str, o: str = None, fact_type: int = defeasible_fact) -> dict:
        facts = {p: silkie.PFact(p)}
        facts[p].addFact(s, o, fact_type)
        return facts
    
    


class SimChannel:

    def __init__(self) -> None:
        self.sim_subscriber = rospy.Subscriber("/mujoco/object_states", ObjectStateArray, self.pose_listener)
        self.pose_update_duration = rospy.Duration(5.0)
        self.pf = PouringFacts()
        self.src_pose = PoseStamped()
        self.dest_pose = PoseStamped()
        self.src_limits = ()
        # TODO : Create a hash with facts in (s,p,o) format. If something has changed wrt current fact. Create facts and call build theory
  
    def get_limits(self, length:float, breadth : float, height : float, position: Point) -> tuple: 
        half_height = height/2
        half_breadth = breadth/2
        half_length = length/2

        ll = (position.x - half_length, position.y - half_breadth, position.z - half_height)
        ul = (position.x + half_length, position.y + half_breadth, position.z + half_height)

        return (ll, ul)

    def pose_listener(self, req):
        
        if req.object_states and  (self.src_pose.header.stamp.secs == 0 or
                                   ((rospy.Time.now() - self.src_pose.header.stamp) >= self.pose_update_duration)):
           

            print("I am in")
            count = 0

            inside = lambda upper, lower, val: val.x > lower[0] and val.x < upper[0] and val.y > lower[1] and val.y < upper[1] and val.z > lower[2] and val.z < upper[2]

            for obj in req.object_states:
                if obj.name == self.pf.source:
                    print("source")
                    self.src_pose.header.frame_id = self.pf.source
                    self.src_pose.header.stamp = rospy.Time.now()
                    self.src_pose.pose = obj.pose
                    self.src_limits = self.get_limits(self.pf.source_dim[0], self.pf.source_dim[1], self.pf.source_dim[2], self.src_pose.pose.position)

                elif obj.name == self.pf.dest: ## Static so sufficient just get it once and not update!
                    print("dests")
                    self.dest_pose.header.frame_id = self.pf.dest
                    self.dest_pose.header.stamp = self.src_pose.header.stamp
                    self.dest_pose.pose = obj.pose
                    self.dest_limits = self.get_limits(self.pf.dest_dim[0], self.pf.dest_dim[1], self.pf.dest_dim[2], self.dest_pose.pose.position)

            for obj in req.object_states:
                if "ball" in obj.name:
                    if not inside(self.src_limits[1], self.src_limits[0], obj.pose.position) and not inside(self.dest_limits[1], self.dest_limits[0], obj.pose.position):
                        count +=1


            distance = math.dist((self.src_pose.pose.position.x, self.src_pose.pose.position.y,
                                  self.src_pose.pose.position.z),
                                 (self.dest_pose.pose.position.x, self.dest_pose.pose.position.y,
                                  self.dest_pose.pose.position.z))
            if distance <= max(self.pf.source_dim):
                self.pf.pred_near = True
                hashed = hash((self.pf.source, "near", self.pf.dest))
                if hashed not in self.pf.old_facts:
                    self.pf.old_facts.add(hashed)
                self.pf.create_facts(self.pf.source, "near", self.pf.dest, defeasible_fact)
                print("near true")
            else:
                print(self.pf.create_facts(self.pf.source, "-near", self.pf.dest, defeasible_fact))
                print("near false")

            if count > 0.25*self.pf.total_particles:
                self.pf.pred_spilling = True
                print("ooopss spilled.... ", count)
            else:
                print("no spilling ", count)

    


if __name__ == '__main__':
    rospy.init_node('reasoner_sim_node')
    SimChannel()
    rospy.spin()
