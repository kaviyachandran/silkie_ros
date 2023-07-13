import rospy
import math

import silkie

from geometry_msgs.msg import PoseStamped
from mujoco_msgs.msg import ObjectStateArray

import shapely as s
from shapely.geometry.polygon import Polygon

# TODO: 1. Subscribe to object states?
# 2. Get object poses
# 3. function for predicates that require some computation E.g : near?
# 4. create facts before sending to build theory


strict_fact = 'STRICT'
defeasible_fact = 'DEFEASIBLE'

class PouringFacts:

    def __init__(self, source : str = None, dest : str = None, src_dim : set = {}) -> None:
        #if src_dim is None: 
        self.source_dim = frozenset({0.0651, 0.0651, 0.08})  ## l, d, h
        self.pred_near = False

        if source is None:
            self.source="sync_cup2"
        if dest is None:
            self.dest="sync_bowl"

    def create_facts(self, fact_type: str, s: str, p: str, o: str = None) -> dict:
        facts = {p: silkie.PFact(p)}
        facts[p].addFact(s, o, fact_type)
        return facts
    
    def is_inside(self, object_dim: tuple, obj_shape: str, point: tuple) -> bool:
        if obj_shape == 'cylinder':
            pass


class SimChannel:

    def __init__(self) -> None:
        print("aallll good")
        self.sim_subscriber = rospy.Subscriber("/mujoco/object_states", ObjectStateArray, self.pose_listener)
        self.pose_update_duration = rospy.Duration(5.0)
        self.pf = PouringFacts()
        self.src_pose = PoseStamped()
        self.dest_pose = PoseStamped()
        self.bounding_box_source = []


    def create_bounding_box(length, breadth, height, center_point):
        half_height = height/2
        half_breadth = breadth/2
        half_length = length/2
        
        # Define the coordinates of the cuboid's vertices
        vertices = [
            (center_point[0] - half_length, center_point[1] - half_breadth, center_point[2] - half_height),
            (center_point[0] + half_length, center_point[1] - half_breadth, center_point[2] - half_height),
            (center_point[0] + half_length, center_point[1] + half_breadth, center_point[2] - half_height),
            (center_point[0] - half_length, center_point[1] + half_breadth, center_point[2] - half_height),
            (center_point[0] - half_length, center_point[1] - half_breadth, center_point[2] + half_height),
            (center_point[0] + half_length, center_point[1] - half_breadth, center_point[2] + half_height),
            (center_point[0] + half_length, center_point[1] + half_breadth, center_point[2] + half_height),
            (center_point[0] - half_length, center_point[1] + half_breadth, center_point[2] + half_height)
        ]

        # Define the cuboid's faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side face 1
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side face 2
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side face 3
            [vertices[3], vertices[0], vertices[4], vertices[7]]   # Side face 4
        ]

        print([
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side face 1
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side face 2
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side face 3
            [vertices[3], vertices[0], vertices[4], vertices[7]]   # Side face 4
        ])
        # Create the cuboid as a Polygon object
        bounding_box = Polygon()

        for face in faces:
            bounding_box = bounding_box.union(Polygon(face))

        return bounding_box


    def pose_listener(self, req):
        # print("time ", (rospy.Time.now() - self.src_pose.header.stamp) >=  self.pose_update_duration)
        # print(req.object_states)
        
        if req.object_states and  (self.src_pose.header.stamp.secs == 0 or
                                   ((rospy.Time.now() - self.src_pose.header.stamp) >= self.pose_update_duration)):
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

            distance = math.dist((self.src_pose.pose.position.x, self.src_pose.pose.position.y,
                                  self.src_pose.pose.position.z),
                                 (self.dest_pose.pose.position.x, self.dest_pose.pose.position.y,
                                  self.dest_pose.pose.position.z))
            if distance <= max(self.pf.source_dim):
                self.pf.pred_near = True
                print("near true")
            else:
                print("near false")

# class Predicates:

#     def __init__(self):
#         self.inside = False

    




if __name__ == '__main__':
    rospy.init_node('reasoner_sim_node')
    SimChannel()
    # rospy.Subscriber("/mujoco/object_states", ObjectState, sim.pose_listener)
    rospy.spin()
