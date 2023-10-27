"""
1. Create a skeleton with blackboard class, Experts like simulation, perception and other process can update the
context values.
2. A controller that publishes messages to Giskard based on the current context
"""

import math
from typing import Dict, Any

import numpy as np
import silkie

import rospy
import visualization_msgs
from geometry_msgs.msg import PoseStamped, Point
from mujoco_msgs.msg import ObjectStateArray
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray


class Blackboard(object):

    def __init__(self):
        self.experts = []
        self.scene_desc = {
            "source": "sync_cup2",
            "dest": "sync_bowl",
            "source_type": "Container",
            "dest_type": "Container",
            "poured_substance_type": "Liquid",
            "poured_substance": "particles",
            "total_particles": 200,
            "source_dim": (0.0651, 0.0651, 0.08),  # l, d, h
            "dest_dim": (0.2, 0.2, 0.1799),
            "dest_goal": 60
        }
        self.context_values = {
            'updatedBy': "",
            'near': False,  # src and dest
            'isTilted': False,  # src
            'poursTo': False,  # nothingOut # particles
            'tooSlow': False,  # particles
            'tooFast': False,  # particles
            'tricklesAlongSource': False,  # particles
            'bouncingOutOfDest': False,  # particles
            'streamTooWide': False,
            'tippingDest': False,
            'collision': False,
            'dir_overshoot': "",  # todo: compute the orientation w.r.t dest pose to find the direction
            "source_pose": PoseStamped(),
            "dest_pose": PoseStamped(),
            "dest_goal_reached": False,
            "hasOpeningWithin": False,
            "sourceUpright": True
        }

    def set_scene_desc(self, key: str, value):
        self.scene_desc[key] = value

    def get_scene_desc(self, key: str):
        return self.context_values[key]

    def set_context_values(self, key: str, value):
        self.context_values[key] = value

    def get_context_values(self, key: str):
        return self.context_values[key]

    def add_experts(self, expert):
        self.experts.append(expert)


class BlackboardController:

    def __init__(self, bb):
        self.concluded_behavior_publisher = rospy.Publisher("/reasoner/concluded_behaviors", String, queue_size=10)
        self.bb_obj = bb
        self.reasoner = Reasoner(bb)

    def update_context(self) -> None:
        print("controller")
        for expert in self.bb_obj.experts:
            expert.update()

    def publish_conclusions(self):
        theory = self.reasoner.build_theory()
        if len(theory):
            theory_canPour, s2i_canPour, i2s_canPour, theoryStr_canPour = theory
            print("theory", theoryStr_canPour)
            concluded_facts = theoryStr_canPour.split("\n")
        else:
            return
        publish_data: Dict = {}
        for conclusion in concluded_facts:
            predicate_list = conclusion.split(" => ")
            temp = {}
            if predicate_list[-1].startswith("P"):
                temp = {"condition": predicate_list[0].split(": ")[-1], "behavior": predicate_list[-1].strip("P_")}
                print("t  ", temp)
            publish_data.update(temp)
        # TODO : publish a string message with behaviors to giskard
        # print("data ", str(publish_data))
        self.concluded_behavior_publisher.publish(str(publish_data))
        self.reasoner.current_facts = {}
        return


class Reasoner:

    def __init__(self, bb):
        self.bb = bb
        self.base_facts = self.create_base_facts()
        self.current_facts = {}

    def create_base_facts(self):
        facts = {}
        facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], self.bb.scene_desc["source_type"], "",
                                           silkie.DEFEASIBLE))
        # as the predicate container is already present in facts
        facts[self.bb.scene_desc["dest_type"]].addFact(self.bb.scene_desc["dest"], "", silkie.DEFEASIBLE)
        facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"],
                                           self.bb.scene_desc["poured_substance_type"], "", silkie.DEFEASIBLE))
        facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "contains",
                                           self.bb.scene_desc["poured_substance"], silkie.DEFEASIBLE))
        return facts

    def create_facts_from_context(self):
        self.current_facts.update(self.base_facts)
        print("near value ", self.bb.context_values["near"])
        if self.bb.context_values["near"]:
            self.current_facts.update(
                Reasoner.create_facts(self.bb.scene_desc["source"], "near", self.bb.scene_desc["dest"],
                                      silkie.DEFEASIBLE))
        elif not self.bb.context_values["near"]:
            self.current_facts.update(
                Reasoner.create_facts(self.bb.scene_desc["source"], "-near", self.bb.scene_desc["dest"],
                                      silkie.DEFEASIBLE))
        if self.bb.context_values["poursTo"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "poursTo",
                                                            self.bb.scene_desc["dest"],
                                                            silkie.DEFEASIBLE))

        elif not self.bb.context_values["poursTo"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-poursTo",
                                                            self.bb.scene_desc["dest"],
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["isTilted"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "isTilted", "",
                                                            silkie.DEFEASIBLE))
        elif not self.bb.context_values["isTilted"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-isTilted", "",
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["tooSlow"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "slowFlowFrom",
                                                            self.bb.scene_desc["dest"],
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["tooFast"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "fastFlowFrom",
                                                            self.bb.scene_desc["dest"],
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["dest_goal_reached"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "goalReached",
                                                            "",
                                                            silkie.DEFEASIBLE))
        else:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "-goalReached",
                                                            "",
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["sourceUpright"]:
            self.create_facts(Reasoner.create_facts(self.bb.scene_desc["source"], "upright", "",
                                                    silkie.DEFEASIBLE))
        else:
            self.create_facts(Reasoner.create_facts(self.bb.scene_desc["source"], "-upright", "",
                                                    silkie.DEFEASIBLE))

        return

    @staticmethod
    def create_facts(s: str, p: str, o: str = None, fact_type: int = silkie.DEFEASIBLE) -> dict:
        facts = {p: silkie.PFact(p)}
        facts[p].addFact(s, o, fact_type)
        return facts

    def build_theory(self) -> tuple:
        conclusions = ()
        self.create_facts_from_context()
        if len(self.current_facts) != 0:
            # print("building conclusion")
            rules = silkie.loadDFLRules('./rules.dfl')
            conclusions = silkie.buildTheory(rules, self.current_facts, {}, debugTheory=True)
        return conclusions


class SimulationSource:

    def __init__(self, bb):
        self.dest_limits: tuple = ()
        self.src_limits: tuple = ()
        self.bb = bb
        self.sim_subscriber = rospy.Subscriber("/mujoco/object_states", ObjectStateArray, self.pose_listener)
        self.bounding_box_subscriber = rospy.Subscriber("/mujoco_object_bb", MarkerArray,
                                                        self.bb_listener)
        #  variables to update context values
        self.distance: float = 0.0
        self.src_orientation: tuple = ()
        self.object_flow: list = []
        # object dependent parameters

        self.src_bounding_box_pose = PoseStamped()
        self.src_bounding_box_dimensions: tuple = (0.0, 0.0, 0.0)
        self.dest_bounding_box_pose = PoseStamped()
        self.dest_bounding_box_dimensions: tuple = (0.0, 0.0, 0.0)
        self.bb_values_set = False

        self.distance_threshold = (0.0, 0.30)
        # in degrees. greater than [ 76.65427899,  -0.310846  , -34.33960301] along x this lead to pouring
        self.source_tilt_angle = 70.0
        self.object_flow_threshold = 10  # no.of particles per cycle
        self.source_upright_angle = 5.0
        self.dest_goal_reached = False

    @staticmethod
    def get_limits(length: float, breadth: float, height: float, position: Point) -> tuple:
        half_height = height / 2
        half_breadth = breadth / 2
        half_length = length / 2

        ll = (position.x - half_length, position.y - half_breadth, position.z - half_height)
        ul = (position.x + half_length, position.y + half_breadth, position.z + half_height)

        return ll, ul

    def bb_listener(self, data: visualization_msgs.msg.MarkerArray):
        src_set = False
        dest_set = False
        for marker in data.markers:
            if marker.ns == self.bb.scene_desc["source"]:
                self.src_bounding_box_dimensions = (marker.scale.x, marker.scale.y, marker.scale.z)
                self.src_bounding_box_pose = marker.pose
                src_set = True
            elif marker.ns == self.bb.scene_desc["dest"]:
                self.dest_bounding_box_dimensions = (marker.scale.x, marker.scale.y, marker.scale.z)
                self.dest_bounding_box_pose = marker.pose
                dest_set = True
        if src_set and dest_set:
            self.bb_values_set = True

    def pose_listener(self, req):

        def inside(upper, lower, val):
            return lower[0] < val.x < upper[0] and lower[1] < val.y < upper[1] and lower[2] \
                   < val.z < upper[2]

        if self.bb_values_set and req.object_states and (rospy.Time(req.header.stamp.secs, req.header.stamp.nsecs) -
                                                         rospy.Time(
                                                             self.bb.context_values["source_pose"].header.stamp.secs,
                                                             self.bb.context_values[
                                                                 "source_pose"].header.stamp.nsecs)).to_sec() >= 0.1:

            print("pose listener", (rospy.Time(req.header.stamp.secs, req.header.stamp.nsecs) -
                                    rospy.Time(self.bb.context_values["source_pose"].header.stamp.secs,
                                               self.bb.context_values["source_pose"].header.stamp.nsecs)).to_sec())
            count = 0
            count_not_in_source = 0
            count_in_dest = 0
            particle_positions = []
            for obj in req.object_states:
                # print("name ", obj.name)
                if obj.name == self.bb.scene_desc["source"]:
                    # print("source")
                    self.bb.context_values["source_pose"].header.frame_id = self.bb.scene_desc["source"]
                    self.bb.context_values["source_pose"].header.stamp = rospy.Time.now()
                    self.bb.context_values["source_pose"].pose = obj.pose
                    self.src_limits = SimulationSource.get_limits(self.src_bounding_box_dimensions[0],
                                                                  self.src_bounding_box_dimensions[1],
                                                                  self.src_bounding_box_dimensions[2],
                                                                  self.src_bounding_box_pose.position)

                    self.src_orientation = np.degrees(euler_from_quaternion([
                        self.bb.context_values["source_pose"].pose.orientation.x,
                        self.bb.context_values["source_pose"].pose.orientation.y,
                        self.bb.context_values["source_pose"].pose.orientation.z,
                        self.bb.context_values["source_pose"].pose.orientation.w]))
                    print("src orient :{}, quat: {}".format(self.src_orientation,
                                                            self.bb.context_values["source_pose"].pose.orientation))

                elif obj.name == self.bb.scene_desc["dest"]:  # Static so sufficient just get it once and not update!
                    # print("dests")
                    self.bb.context_values["dest_pose"].header.frame_id = self.bb.scene_desc["dest"]
                    self.bb.context_values["dest_pose"].header.stamp = \
                        self.bb.context_values["source_pose"].header.stamp
                    self.bb.context_values["dest_pose"].pose = obj.pose
                    self.dest_limits = self.get_limits(self.dest_bounding_box_dimensions[0],
                                                       self.dest_bounding_box_dimensions[1],
                                                       self.dest_bounding_box_dimensions[2],
                                                       self.dest_bounding_box_pose.position)
            # print(f'src pose: {self.bb.context_values["source_pose"]}',
            #     f'dest pose: {self.bb.context_values["dest_pose"]}')
            for obj in req.object_states:
                if "ball" in obj.name:
                    inside_src = inside(self.src_limits[1], self.src_limits[0], obj.pose.position)
                    inside_dest = inside(self.dest_limits[1], self.dest_limits[0], obj.pose.position)

                    if not inside_src and not inside_dest:
                        particle_positions.append([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
                        count += 1
                        if not inside_src:  # ToDo : check if the particle is inside the source has a velocity
                            count_not_in_source += 1
                    elif inside_dest:
                        count_in_dest += 1
            print("not in source: {}, in dest: {}".format(count_not_in_source, count_in_dest))
            current_particle_out = count_not_in_source - sum(self.object_flow)
            if count_in_dest == self.bb.scene_desc["dest_goal"]:
                self.dest_goal_reached = True
            if current_particle_out >= 0:
                self.object_flow.append(current_particle_out)
            # if len(self.object_flow) > 3:
            #     obj_avg = np.average(self.object_flow[-3:])
            #     # print("obj flow:{}, avg: {}".format(self.object_flow, obj_avg))
            # elif len(self.object_flow):
            #     obj_avg = np.average(self.object_flow)
            #     # print("obj flow:{}, avg: {}".format(self.object_flow, obj_avg))
            #  moving towards the dest obj.velocity.linear.x

            self.distance = math.dist((self.src_bounding_box_pose.position.x,
                                       self.src_bounding_box_pose.position.y),
                                      # self.bb.context_values["source_pose"].pose.position.z),
                                      (self.dest_bounding_box_pose.position.x,
                                       self.dest_bounding_box_pose.position.y))
            # self.bb.context_values["dest_pose"].pose.position.z))
            print("dist {}".format(self.distance))

            if count > 0.20 * self.bb.scene_desc["total_particles"]:
                self.bb.pred_spilling = True
                print("ooopss spilled.... ", count)
            else:
                print("no spilling ", count)

    def update(self):
        if self.distance_threshold[0] < self.distance <= self.distance_threshold[1]:
            # todo: add a value based on the objects involved

            self.bb.context_values["near"] = True
            # hashed = hash((self.bb.source, "near", self.bb.dest))
            # if hashed not in self.bb.old_facts:
            #     self.bb.old_facts.add(hashed)
            # near_pred = "near_"+self.bb.source+"_"+self.bb.dest
            # if near_pred not in self.bb.current_facts:
            #     self.bb.current_facts.append(near_pred)

            # print("near true")
        else:
            # self.bb.current_facts.update(self.bb.create_facts(self.bb.source, "-near", self.bb.dest,
            #                                                   silkie.DEFEASIBLE))
            self.bb.context_values["near"] = False
            # print("near false")

        # tilted
        check_tilt = abs(np.asarray(self.src_orientation)) > self.source_tilt_angle
        if check_tilt.any():
            self.bb.context_values["isTilted"] = True
        else:
            self.bb.context_values["isTilted"] = False

        # source upright
        if self.src_orientation[2] <= self.source_upright_angle:
            self.bb.context_values["sourceUpright"] = True
        else:
            self.bb.context_values["sourceUpright"] = False

        # poursTo
        if len(self.object_flow) and self.bb.context_values["isTilted"] and self.object_flow[-1] > 0:
            self.bb.context_values["poursTo"] = True
        else:
            self.bb.context_values["poursTo"] = False

        obj_avg = 0
        if len(self.object_flow) > 3:
            obj_avg = np.average(self.object_flow[-3:])
        elif len(self.object_flow) > 0:
            obj_avg = np.average(self.object_flow)

        if self.bb.context_values["poursTo"] and obj_avg < self.object_flow_threshold:
            self.bb.context_values["tooSlow"] = True
            print("slow true")
        else:
            self.bb.context_values["tooSlow"] = False

        if obj_avg > 100:
            self.bb.context_values["tooFast"] = True
            print("fast true")
        else:
            self.bb.context_values["tooFast"] = False

        if self.dest_goal_reached:
            self.bb.context_values["dest_goal_reached"] = True
        else:
            self.bb.context_values["dest_goal_reached"] = False


class Perception:
    pass


if __name__ == '__main__':
    rospy.init_node('reasoner_sim_node')
    blackboard = Blackboard()
    blackboard.add_experts(SimulationSource(blackboard))

    controller = BlackboardController(blackboard)

    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        controller.update_context()
        controller.publish_conclusions()
        rate.sleep()
