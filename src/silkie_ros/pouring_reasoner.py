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
from geometry_msgs.msg import PoseStamped, Point
from mujoco_msgs.msg import ObjectStateArray
from std_msgs.msg import String


class Blackboard(object):

    def __init__(self):
        self.experts = []
        self.scene_desc = {
            "source": "sync_cup2",
            "dest": "sync_bowl",
            "source_type": "Container",
            "dest_type": "Container",
            "total_particles": 200,
            "source_dim": (0.0651, 0.0651, 0.08),  # l, d, h
            "dest_dim": (0.1, 0.1, 0.05)
        }
        self.context_values = {
            'updatedBy': "",
            'near': False,
            'nothingOut': False,
            'tooSlow': False,
            'tooFast': False,
            'tricklesAlongSource': False,
            'bouncingOutOfDest': False,
            'streamTooWide': False,
            'tippingDest': False,
            'collision': False,
            'dir_overshoot': "",  # todo: compute the orientation w.r.t dest pose to find the direction
            "source_pose": PoseStamped(),
            "dest_pose": PoseStamped()
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
        return facts

    def create_facts_from_context(self):
        self.current_facts.update(self.base_facts)
        print("near value ", self.bb.context_values["near"])
        if self.bb.context_values["near"]:
            self.current_facts.update(
                Reasoner.create_facts(self.bb.scene_desc["source"], "near", self.bb.scene_desc["dest"],
                                      silkie.DEFEASIBLE))
        if not self.bb.context_values["near"]:
            self.current_facts.update(
                Reasoner.create_facts(self.bb.scene_desc["source"], "-near", self.bb.scene_desc["dest"],
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
            print("building conclusion")
            rules = silkie.loadDFLRules('./rules.dfl')
            conclusions = silkie.buildTheory(rules, self.current_facts, {}, debugTheory=True)
        return conclusions


class SimulationSource:

    def __init__(self, bb):
        self.dest_limits: tuple = ()
        self.src_limits: tuple = ()
        self.bb = bb
        self.sim_subscriber = rospy.Subscriber("/mujoco/object_states", ObjectStateArray, self.pose_listener)
        #  variables to update context values
        self.distance: float = 0.0

    @staticmethod
    def get_limits(length: float, breadth: float, height: float, position: Point) -> tuple:
        half_height = height / 2
        half_breadth = breadth / 2
        half_length = length / 2

        ll = (position.x - half_length, position.y - half_breadth, position.z - half_height)
        ul = (position.x + half_length, position.y + half_breadth, position.z + half_height)

        return ll, ul

    def pose_listener(self, req):

        def inside(upper, lower, val):
            return lower[0] < val.x < upper[0] and lower[1] < val.y < upper[1] and lower[2] \
                   < val.z < upper[2]

        if req.object_states:
            print("pose listener")
            count = 0
            particle_positions = []
            for obj in req.object_states:
                # print("name ", obj.name)
                if obj.name == self.bb.scene_desc["source"]:
                    # print("source")
                    self.bb.context_values["source_pose"].header.frame_id = self.bb.scene_desc["source"]
                    self.bb.context_values["source_pose"].header.stamp = rospy.Time.now()
                    self.bb.context_values["source_pose"].pose = obj.pose
                    self.src_limits = SimulationSource.get_limits(self.bb.scene_desc["source_dim"][0],
                                                                  self.bb.scene_desc["source_dim"][1],
                                                                  self.bb.scene_desc["source_dim"][2],
                                                                  self.bb.context_values[
                                                                      "source_pose"].pose.position)
                    # print("src limits ", self.src_limits)

                elif obj.name == self.bb.scene_desc["dest"]:  # Static so sufficient just get it once and not update!
                    # print("dests")
                    self.bb.context_values["dest_pose"].header.frame_id = self.bb.scene_desc["dest"]
                    self.bb.context_values["dest_pose"].header.stamp = \
                        self.bb.context_values["source_pose"].header.stamp
                    self.bb.context_values["dest_pose"].pose = obj.pose
                    self.dest_limits = self.get_limits(self.bb.scene_desc["dest_dim"][0],
                                                       self.bb.scene_desc["dest_dim"][1],
                                                       self.bb.scene_desc["dest_dim"][2],
                                                       self.bb.context_values["dest_pose"].pose.position)
            for obj in req.object_states:
                if "ball" in obj.name:
                    if not inside(self.src_limits[1], self.src_limits[0], obj.pose.position) and not inside(
                            self.dest_limits[1], self.dest_limits[0], obj.pose.position):
                        particle_positions.append([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
                        count += 1

            self.distance = math.dist((self.bb.context_values["source_pose"].pose.position.x,
                                       self.bb.context_values["source_pose"].pose.position.y,
                                       self.bb.context_values["source_pose"].pose.position.z),
                                      (self.bb.context_values["dest_pose"].pose.position.x,
                                       self.bb.context_values["dest_pose"].pose.position.y,
                                       self.bb.context_values["dest_pose"].pose.position.z))

            diff = [self.bb.context_values["dest_pose"].pose.position.x,
                    self.bb.context_values["dest_pose"].pose.position.y,
                    self.bb.context_values["dest_pose"].pose.position.z - np.mean(particle_positions)]

            if count > 0.25 * self.bb.scene_desc["total_particles"]:
                self.bb.pred_spilling = True
                print("ooopss spilled.... ", count)
            # else:
            # print("no spilling ", count)

    def update(self):
        if 0.0 < self.distance <= 0.2:  # todo: add a value based on the objects involved

            self.bb.context_values["near"] = True
            # hashed = hash((self.bb.source, "near", self.bb.dest))
            # if hashed not in self.bb.old_facts:
            #     self.bb.old_facts.add(hashed)
            # near_pred = "near_"+self.bb.source+"_"+self.bb.dest
            # if near_pred not in self.bb.current_facts:
            #     self.bb.current_facts.append(near_pred)

            print("near true")
        else:
            # self.bb.current_facts.update(self.bb.create_facts(self.bb.source, "-near", self.bb.dest,
            #                                                   silkie.DEFEASIBLE))
            self.bb.context_values["near"] = False
            print("near false")


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
