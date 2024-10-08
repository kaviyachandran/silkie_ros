"""
        1. Create a skeleton with blackboard class, Experts like simulation, perception and other process can update the
        context values.
        2. A controller that publishes messages to Giskard based on the current context
3. Models / objects are assumed to be with length along x-axis, depth along y-axis
            Separate the predicates based on the actions?

"""

import math
from typing import Dict

import numpy as np
import silkie
import matplotlib.pyplot as plt
import argparse

from utils import Utils

import rospy
from geometry_msgs.msg import PoseStamped, Point, Pose
from mujoco_msgs.msg import ObjectStateArray
from mujoco_msgs.msg import ContactInfo
from std_msgs.msg import String, Float64
from visualization_msgs.msg import MarkerArray


class Blackboard(object):
    # While adding new scenes, change name, dimensions

    def __init__(self):
        self.corner_regions = {"1,1": "top_right", "1,-1": "top_left", "-1,1": "bottom_right", "-1,-1": "bottom_left"}
        # width or diameter of the opening
        self.dest_opening = {"narrow": (0.05, 0.1),
                             "medium": (0.10, 0.15),
                             "wide": (0.15, 0.2),
                             "veryWide": (0.2, 0.3)}
        self.src_height = (0.15, 0.2, 0.3, 0.35)

        # self.max_pouring_height = 0.3

        self.experts = []
        self.scene_desc = {
            "source": "pot",
            "dest": "bowl",
            "source_type": "Container",
            "dest_type": "Container",
            "poured_substance_type": "Thing",  # changing Thing to Liquid
            "poured_substance": "particles",
            "total_particles": 100,
            "source_dim": (),
            "dest_dim": (),
            "dest_goal": 70,
            "sourceHasEdges": True,  # This can be obtained from some vis marker array if the type is set correctly
            "retained_substance_type": "Thing",
            "retained_substance": "big_ball",
            "retained_substance_count": 1,
            "action_type": "draining"
            # todo: More info on actions performed? for e.g: draining requires retained substance and poured substance
        }
        self.context_values = {
            'updatedBy': "",
            'near': False,  # src and dest
            'isTilted': False,  # src
            'isDestTilted': False,  # dest
            'poursTo': False,  # nothingOut # particles
            'tooSlow': False,  # particles
            'tooFast': False,  # particles
            'tricklesAlongSource': False,  # particles
            'bouncingOutOfDest': False,  # particles
            'streamTooWide': False,
            'tippingDest': False,
            'isSpilled': False,
            'isSpilling': False,
            'collision': False,
            'dir_overshoot': "",  # todo: compute the orientation w.r.t dest pose to find the direction
            "source_pose": PoseStamped(),
            "dest_pose": PoseStamped(),
            "retained_substance_pose": PoseStamped(),
            "dest_goal_reached": False,
            "sourceUpright": True,
            "hasOpeningWithin": False,
            "movesUpIn": False,
            "almostGoalReached": False,
            "srcCornerAligned": False,
            "rotationDirection": "",
            "overshoot": False,
            "undershoot": False,
            "src_empty": False,
            # - (0, -1) ---> Behind. move upwards. in +y direction
            # - (0, 1) ---> FrontOf. move downwards. in -y direction
            # - (1, 0) ---> Right. move left. in -x direction
            # - (-1, 0) ---> Left. move right. in +x direction
            "srcAbove": False,
            "locationOfSourceRelativeToDestination": [],
            "srcCornerSpoutRegion": "",
            "collides": False,
            "retainedSubstanceCloseToOpening": False,
            "sourceContainsRetainedSubstance": False,
            "destinationHasOpening": "",
            "srcHeightLimits": 0.0,
            "srcFarAbove": False
        }
        # dimension_data = rospy.wait_for_message("/mujoco_object_bb", MarkerArray, timeout=5)
        # if dimension_data:
        #     self.set_dimension_values(dimension_data)
        # else:
        self.scene_desc["source_dim"] = (0.2, 0.2, 0.12)
        self.scene_desc["dest_dim"] = (0.2, 0.2, 0.18)
        dest_param_to_test = np.sort([self.scene_desc["dest_dim"][0], self.scene_desc["dest_dim"][1]])

        for index, value in enumerate(self.dest_opening.items()):
            if value[1][0] <= dest_param_to_test[0] < value[1][1]:
                self.context_values["destinationHasOpening"] = value[0]
                self.context_values["srcHeightLimits"] = self.src_height[index]
                break

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

    def set_dimension_values(self, data):
        for marker in data.markers:
            if marker.ns == self.scene_desc["source"]:
                self.scene_desc["source_dim"] = (marker.scale.x, marker.scale.y, marker.scale.z)
            elif marker.ns == self.scene_desc["dest"]:
                self.scene_desc["dest_dim"] = (marker.scale.x, marker.scale.y, marker.scale.z)

        print("dimensions  ", self.scene_desc["source_dim"], self.scene_desc["dest_dim"])


class BlackboardController:

    def __init__(self, bb: Blackboard, visualize: bool):
        self.concluded_behavior_publisher = rospy.Publisher("/reasoner/concluded_behaviors", String, queue_size=10)

        self.bb_obj = bb
        self.reasoner = Reasoner(bb)
        self.queries_for_experts = []
        self.visualize = visualize
        self.fig_counter = 0
        self.fig, self.plotWindow = None, None
        if self.visualize:
            self.fig, self.plotWindow = plt.subplots(figsize=(7, 7))

    def update_context(self) -> None:
        print("controller ", self.queries_for_experts)
        print("vis ", self.visualize)
        for expert in self.bb_obj.experts:
            expert.update()
            expert.query(self.queries_for_experts)
        self.queries_for_experts = []

    def reportTargetConclusions(self, theory, i2s, whitelist, fig, plotWindow, graphStoragePrefix=None, k=0):
        # print(type(theory), type(i2s))

        if plotWindow is not None:
            plt.cla()
        if "" == graphStoragePrefix:
            graphStoragePrefix = None
        visFile = None
        if graphStoragePrefix is not None:
            visFile = graphStoragePrefix + "_%d.gml" % k

        conclusions = silkie.dflInference(theory, i2s=i2s, fig=fig, plotWindow=plotWindow, visFile=visFile)
        conclusions = silkie.idx2strConclusions(conclusions, i2s)
        for c in conclusions.defeasiblyProvable:
            if c[0] in whitelist:
                print("def ", c)
        if plotWindow is not None:
            plt.draw()
            plt.pause(0.001)
            input("Press ENTER to continue")

    def get_consequents(self):
        theory = self.reasoner.build_theory()
        if len(theory):
            theory_canPour, s2i_canPour, i2s_canPour, theoryStr_canPour = theory
            print(f'counter:{self.fig_counter} theory: {theoryStr_canPour}')
            concluded_facts = theoryStr_canPour.split("\n")
            # print("t ", theory_canPour)
            # print("i2s ", i2s_canPour)
            if self.visualize:
                self.reportTargetConclusions(theory_canPour, i2s_canPour, ["canPour", "-canPour"],
                                             graphStoragePrefix=None, fig=self.fig, plotWindow=self.plotWindow,
                                             k=self.fig_counter)
            self.fig_counter += 1
        else:
            return
        publish_data: Dict = {}
        for conclusion in concluded_facts:
            predicate_list = conclusion.split(" => ")
            temp = {}
            if predicate_list[-1].startswith("P_"):
                temp = {predicate_list[0].split(": ")[-1]: predicate_list[-1].strip("P_")}
                # print("t  ", temp)
                publish_data.update(temp)
            elif predicate_list[-1].startswith("Q"):
                self.queries_for_experts.append(predicate_list[-1].strip("Q_"))
        # TODO : publish a string message with behaviors to giskard
        print("data ", str(publish_data))
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
        facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "PouredSubstanceRole", "",
                                           silkie.DEFEASIBLE))
        # facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "contains",
        #                                    self.bb.scene_desc["poured_substance"], silkie.DEFEASIBLE))
        facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "SourceRole", "", silkie.DEFEASIBLE))
        facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "DestinationRole", "", silkie.DEFEASIBLE))
        if self.bb.scene_desc["action_type"] == "draining":
            facts.update(Reasoner.create_facts(self.bb.scene_desc["action_type"], "Draining", "", silkie.DEFEASIBLE))
            facts.update(Reasoner.create_facts(self.bb.scene_desc["retained_substance"], "RetainedSubstanceRole", "",
                                               silkie.DEFEASIBLE))
            facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "contains",
                                               self.bb.scene_desc["retained_substance"], silkie.DEFEASIBLE))
        if self.bb.scene_desc["sourceHasEdges"]:
            facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "hasEdges", "", silkie.DEFEASIBLE))
        elif not self.bb.scene_desc["sourceHasEdges"]:
            facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-hasEdges", "", silkie.DEFEASIBLE))
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
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))

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
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "-goalReached", "",
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["sourceUpright"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "upright", "",
                                                            silkie.DEFEASIBLE))
        else:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-upright", "",
                                                            silkie.DEFEASIBLE))
        if self.bb.context_values["isDestTilted"]:
            if "isTilted" in self.current_facts.keys():
                self.current_facts["isTilted"].addFact(self.bb.scene_desc["dest"], "", silkie.DEFEASIBLE)
            else:
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "isTilted", "",
                                                                silkie.DEFEASIBLE))
        elif not self.bb.context_values["isDestTilted"]:
            if "-isTilted" in self.current_facts.keys():
                self.current_facts["-isTilted"].addFact(self.bb.scene_desc["dest"], "", silkie.DEFEASIBLE)
            else:
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "-isTilted", "",
                                                                silkie.DEFEASIBLE))

        if self.bb.context_values["hasOpeningWithin"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "hasOpeningWithin",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        elif not self.bb.context_values["hasOpeningWithin"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-hasOpeningWithin",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))

        if self.bb.context_values["collides"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "contact",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        elif not self.bb.context_values["collides"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-contact",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))

        for direction in self.bb.context_values["locationOfSourceRelativeToDestination"]:
            # print("direction ", direction)
            if direction == "inFront":
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "inFrontOf",
                                                                self.bb.scene_desc["source"], silkie.DEFEASIBLE))
            elif direction == "behind":
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "behind",
                                                                self.bb.scene_desc["source"], silkie.DEFEASIBLE))
            if direction == "left":
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "leftOf",
                                                                self.bb.scene_desc["source"], silkie.DEFEASIBLE))
            elif direction == "right":
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "rightOf",
                                                                self.bb.scene_desc["source"], silkie.DEFEASIBLE))

        self.bb.context_values["locationOfSourceRelativeToDestination"] = []

        if self.bb.context_values["isSpilled"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "isSpilled", "",
                                                            silkie.DEFEASIBLE))
        elif not self.bb.context_values["isSpilled"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "-isSpilled", "",
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["isSpilling"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "isSpilling", "",
                                                            silkie.DEFEASIBLE))
        elif not self.bb.context_values["isSpilling"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "-isSpilling", "",
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["movesUpIn"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "movesUpIn",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        elif not self.bb.context_values["movesUpIn"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "-movesUpIn",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        if self.bb.context_values["almostGoalReached"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "almostGoalReached",
                                                            "", silkie.DEFEASIBLE))
        elif not self.bb.context_values["almostGoalReached"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "-almostGoalReached",
                                                            "", silkie.DEFEASIBLE))
        if self.bb.context_values["rotationDirection"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.context_values["rotationDirection"],
                                                            "RotationDirection",
                                                            "", silkie.DEFEASIBLE))

        if self.bb.scene_desc["sourceHasEdges"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.context_values["srcCornerSpoutRegion"],
                                                            "CornerRegion", "", silkie.DEFEASIBLE))

            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"],
                                                            "hasLowestOpeningCorner",
                                                            self.bb.context_values["srcCornerSpoutRegion"],
                                                            silkie.DEFEASIBLE))

            if self.bb.context_values["srcCornerAligned"]:
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "aligned",
                                                                self.bb.context_values["srcCornerSpoutRegion"],
                                                                silkie.DEFEASIBLE))
            elif not self.bb.context_values["srcCornerAligned"]:
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "-aligned",
                                                                self.bb.context_values["srcCornerSpoutRegion"],
                                                                silkie.DEFEASIBLE))

        if self.bb.context_values["srcAbove"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "above",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        elif not self.bb.context_values["srcAbove"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-above",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        if self.bb.context_values["undershoot"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "undershoot", "",
                                                            silkie.DEFEASIBLE))
        else:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["poured_substance"], "-undershoot", "",
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["sourceEmpty"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-contains",
                                                            self.bb.scene_desc["poured_substance"],
                                                            silkie.DEFEASIBLE))
        else:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "contains",
                                                            self.bb.scene_desc["poured_substance"],
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["srcFarAbove"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "farAbove",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        else:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-farAbove",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))

        if self.bb.context_values["retainedSubstanceCloseToOpening"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["retained_substance"], "closeToOpening",
                                                            "", silkie.DEFEASIBLE))
        elif not self.bb.context_values["retainedSubstanceCloseToOpening"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["retained_substance"], "-closeToOpening",
                                                            "", silkie.DEFEASIBLE))
        if self.bb.context_values["sourceContainsRetainedSubstance"]:
            if "contains" in self.current_facts.keys():
                self.current_facts["contains"].addFact(self.bb.scene_desc["source"],
                                                       self.bb.scene_desc["retained_substance"], silkie.DEFEASIBLE)
            else:
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "contains",
                                                                self.bb.scene_desc["retained_substance"],
                                                                silkie.DEFEASIBLE))
        elif not self.bb.context_values["sourceContainsRetainedSubstance"]:
            if "-contains" in self.current_facts.keys():
                self.current_facts["-contains"].addFact(self.bb.scene_desc["source"],
                                                        self.bb.scene_desc["retained_substance"], silkie.DEFEASIBLE)
            else:
                self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-contains",
                                                                self.bb.scene_desc["retained_substance"],
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

    def __init__(self, bb, doDebug):
        self.spilled_particle_poses: list = []
        self.util_helper = Utils()
        self.marker_array_publisher = rospy.Publisher("/test_markers", MarkerArray, queue_size=10)
        self.dest_points_to_project_to: tuple = ()
        self.src_points_to_project_to: tuple = ()
        self.bb = bb
        self.sim_subscriber = rospy.Subscriber("/mujoco/object_states", ObjectStateArray, self.pose_listener)
        self.contact_data = rospy.Subscriber("/mujoco/contact_data", ContactInfo, self.contact_listener, queue_size=1)
        # self.bounding_box_subscriber = rospy.Subscriber("/mujoco_object_bb", MarkerArray, self.bb_listener)
        self.sim_queries: list = []
        #  variables to update context values
        self.distance: float = 0.0
        self.spilling: bool = False
        self.undershoot: bool = False
        # self.src_orientation: tuple = (0, 0, 0)
        self.object_flow: list = []
        self.spilled_particles: list = []
        self.object_in_dest: int = 0

        # object dependent parameters

        self.src_bounding_box_pose = Pose()
        self.src_bounding_box_dimensions: tuple = (0.0, 0.0, 0.0)
        self.dest_bounding_box_pose = Pose()
        self.dest_bounding_box_dimensions: tuple = (0.0, 0.0, 0.0)
        # self.bb_values_set = False

        self.distance_threshold = (0.0, 0.30)
        # in degrees. greater than [ 76.65427899,  -0.310846  , -34.33960301] along x this lead to pouring
        self.object_flow_threshold = 5  # no.of particles per cycle

        self.source_tilt_angle = 45.0
        self.source_upright_angle = 15.0
        self.dest_upright_angle = 0.0
        self.src_orientation = 0.0
        self.src_direction = 1
        self.dest_orientation = 0.0
        self.dest_direction = 1
        self.dest_goal_reached = False
        self.almost_goal_reached = False  # 75 % goal reached
        self.particle_increase_in_dest = False

        # self.normal_vector = np.array([0, 0, 1])
        self.opening_within = False
        self.direction_vector = (0, 0)  # to make the opening_within true
        self.debug = doDebug
        self.corner_aligned = False
        self.corner_region = ""
        self.distance_from_retained_substance_to_opening = 0.0
        self.distance_of_retained_substance_to_opening_threshold = 0.05

        self.rot_dir = ""
        self.is_src_above = False
        self.collides = False
        self.contact_point = []
        self.count_in_src = 0
        self.retained_substance_count = 0
        self.src_far_above = False

        self.dist_pub = rospy.Publisher('/reasoner/retained_dist', Float64, queue_size=10)

    # def bb_listener(self, data: visualization_msgs.msg.MarkerArray):
    #     src_set = False
    #     dest_set = False
    #     for marker in data.markers:
    #         if marker.ns == self.bb.scene_desc["source"]:
    #             self.src_bounding_box_dimensions = (marker.scale.x, marker.scale.y, marker.scale.z)
    #             self.src_bounding_box_pose = marker.pose
    #             src_set = True
    #         elif marker.ns == self.bb.scene_desc["dest"]:
    #             self.dest_bounding_box_dimensions = (marker.scale.x, marker.scale.y, marker.scale.z)
    #             self.dest_bounding_box_pose = marker.pose
    #             dest_set = True
    #     if src_set and dest_set:
    #         self.bb_values_set = True

    def contact_listener(self, data: ContactInfo):
        # print("in contact listener")
        self.collides = False
        for contact in data.contact_states:
            if (contact.contact_body1 == self.bb.scene_desc["source"] and
                contact.contact_body2 == self.bb.scene_desc["dest"]) or \
                    (contact.contact_body1 == self.bb.scene_desc["dest"] and
                     contact.contact_body2 == self.bb.scene_desc["source"]):
                print("collidessssssssss")
                self.collides = True
                self.contact_point = [contact.contact_point.x, contact.contact_point.y, contact.contact_point.z]
                self.direction_vector = self.util_helper.get_direction_vector(np.array(
                    [self.bb.context_values["dest_pose"].pose.position.x,
                     self.bb.context_values["dest_pose"].pose.position.y]),
                    self.contact_point[0:2])
                break

    def pose_listener(self, req):

        if req.object_states and (rospy.Time(req.header.stamp.secs, req.header.stamp.nsecs) -
                                  rospy.Time(
                                      self.bb.context_values["source_pose"].header.stamp.secs,
                                      self.bb.context_values[
                                          "source_pose"].header.stamp.nsecs)).to_sec() >= 0.05:

            print("pose listener", (rospy.Time(req.header.stamp.secs, req.header.stamp.nsecs) -
                                    rospy.Time(self.bb.context_values["source_pose"].header.stamp.secs,
                                               self.bb.context_values["source_pose"].header.stamp.nsecs)).to_sec())
            spill_count = 0
            count_in_dest = 0
            self.count_in_src = 0
            retained_substance_name = []
            # particle_positions = []
            for obj in req.object_states:
                # print("name ", obj.name)
                if obj.name == self.bb.scene_desc["source"]:
                    # print("source")
                    self.bb.context_values["source_pose"].header.frame_id = self.bb.scene_desc["source"]
                    self.bb.context_values["source_pose"].header.stamp = rospy.Time.now()
                    self.bb.context_values["source_pose"].pose = obj.pose
                    self.src_points_to_project_to = self.util_helper.get_points_to_project(
                        self.bb.scene_desc["source_dim"],
                        self.bb.context_values["source_pose"].pose)

                elif obj.name == self.bb.scene_desc["dest"]:  # Static so sufficient just get it once and not update!
                    # print("dests")
                    self.bb.context_values["dest_pose"].header.frame_id = self.bb.scene_desc["dest"]
                    self.bb.context_values["dest_pose"].header.stamp = \
                        self.bb.context_values["source_pose"].header.stamp
                    self.bb.context_values["dest_pose"].pose = obj.pose
                    self.dest_points_to_project_to = self.util_helper.get_points_to_project(
                        self.bb.scene_desc["dest_dim"],
                        self.bb.context_values["dest_pose"].pose)
            # print(f'src pose: {self.bb.context_values["source_pose"]}',
            #     f'dest pose: {self.bb.context_values["dest_pose"]}')
            # if len(self.object_flow) > 3:
            #     self.object_flow = self.object_flow[-3:]
            self.spilled_particle_poses = []
            if len(self.spilled_particles) > 10:
                self.spilled_particles = self.spilled_particles[-10:]

            src_A, src_B, src_C, src_D = self.src_points_to_project_to
            src_ab = src_B - src_A
            src_ac = src_C - src_A
            src_ad = src_D - src_A

            dest_A, dest_B, dest_C, dest_D = self.dest_points_to_project_to
            dest_ab = dest_B - dest_A
            dest_ac = dest_C - dest_A
            dest_ad = dest_D - dest_A
            for obj in req.object_states:
                if "sync_ball" in obj.name:
                    src_ap = np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z]) - src_A
                    dest_ap = np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z]) - dest_A
                    inside_src = self.util_helper.points_within_bounds_3d(src_ab, src_ac, src_ad, src_ap)

                    # inside_src = inside(self.src_limits[1], self.src_limits[0], obj.pose.position)
                    # inside_dest = inside(self.dest_limits[1], self.dest_limits[0], obj.pose.position)

                    # if not inside_src and not inside_dest:
                    # particle_positions.append([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
                    # count += 1
                    if inside_src:  # ToDo : check if the particle is inside the source has a velocity
                        self.count_in_src += 1
                    elif self.util_helper.points_within_bounds_3d(dest_ab, dest_ac, dest_ad, dest_ap):
                        count_in_dest += 1
                    else:
                        spill_count += 1
                        self.spilled_particle_poses.append([obj.pose.position.x, obj.pose.position.y,
                                                            obj.pose.position.z])
                elif "big_ball" in obj.name:
                    self.retained_substance_count += 1
                    retained_substance_name.append(obj.name)
            print("in source: {}, in dest: {}".format(self.count_in_src, count_in_dest))
            all_particles_not_in_src = sum(self.object_flow)
            current_particle_out = (spill_count + count_in_dest) - all_particles_not_in_src
            self.spilled_particles.append(spill_count)
            if count_in_dest >= self.bb.scene_desc["dest_goal"]:
                self.dest_goal_reached = True
            elif count_in_dest >= 0.8 * self.bb.scene_desc["dest_goal"]:
                self.almost_goal_reached = True
            if current_particle_out >= 0:
                self.object_flow.append(current_particle_out)
            if self.object_in_dest < count_in_dest:
                self.particle_increase_in_dest = True
                self.object_in_dest = count_in_dest
            else:
                self.particle_increase_in_dest = False

            self.distance = math.dist((self.bb.context_values["source_pose"].pose.position.x,
                                       self.bb.context_values["source_pose"].pose.position.y),
                                      # self.bb.context_values["source_pose"].pose.position.z),
                                      (self.bb.context_values["dest_pose"].pose.position.x,
                                       self.bb.context_values["dest_pose"].pose.position.y))
            # self.bb.context_values["dest_pose"].pose.position.z))
            # print("dist {}".format(self.distance))
            if len(self.spilled_particles) >= 2:
                val = self.spilled_particles[-1] - self.spilled_particles[-2]
                spill_count = val if val > 0 else 0
            else:
                spill_count = self.spilled_particles[-1]
            if spill_count > 0.20 * self.bb.scene_desc["total_particles"]:
                self.spilling = True
                print("ooopss spilled.... ", spill_count)
            else:
                self.spilling = False
                print("no spilling ", spill_count)

            print(
                f"count spill:{spill_count}, spilled_particles:{self.spilled_particles}, object out:{self.object_flow}")

            # source upright
            self.src_direction, self.src_orientation = self.util_helper.get_direction_and_orientation(
                self.bb.scene_desc["source_dim"][2],
                self.bb.context_values["source_pose"].pose)

            # dest upright
            self.dest_direction, self.dest_orientation = self.util_helper.get_direction_and_orientation(
                self.bb.scene_desc["dest_dim"][2],
                self.bb.context_values["dest_pose"].pose)

            # print("pose ", self.bb.context_values["source_pose"].pose)
            # print(f'q :{self.bb.context_values["source_pose"].pose.orientation}, ANGLEEEE:{angle}, '
            #       f'point:{point_cup_bottom}, rotated_pt:{point_map_bottom},'
            #       f' src_vector:{src_vector}')

            # compute above (in dest frame)
            if self.util_helper.is_above(self.bb.context_values["source_pose"].pose,
                                         0.75 * self.bb.scene_desc["source_dim"][2],
                                         self.bb.context_values["dest_pose"].pose,
                                         self.bb.scene_desc["dest_dim"][2],
                                         self.bb.scene_desc["dest"]):
                self.is_src_above = True
            else:
                self.is_src_above = False
            # compute opening within or not
            # print(f'src_Pose: {self.bb.context_values["source_pose"].pose}, src_dim: {self.bb.scene_desc[
            # "source_dim"]}, ' f'dest_pose: {self.bb.context_values["dest_pose"].pose}, dest_dim: {
            # self.bb.scene_desc["dest_dim"]}')
            corner_points = False
            if self.bb.scene_desc["sourceHasEdges"]:
                corner_points = True
            within, closest_point = self.util_helper.is_source_opening_within(
                self.bb.context_values["source_pose"].pose,
                self.bb.scene_desc["source_dim"],
                self.bb.context_values["dest_pose"].pose,
                self.bb.scene_desc["dest_dim"],
                corner_points)
            if within:
                self.opening_within = True
            # print("opening within")
            else:
                # dest_src
                # print("dir vector ", dest_opening_point, src_opening_point)
                self.direction_vector = self.util_helper.get_direction_vector(np.array(
                    [self.bb.context_values["dest_pose"].pose.position.x,
                     self.bb.context_values["dest_pose"].pose.position.y]), closest_point[0:2])
                self.opening_within = False

            if self.bb.scene_desc["sourceHasEdges"]:
                self.corner_aligned, self.rot_dir = self.util_helper.is_corner_aligned(
                    self.bb.context_values["source_pose"].pose.position,
                    self.bb.scene_desc["source_dim"][2],
                    self.bb.context_values["dest_pose"].pose.position,
                    self.bb.scene_desc["dest_dim"][2],
                    closest_point)

                pt = closest_point - np.array([self.bb.context_values["source_pose"].pose.position.x,
                                               self.bb.context_values["source_pose"].pose.position.y,
                                               self.bb.context_values["source_pose"].pose.position.z])
                temp_str = ""
                if pt[0] > 0:
                    temp_str += "1"
                else:
                    temp_str += "-1"
                temp_str += ","
                if pt[1] > 0:
                    temp_str += "1"
                else:
                    temp_str += "-1"

                self.corner_region = self.bb.corner_regions[temp_str]
            else:
                self.corner_aligned = False
                self.corner_region = None
                self.rot_dir = None

            # detect undershooting
            num_of_particles_in_src_boundary = self.util_helper.get_particles_in_src_spilling_boundary(
                np.array([self.bb.context_values["source_pose"].pose.position.x,
                          self.bb.context_values["source_pose"].pose.position.y]),
                radius=self.bb.scene_desc["source_dim"][2] * 1.5,
                particle_poses=self.spilled_particle_poses)

            print(f'particles in src boundary :{num_of_particles_in_src_boundary}')

            if num_of_particles_in_src_boundary >= 0.2 * self.spilled_particles[-1]:
                self.undershoot = True
            else:
                self.undershoot = False

            top_src_point = self.bb.context_values["source_pose"].pose.position.z + \
                            self.bb.scene_desc["source_dim"][2] / 2
            top_dest_point = self.bb.context_values["dest_pose"].pose.position.z + \
                             self.bb.scene_desc["dest_dim"][2] / 2
            print("src heights ", (top_src_point - top_dest_point), self.bb.context_values["srcHeightLimits"])

            if (top_src_point - top_dest_point) > self.bb.context_values["srcHeightLimits"]:
                self.src_far_above = True
            else:
                self.src_far_above = False

            print("src far above ", self.src_far_above)

            self.distance_from_retained_substance_to_opening = self.util_helper.get_distance_to_retained_object(
                 lies_along="y",
                 direction="+x",
                 obj_pose=self.bb.context_values["source_pose"].pose,
                 ref_name=self.bb.scene_desc["source"],
                 obj_dim=self.bb.scene_desc["source_dim"],
                 obj_name=retained_substance_name)
            msg = Float64()
            msg.data = self.distance_from_retained_substance_to_opening
            self.dist_pub.publish(msg)
            print("distance_from_retained_substance ", self.distance_from_retained_substance_to_opening)

        if self.debug:
            self.publish_test_markers()

    def publish_test_markers(self):
        self.marker_array_publisher.publish(self.util_helper.get_test_visualization_marker_array())
        self.util_helper.test_marker_array.markers = []

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

        if self.retained_substance_count >= 0.8*self.bb.scene_desc["retained_substance_count"]:
            self.bb.context_values["sourceContainsRetainedSubstance"] = True
        else:
            self.bb.context_values["sourceContainsRetainedSubstance"] = False

        if self.count_in_src == 0:
            self.bb.context_values["sourceEmpty"] = True
        else:
            self.bb.context_values["sourceEmpty"] = False

        # spilling
        if self.spilling:
            self.bb.context_values["isSpilled"] = True
        else:
            self.bb.context_values["isSpilled"] = False

        # spilling now
        if np.count_nonzero(np.diff(self.spilled_particles[-5:]) > 0) >= 3:
            self.bb.context_values["isSpilling"] = True
        else:
            self.bb.context_values["isSpilling"] = False

        # tilted
        if self.src_orientation >= self.source_tilt_angle:
            self.bb.context_values['isTilted'] = True
        else:
            self.bb.context_values['isTilted'] = False

        # upright
        if self.src_direction > 0 and self.src_orientation <= self.source_upright_angle:
            self.bb.context_values["sourceUpright"] = True
        else:
            self.bb.context_values["sourceUpright"] = False

        # dest upright
        if not (self.dest_direction > 0 and np.isclose(self.dest_orientation, self.dest_upright_angle, atol=0.5)):
            self.bb.context_values["isDestTilted"] = True
        else:
            self.bb.context_values["isDestTilted"] = False

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

        if obj_avg > 10:
            self.bb.context_values["tooFast"] = True
            print("fast true")
        else:
            self.bb.context_values["tooFast"] = False

        if self.dest_goal_reached:
            self.bb.context_values["dest_goal_reached"] = True
        else:
            self.bb.context_values["dest_goal_reached"] = False

        # corner aligned
        if self.corner_aligned:
            self.bb.context_values["srcCornerAligned"] = True
        else:
            self.bb.context_values["srcCornerAligned"] = False

        # corner region
        if self.corner_region:
            self.bb.context_values["srcCornerSpoutRegion"] = self.corner_region
        else:
            self.bb.context_values["srcCornerSpoutRegion"] = ""

        # rot direction
        if self.rot_dir:
            self.bb.context_values["rotationDirection"] = self.rot_dir
            self.rot_dir = ""
        else:
            self.bb.context_values["rotationDirection"] = ""

        # src above
        if self.is_src_above:
            self.bb.context_values["srcAbove"] = True
        else:
            self.bb.context_values["srcAbove"] = False

        # src far above
        if self.src_far_above:
            self.bb.context_values["srcFarAbove"] = True
        else:
            self.bb.context_values["srcFarAbove"] = False

        if self.opening_within:
            self.bb.context_values["hasOpeningWithin"] = True
        else:
            self.bb.context_values["hasOpeningWithin"] = False
            self.bb.context_values["locationOfSourceRelativeToDestination"] = \
                self.util_helper.get_direction_relative_to_dest(self.direction_vector)

        if self.collides:
            self.bb.context_values["collides"] = True
        else:
            self.bb.context_values["collides"] = False

        if self.bb.context_values["collides"] and self.bb.context_values["hasOpeningWithin"]:
            self.bb.context_values["locationOfSourceRelativeToDestination"] = \
                self.util_helper.get_direction_relative_to_dest(self.direction_vector)

        print(f'direction: {self.direction_vector}, '
              f'LOCATION  : {self.bb.context_values["locationOfSourceRelativeToDestination"]}')

        # undershoot
        if self.undershoot:
            self.bb.context_values["undershoot"] = True
        else:
            self.bb.context_values["undershoot"] = False
        # retained_substance_close_to_opening
        if self.distance_from_retained_substance_to_opening <= self.distance_of_retained_substance_to_opening_threshold:
            self.bb.context_values["retainedSubstanceCloseToOpening"] = True
        else:
            self.bb.context_values["retainedSubstanceCloseToOpening"] = False

        self.collides = False

    def query(self, queries: list):
        # Just compute the queries here. the variables you use should come from context values which is already set
        # moves up - increasing particles by time
        # almost full - 80 % goal reached
        for queryStr in queries:
            query = queryStr.split("(")[0]
            if query == "movesUpIn":
                self.bb.context_values[query] = self.particle_increase_in_dest
            if query == "almostGoalReached":
                self.bb.context_values[query] = self.almost_goal_reached
            if query == "poursTo":
                self.bb.context_values[query] = self.bb.context_values[query]


class Perception:
    pass


if __name__ == '__main__':
    rospy.init_node('reasoner_sim_node')

    parser = argparse.ArgumentParser(prog="pouring_reasoner", description="Visualize theory",
                                     epilog="")
    parser.add_argument('-v', '--visualize', action='store_true', help='Enable visualization.')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable visualizing markers.')
    parser.add_argument('-s', '--storeplots', default='', help='Prefix of filenames to store visualizations in.')

    arguments = parser.parse_args()
    doVisualize = arguments.visualize
    debug = arguments.debug

    blackboard = Blackboard()
    blackboard.add_experts(SimulationSource(blackboard, debug))

    controller = BlackboardController(blackboard, visualize=doVisualize)

    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        controller.update_context()
        controller.get_consequents()
        rate.sleep()
