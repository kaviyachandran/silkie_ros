"""
1. Create a skeleton with blackboard class, Experts like simulation, perception and other process can update the
context values.
2. A controller that publishes messages to Giskard based on the current context
3. Models / objects are assumed to be with length along x-axis, depth along y-axis


"""

import math
from typing import Dict, Any

import numpy as np
import silkie

from utils import Utils

import rospy
import visualization_msgs
from geometry_msgs.msg import PoseStamped, Point, Pose
from mujoco_msgs.msg import ObjectStateArray
from std_msgs.msg import String
import tf
from tf.transformations import euler_from_quaternion, quaternion_matrix
from visualization_msgs.msg import MarkerArray


class Blackboard(object):
    # While adding new scenes, change name, dimensions

    def __init__(self):
        self.experts = []
        self.scene_desc = {
            "source": "free_cup",
            "dest": "free_cup2",
            "source_type": "Container",
            "dest_type": "Container",
            "poured_substance_type": "Liquid",  # changing Thing to Liquid
            "poured_substance": "particles",
            "total_particles": 200,
            "source_dim": (),
            "dest_dim": (),
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
            'isSpilled': False,
            'collision': False,
            'dir_overshoot': "",  # todo: compute the orientation w.r.t dest pose to find the direction
            "source_pose": PoseStamped(),
            "dest_pose": PoseStamped(),
            "dest_goal_reached": False,
            "sourceUpright": True,
            "hasOpeningWithin": False,
            "movesUpIn": False,
            "almostGoalReached": False,
            # - (0, -1) ---> Behind. move upwards. in +y direction
            # - (0, 1) ---> FrontOf. move downwards. in -y direction
            # - (1, 0) ---> Right. move left. in -x direction
            # - (-1, 0) ---> Left. move right. in +x direction
            "locationOfSourceRelativeToDestination": [],
        }
        # dimension_data = rospy.wait_for_message("/mujoco_object_bb", MarkerArray, timeout=5)
        # if dimension_data:
        #     self.set_dimension_values(dimension_data)
        # else:
        self.scene_desc["source_dim"] = (0.0646, 0.0646, 0.18)
        self.scene_desc["dest_dim"] = (0.0646, 0.0646, 0.18)


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

    def __init__(self, bb):
        self.concluded_behavior_publisher = rospy.Publisher("/reasoner/concluded_behaviors", String, queue_size=10)

        self.bb_obj = bb
        self.reasoner = Reasoner(bb)
        self.queries_for_experts = []

    def update_context(self) -> None:
        print("controller")
        for expert in self.bb_obj.experts:
            expert.update()
            expert.query(self.queries_for_experts)

    def get_consequents(self):
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
        facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "contains",
                                           self.bb.scene_desc["poured_substance"], silkie.DEFEASIBLE))
        facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "SourceRole", "", silkie.DEFEASIBLE))
        facts.update(Reasoner.create_facts(self.bb.scene_desc["dest"], "DestinationRole", "", silkie.DEFEASIBLE))

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
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "upright", "",
                                                            silkie.DEFEASIBLE))
        else:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-upright", "",
                                                            silkie.DEFEASIBLE))

        if self.bb.context_values["hasOpeningWithin"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "hasOpeningWithin",
                                                            self.bb.scene_desc["dest"], silkie.DEFEASIBLE))
        elif not self.bb.context_values["hasOpeningWithin"]:
            self.current_facts.update(Reasoner.create_facts(self.bb.scene_desc["source"], "-hasOpeningWithin",
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
        self.util_helper = Utils()
        self.marker_array_publisher = rospy.Publisher("/test_markers", MarkerArray, queue_size=10)
        self.dest_limits: tuple = ()
        self.src_limits: tuple = ()
        self.tf_transform = tf.TransformListener()
        self.bb = bb
        self.sim_subscriber = rospy.Subscriber("/mujoco/object_states", ObjectStateArray, self.pose_listener)
        self.bounding_box_subscriber = rospy.Subscriber("/mujoco_object_bb", MarkerArray, self.bb_listener)
        self.sim_queries: list = []
        #  variables to update context values
        self.distance: float = 0.0
        self.spilling: bool = False
        # self.src_orientation: tuple = (0, 0, 0)
        self.object_flow: list = []
        self.object_in_dest: int = 0
        # object dependent parameters

        self.src_bounding_box_pose = Pose()
        self.src_bounding_box_dimensions: tuple = (0.0, 0.0, 0.0)
        self.dest_bounding_box_pose = Pose()
        self.dest_bounding_box_dimensions: tuple = (0.0, 0.0, 0.0)
        self.bb_values_set = False

        self.distance_threshold = (0.0, 0.30)
        # in degrees. greater than [ 76.65427899,  -0.310846  , -34.33960301] along x this lead to pouring
        self.object_flow_threshold = 10  # no.of particles per cycle

        self.source_tilt_angle = 45.0
        self.source_upright_angle = 10.0
        self.cup_orientation = 0.0
        self.cup_direction = 1
        self.dest_goal_reached = False
        self.almost_goal_reached = False  # 75 % goal reached
        self.particle_increase_in_dest = False

        self.normal_vector = np.array([0, 0, 1])
        self.opening_within = False
        self.direction_vector = (0, 0)  # to make the opening_within true
        self.debug = False

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
                                                                 "source_pose"].header.stamp.nsecs)).to_sec() >= 0.09:

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
            if count_in_dest >= self.bb.scene_desc["dest_goal"]:
                self.dest_goal_reached = True
            elif count_in_dest >= 0.8 * self.bb.scene_desc["dest_goal"]:
                self.almost_goal_reached = True
            if current_particle_out >= 0:
                self.object_flow.append(current_particle_out)
            if self.object_in_dest < count_in_dest:
                self.particle_increase_in_dest = True
                self.object_in_dest = count_in_dest

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
            # print("dist {}".format(self.distance))

            if count > 0.20 * self.bb.scene_desc["total_particles"]:
                self.spilling = True
                print("ooopss spilled.... ", count)
            else:
                print("no spilling ", count)

            # upright
            # tf_wrist_cup = tf.lookupTransform(self.bb.scene_desc['source'], 'wrist_roll_link', rospy.Time())

            point_cup_bottom = np.array([0.,
                                         0.,
                                         -self.bb.scene_desc["source_dim"][2] / 2,
                                         1])

            tf_map_cup = quaternion_matrix(np.array([self.bb.context_values["source_pose"].pose.orientation.x,
                                                     self.bb.context_values["source_pose"].pose.orientation.y,
                                                     self.bb.context_values["source_pose"].pose.orientation.z,
                                                     self.bb.context_values["source_pose"].pose.orientation.w]))
            tf_map_cup[:, 3] = np.array([self.bb.context_values["source_pose"].pose.position.x,
                                         self.bb.context_values["source_pose"].pose.position.y,
                                         self.bb.context_values["source_pose"].pose.position.z, 1])

            # rotated_point
            point_map_bottom = np.matmul(tf_map_cup, point_cup_bottom)

            src_vector = np.array([self.bb.context_values["source_pose"].pose.position.x,
                                   self.bb.context_values["source_pose"].pose.position.y,
                                   self.bb.context_values["source_pose"].pose.position.z]) - point_map_bottom[:3]

            self.cup_direction = np.dot(self.normal_vector, src_vector)
            self.cup_orientation = np.degrees(np.arccos(self.cup_direction / np.linalg.norm(src_vector)))

            # print("pose ", self.bb.context_values["source_pose"].pose)
            # print(f'q :{self.bb.context_values["source_pose"].pose.orientation}, ANGLEEEE:{angle}, '
            #       f'point:{point_cup_bottom}, rotated_pt:{point_map_bottom},'
            #       f' src_vector:{src_vector}')

            # compute opening within or not

            # print(f'src_Pose: {self.bb.context_values["source_pose"].pose}, src_dim: {self.bb.scene_desc[
            # "source_dim"]}, ' f'dest_pose: {self.bb.context_values["dest_pose"].pose}, dest_dim: {
            # self.bb.scene_desc["dest_dim"]}')
            within, closest_point = self.util_helper.is_source_opening_within(self.bb.context_values["source_pose"].pose,
                                                                              self.bb.scene_desc["source_dim"],
                                                                              self.bb.context_values["dest_pose"].pose,
                                                                              self.bb.scene_desc["dest_dim"])
            if within:
                self.opening_within = True
                # print("opening within")
            else:
                # dest_src
                # print("dir vector ", dest_opening_point, src_opening_point)
                v_src_dest = np.array(
                    [self.bb.context_values["dest_pose"].pose.position.x,
                     self.bb.context_values["dest_pose"].pose.position.y]) - closest_point[0:2]
                self.direction_vector = v_src_dest / np.linalg.norm(v_src_dest)

        if self.debug:
            self.publish_test_markers()

    def publish_test_markers(self):
        self.marker_array_publisher.publish(self.util_helper.get_test_visualization_marker_array())

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

        # spilling
        if self.spilling:
            self.bb.context_values["isSpilled"] = True
        else:
            self.bb.context_values["isSpilled"] = False

        # tilted
        if self.cup_orientation >= self.source_tilt_angle:
            self.bb.context_values['isTilted'] = True
        else:
            self.bb.context_values['isTilted'] = False

        # upright
        if self.cup_direction > 0 and self.cup_orientation <= self.source_upright_angle:
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

        if self.opening_within:
            self.bb.context_values["hasOpeningWithin"] = True
        else:
            self.bb.context_values["hasOpeningWithin"] = False
            coordinate = np.argsort(self.direction_vector)[::-1]  # descending order

            for index in coordinate:
                if index == 0:  # Along x-axis w.r.t map
                    if self.direction_vector[index] > 0:
                        self.bb.context_values["locationOfSourceRelativeToDestination"].append("inFront")
                    elif self.direction_vector[index] < 0:
                        self.bb.context_values["locationOfSourceRelativeToDestination"].append("behind")
                else:  # Along y-axis w.r.t map
                    if self.direction_vector[index] > 0:
                        self.bb.context_values["locationOfSourceRelativeToDestination"].append("left")
                    elif self.direction_vector[index] < 0:
                        self.bb.context_values["locationOfSourceRelativeToDestination"].append("right")
            print(f'direction: {self.direction_vector}, '
                  f'LOCATION  : {self.bb.context_values["locationOfSourceRelativeToDestination"]}')

    def query(self, queries: list):
        # Just compute the queries here. the variables you use should come from context values which is already set
        # moves up - increasing particles by time
        # almost full - 80 % goal reached
        for query in queries:
            if query == "movesUpIn":
                self.bb.context_values[query] = self.particle_increase_in_dest
            elif query == "almostGoalReached":
                self.bb.context_values[query] = self.almost_goal_reached
            elif query == "poursTo":
                self.bb.context_values[query] = self.bb.context_values[query]


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
        controller.get_consequents()
        rate.sleep()
