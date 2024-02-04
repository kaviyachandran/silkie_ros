import numpy as np
from utils import Utils
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import MarkerArray
import rospy

import pytest


def test_closest_point_on_rectangle_to_point1():
    u = Utils()
    container_pose = Pose()
    container_pose.position.x = 2
    container_pose.position.y = 0.5
    container_pose.position.z = 1
    container_pose.orientation.w = 1

    container_dim = (1, 1, 1)
    pose_pot_obj = Point()
    pose_pot_obj.x = 0.5
    pose_pot_obj.y = 0.5
    pose_pot_obj.z = -0.5

    closest_pt, distance = u.closest_point_on_rectangle_to_point(container_pose, container_dim, pose_pot_obj)

    expected_closest_pt = np.array([0.5, 0.5, 0.5])
    expected_distance = 1
    assert np.allclose(closest_pt, expected_closest_pt)
    assert np.isclose(distance, expected_distance)


def test_closest_point_on_rectangle_to_point2():
    u = Utils()
    container_pose = Pose()
    container_pose.position.x = 2
    container_pose.position.y = 0.5
    container_pose.position.z = 1
    container_pose.orientation.z = 0.3826
    container_pose.orientation.w = 0.9238

    container_dim = (1, 1, 1)
    pose_pot_obj = Point()
    pose_pot_obj.x = 0.5
    pose_pot_obj.y = 0.5
    pose_pot_obj.z = -0.5

    closest_pt, distance = u.closest_point_on_rectangle_to_point(container_pose, container_dim, pose_pot_obj)

    expected_closest_pt = np.array([0.3535, 0.3535, 0.5])
    expected_distance = 1.021
    assert np.allclose(closest_pt, expected_closest_pt)
    assert np.isclose(distance, expected_distance)


def test_opening_within1():
    # rospy.init_node('reasoner_sim_node')
    # pub = rospy.Publisher('/test_marker', MarkerArray, queue_size=10)
    map_Pose_src = ([1.9, -0.6054, 0.38481],
                     [0, 0.38268, 0, 0.92388])
    src_dim = (0.0646, 0.0646, 0.18)
    dest_pose = Pose()
    dest_pose.position.x = 1.9998643009882282
    dest_pose.position.y = -0.6001155389875352
    dest_pose.position.z = 0.30538641729671684
    dest_pose.orientation.x = -3.328273309018623e-06
    dest_pose.orientation.y = 1.298903533812593e-05
    dest_pose.orientation.z = -2.3896462101018977e-06
    dest_pose.orientation.w = 0.9999999999072484
    dest_dim = (0.0646, 0.0646, 0.18)

    u = Utils()
    within = u.is_source_opening_within(map_Pose_src, src_dim, dest_pose, dest_dim)
    # vis_array = u.get_test_visualization_marker_array()
    # pub.publish(vis_array)

    assert within == True


def test_opening_within2():
    # rospy.init_node('reasoner_sim_node')
    # pub = rospy.Publisher('/test_marker', MarkerArray, queue_size=10)
    map_Pose_src = ([1.9, -0.7054, 0.38481],
                     [0, 0.38268, 0, 0.92388])
    src_dim = (0.0646, 0.0646, 0.18)
    dest_pose = Pose()
    dest_pose.position.x = 1.9998643009882282
    dest_pose.position.y = -0.6001155389875352
    dest_pose.position.z = 0.30538641729671684
    dest_pose.orientation.x = -3.328273309018623e-06
    dest_pose.orientation.y = 1.298903533812593e-05
    dest_pose.orientation.z = -2.3896462101018977e-06
    dest_pose.orientation.w = 0.9999999999072484
    dest_dim = (0.0646, 0.0646, 0.18)

    u = Utils()
    within = u.is_source_opening_within(map_Pose_src, src_dim, dest_pose, dest_dim)
    # vis_array = u.get_test_visualization_marker_array()
    # pub.publish(vis_array)

    assert within == False
