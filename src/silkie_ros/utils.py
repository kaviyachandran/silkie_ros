import numpy as np
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray


def rotate_point(point_to_rotate: np.array, rot_matrix: np.array) -> np.array:
    return np.matmul(rot_matrix[:3, :3], point_to_rotate)


def closest_point_on_rectangle_to_point(container_pose: Pose, container_dim: tuple, pot_P_obj: Point) -> (
        np.array, float):
    # rospy.init_node('test_utils')
    A, B, C = _get_points(container_pose, container_dim)
    P = np.array([pot_P_obj.x, pot_P_obj.y, pot_P_obj.z])
    ab = B - A
    ac = C - A
    ap = P - A

    ab_len = np.linalg.norm(ab)
    ab_unit = ab / ab_len
    ac_len = np.linalg.norm(ac)
    ac_unit = ac / ac_len

    transformed_pt = [np.dot(ac_unit, ap / ac_len), np.dot(ab_unit, ap / ab_len)]
    projected_pt = np.clip(transformed_pt, 0, 1)

    closest_pt = A + projected_pt[0] * ac + projected_pt[1] * ab
    # print(closest_pt, np.linalg.norm(P - closest_pt))
    # pub = rospy.Publisher('/test_marker_array', MarkerArray, queue_size=1, latch=True)
    # m = MarkerArray()
    # m.markers.append(
    #     _create_vis_marker(parent_frame='pot', ns='A', obj_type=2, action=0, color=(1, 1, 0), lifetime=0, position=A,
    #                        size=(0.01, 0.01, 0.01)))
    # m.markers.append(
    #     _create_vis_marker(parent_frame='pot', ns='B', obj_type=2, action=0, color=(1, 1, 0), lifetime=0, position=B,
    #                        size=(0.01, 0.01, 0.01)))
    # m.markers.append(
    #     _create_vis_marker(parent_frame='pot', ns='C', obj_type=2, action=0, color=(1, 1, 0), lifetime=0, position=C,
    #                        size=(0.01, 0.01, 0.01)))
    # m.markers.append(_create_vis_marker(parent_frame='pot', ns='P', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
    #                                     position=closest_pt, size=(0.01, 0.01, 0.01)))
    # m.markers.append(_create_vis_marker(parent_frame='map', ns='Pot', obj_type=1, action=0, color=(0, 1, 1), lifetime=0,
    #                                     position=(container_pose.position.x, container_pose.position.y,
    #                                               container_pose.position.z),
    #                                     orientation=container_pose.orientation,
    #                                     size=container_dim))
    # pub.publish(m)
    # rospy.spin()

    return closest_pt, np.linalg.norm(P - closest_pt)  # closest pt is w.r.t local frame and not global frame


def closest_point_on_line_to_point(A, B, P) -> float:
    ab = B - A
    ap = P - A

    ab_len = np.linalg.norm(ab)
    ab_unit = ab / ab_len

    transformed_pt = [np.dot(ab_unit, ap / ab_len)]
    projected_pt = np.clip(transformed_pt, 0, 1)
    closest_pt = A + projected_pt * ab
    # print(closest_pt, np.linalg.norm(P - closest_pt))

    return np.linalg.norm(P - closest_pt)


def get_distance_to_retained_object(obj_pose: Pose, obj_dim: tuple, pot_P_obj: Point, lies_along: str,
                                    direction: str):
    A, B = _get_points_on_line(lies_along, direction, obj_pose, obj_dim)
    P = np.array([pot_P_obj.x, pot_P_obj.y, pot_P_obj.z])
    return closest_point_on_line_to_point(A, B, P)


def is_source_opening_within(src_pose: Pose, src_dim: tuple, dest_pose: Pose, dest_dim: tuple) -> bool:
    # This is computed with the assumption that the length of the obj is along x-axis, depth / breadth along y-axis
    src_opening_point = np.array(
        [src_pose.position.x, src_pose.position.y, src_pose.position.z + src_dim[2] / 2])
    src_A = np.array([src_opening_point[0], src_opening_point[1] + src_dim[1] / 2, src_opening_point[2]])
    src_B = np.array([src_opening_point[0], src_opening_point[1] - src_dim[1] / 2, src_opening_point[2]])
    src_C = np.array([src_opening_point[0] + src_dim[0] / 2, src_opening_point[1], src_opening_point[2]])
    src_D = np.array([src_opening_point[0] - src_dim[0] / 2, src_opening_point[1], src_opening_point[2]])

    rotation_mat = quaternion_matrix(np.array([src_pose.orientation.x, src_pose.orientation.y, src_pose.orientation.z,
                                               src_pose.orientation.w]))
    src_opening_point = rotate_point(src_opening_point, rotation_mat)
    src_A = rotate_point(src_A, rotation_mat)
    src_B = rotate_point(src_B, rotation_mat)
    src_C = rotate_point(src_C, rotation_mat)
    src_D = rotate_point(src_D, rotation_mat)

    dest_opening_point = np.array(
        [dest_pose.position.x,
         dest_pose.position.y,
         dest_pose.position.z + dest_dim[2] / 2])
    # self.dest_bounding_box_pose.position.z + self.bb.scene_desc["dest_dim"][2] / 2)
    # 0.75 of l and d is to keep the rectangle smaller than the bounding box.
    # To ensure the source opening lies within the dest
    l, d, h = (dest_dim[0] * 0.75) / 2, (dest_dim[1] * 0.75) / 2, dest_opening_point[2]
    a = np.array([dest_opening_point[0] - l, dest_opening_point[1] + d], h)
    b = np.array([dest_opening_point[0] - l, dest_opening_point[1] - d], h)
    c = np.array([dest_opening_point[0] + l, dest_opening_point[1] + d], h)

    ab = b - a
    ac = c - a

    # print("points to compute opening ", a, b, c, src_opening_point)
    # print("vectors ", ab, ap, ac)
    # print("lengths :", np.dot(ap, ab), np.dot(ab, ab), np.dot(ap, ac), np.dot(ac, ac))

    return point_within_bounds(ab, ac, src_opening_point - a) or point_within_bounds(ab, ac, src_A - a) or \
           point_within_bounds(ab, ac, src_B - a) or point_within_bounds(ab, ac, src_C - a) or \
           point_within_bounds(ab, ac, src_D - a)


def _get_points_on_line(lies_along: str, direction: str, obj_pose: Pose, obj_dim: tuple):
    x_val = obj_dim[0] / 2
    y_val = obj_dim[1] / 2
    z_val = obj_dim[2] / 2

    A = []
    B = []
    if lies_along == "y":
        if direction == "+x":
            A = np.array([x_val, -y_val, z_val])
            B = np.array([x_val, y_val, z_val])
        elif direction == "-x":
            A = np.array([-x_val, -y_val, z_val])
            B = np.array([-x_val, y_val, z_val])

    rot_matrix = quaternion_matrix(np.array([obj_pose.orientation.x, obj_pose.orientation.y, obj_pose.orientation.z,
                                             obj_pose.orientation.w]))
    A = rotate_point(A, rot_matrix)
    B = rotate_point(B, rot_matrix)

    return A, B


def _get_points(obj_pose: Pose, obj_dim):
    # top lies in XY plane. points in local frame
    # A is bottom left
    A = np.array([obj_dim[0] / 2, -obj_dim[1] / 2, obj_dim[2] / 2])
    B = np.array([obj_dim[0] / 2, obj_dim[1] / 2, obj_dim[2] / 2])
    C = np.array([-obj_dim[0] / 2, -obj_dim[1] / 2, obj_dim[2] / 2])

    rot_matrix = quaternion_matrix(np.array([obj_pose.orientation.x, obj_pose.orientation.y,
                                             obj_pose.orientation.z, obj_pose.orientation.w]))
    A = rotate_point(A, rot_matrix)
    B = rotate_point(B, rot_matrix)
    C = rotate_point(C, rot_matrix)

    return A, B, C


def point_within_bounds(ab, ac, ap) -> bool:
    # projecting the point src_opening_point on the ab and ac vectors. ab perpendicular to  ac
    return 0 < abs(np.dot(ap, ab)) < abs(np.dot(ab, ab)) and 0 < abs(np.dot(ap, ac)) < abs(np.dot(ac, ac))


def _create_vis_marker(parent_frame, ns, obj_type, action, color, lifetime, position, size,
                       orientation=Quaternion(0, 0, 0, 1)) -> Marker:
    marker = Marker()
    marker.header.frame_id = parent_frame
    # marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = 1
    # marker.lifetime = lifetime
    marker.type = obj_type
    marker.action = action
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = orientation.x
    marker.pose.orientation.y = orientation.y
    marker.pose.orientation.z = orientation.z
    marker.pose.orientation.w = orientation.w
    marker.scale.x = size[0]
    marker.scale.y = size[1]
    marker.scale.z = size[2]
    marker.color.a = 0.5
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    return marker

# if __name__ == '__main__':
# 	container_pose = Pose()
# 	container_pose.position.x = 2
# 	container_pose.position.y = -0.1998
# 	container_pose.position.z = 0.670
# 	# container_pose.orientation.w = 1
# 	container_pose.orientation.x = 0
# 	container_pose.orientation.y = 0.38268
# 	container_pose.orientation.z = 0
# 	container_pose.orientation.w = 0.92387
# 	container_dim = (0.2089, 0.2089, 0.06)
# 	pose_pot_obj = Point()
# 	pose_pot_obj.x = 0.043
# 	pose_pot_obj.y = -0.052
# 	pose_pot_obj.z = 0.001
#
# 	# closest_point_on_rectangle_to_point(container_pose, container_dim, pose_pot_obj)
# 	get_distance_to_retained_object(container_pose, container_dim, pose_pot_obj)
