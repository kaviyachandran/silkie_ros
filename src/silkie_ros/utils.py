import numpy as np
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Point, TransformStamped
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray
import tf


## Note: Rotate in local frame

class Utils(object):

    def __init__(self):
        self.test_marker_array = MarkerArray()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

    def get_test_visualization_marker_array(self):
        return self.test_marker_array

    def rotate_point(self, point_to_rotate: np.array, rot_matrix: np.array) -> np.array:
        return np.dot(rot_matrix[:3, :3], point_to_rotate)

    def closest_point_on_rectangle_to_point(self, container_pose: Pose, container_dim: tuple, pot_P_obj: Point) -> (
            np.array, float):
        # rospy.init_node('test_utils')
        A, B, C, _ = self._get_corner_points(container_pose.orientation, container_dim)
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

    def closest_point_on_line_to_point(self, A, B, P) -> float:
        ab = B - A
        ap = P - A

        ab_len = np.linalg.norm(ab)
        ab_unit = ab / ab_len

        transformed_pt = [np.dot(ab_unit, ap / ab_len)]
        projected_pt = np.clip(transformed_pt, 0, 1)
        closest_pt = A + projected_pt * ab
        # print(closest_pt, np.linalg.norm(P - closest_pt))

        return np.linalg.norm(P - closest_pt)

    def get_distance_to_retained_object(self, obj_pose: Pose, obj_dim: tuple, pot_P_obj: Point, lies_along: str,
                                        direction: str):
        A, B = self._get_points_on_line(lies_along, direction, obj_pose, obj_dim)
        P = np.array([pot_P_obj.x, pot_P_obj.y, pot_P_obj.z])
        return self.closest_point_on_line_to_point(A, B, P)

    def is_source_opening_within(self, src_pose: Pose, src_dim: tuple, dest_pose: Pose, dest_dim: tuple, corner: bool) \
            -> (bool, tuple):
        # print("src pose ", map_Pose_src)
        # opening_within = False
        src_A = np.zeros(3)
        src_B = np.zeros(3)
        src_C = np.zeros(3)
        src_D = np.zeros(3)
        # This is computed with the assumption that the length of the obj is along x-axis, depth / breadth along y-axis
        if corner:
            src_A, src_B, src_C, src_D = self._get_corner_points(src_pose.orientation, (src_dim[0] * 0.75,
                                                                                        src_dim[1] * 0.75,
                                                                                        src_dim[2]))
        elif not corner:
            src_A, src_B, src_C, src_D = self._get_edge_points(src_pose.orientation, (src_dim[0] * 0.75,
                                                                                      src_dim[1] * 0.75,
                                                                                      src_dim[2]))

        # src_x = src_dim[0] / 2 * .75
        # src_y = src_dim[1] / 2 * .75
        # src_z = src_dim[2] / 2
        # src_opening_point = np.array([0, 0, src_z])
        # src_A = np.array([0, src_y, src_z])
        # src_B = np.array([0, -src_y, src_z])
        # src_C = np.array([src_x, 0, src_z])
        # src_D = np.array([-src_x, 0, src_z])

        # print("src points ", src_A, src_B, src_C, src_D, src_opening_point)

        # rotation_mat = quaternion_matrix(
        #     np.array([src_pose.orientation.x, src_pose.orientation.y, src_pose.orientation.z,
        #               src_pose.orientation.w]))
        #
        # tf_map_src = np.hstack((rotation_mat[:3, :3],
        #                         np.array([src_pose.position.x, src_pose.position.y,
        #                                   src_pose.position.z]).reshape(3, 1)))
        # tf_map_src = np.vstack((tf_map_src, np.array([0, 0, 0, 1]).reshape(1, 4)))

        map_P_src_A = src_A + np.array([src_pose.position.x, src_pose.position.y, src_pose.position.z])
        map_P_src_B = src_B + np.array([src_pose.position.x, src_pose.position.y, src_pose.position.z])
        map_P_src_C = src_C + np.array([src_pose.position.x, src_pose.position.y, src_pose.position.z])
        map_P_src_D = src_D + np.array([src_pose.position.x, src_pose.position.y, src_pose.position.z])
        # map_P_src_opening_point = np.matmul(tf_map_src, np.hstack((src_opening_point, [1])))[0:3]

        # print("rotated pt ", map_P_src_A, map_P_src_B, map_P_src_C, map_P_src_D, map_P_src_opening_point)

        dest_opening_point = np.array(
            [dest_pose.position.x,
             dest_pose.position.y,
             dest_pose.position.z + dest_dim[2] / 2])
        # self.dest_bounding_box_pose.position.z + self.bb.scene_desc["dest_dim"][2] / 2)
        # 0.75 of l and d is to keep the rectangle smaller than the bounding box.
        # To ensure the source opening lies within the dest
        l, d, h = (dest_dim[0] * 0.65) / 2, (dest_dim[1] * 0.65) / 2, dest_opening_point[2]
        a = np.array([dest_opening_point[0] - l, dest_opening_point[1] + d, h])
        b = np.array([dest_opening_point[0] - l, dest_opening_point[1] - d, h])
        c = np.array([dest_opening_point[0] + l, dest_opening_point[1] + d, h])

        ab = b - a
        ac = c - a

        # print("dest points: ", a, b, c)
        # print("vectors ", ab, ap, ac)
        # print("lengths :", np.dot(ap, ab), np.dot(ab, ab), np.dot(ap, ac), np.dot(ac, ac))
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='free_cup2', obj_type=1, action=0, color=(0, 0, 1),
        #                             lifetime=0,
        #                             position=(dest_pose.position.x, dest_pose.position.y, dest_pose.position.z),
        #                             orientation=dest_pose.orientation, size=dest_dim))
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='free_cup', obj_type=1, action=0, color=(0, 0, 1),
        #                             lifetime=0,
        #                             position=(src_pose.position.x, src_pose.position.y, src_pose.position.z),
        #                             orientation=src_pose.orientation, size=src_dim))
        #
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='Ad', obj_type=2, action=0, color=(0, 1, 0), lifetime=0,
        #                             position=a, size=(0.01, 0.01, 0.01)))
        #
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='Bd', obj_type=2, action=0, color=(0, 1, 0), lifetime=0,
        #                             position=b, size=(0.01, 0.01, 0.01)))
        #
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='Cd', obj_type=2, action=0, color=(0, 1, 0), lifetime=0,
        #                             position=c, size=(0.01, 0.01, 0.01)))
        #
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='A', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
        #                             position=map_P_src_A, size=(0.01, 0.01, 0.01)))
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='B', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
        #                             position=map_P_src_B, size=(0.01, 0.01, 0.01)))
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='C', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
        #                             position=map_P_src_C, size=(0.01, 0.01, 0.01)))
        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='D', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
        #                             position=map_P_src_D, size=(0.01, 0.01, 0.01)))

        # self.test_marker_array.markers.append(
        #     self._create_vis_marker(parent_frame='map', ns='src_opening_point', obj_type=2, action=0,
        #                             color=(1, 1, 0), lifetime=0, position=map_P_src_opening_point,
        #                             size=(0.01, 0.01, 0.01)))
        distance_val = []
        dist_position_array = np.array([dest_pose.position.x, dest_pose.position.y, dest_pose.position.z])
        src_points = (map_P_src_A, map_P_src_B, map_P_src_C, map_P_src_D)
        # store the values in a list and then find the corresponding point
        for point in src_points:
            distance_val.append(self.distance(point, dist_position_array))

        closest_index = np.argsort(distance_val)[0]
        self.test_marker_array.markers.append(
            self._create_vis_marker(parent_frame='map', ns='D', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
                                    position=src_points[closest_index], size=(0.03, 0.03, 0.03)))

        # test_check = [self.point_within_bounds(ab, ac, map_P_src_A - a),
        #               self.point_within_bounds(ab, ac, map_P_src_B - a),
        #               self.point_within_bounds(ab, ac, map_P_src_C - a),
        #               self.point_within_bounds(ab, ac, map_P_src_D - a)]

        # print("checkss ", test_check)
        # if test_check.count(True) >= 1:
        #     opening_within = True

        return self.point_within_bounds(ab, ac, src_points[closest_index] - a), src_points[closest_index]

    def is_corner_aligned(self, src_pose: Point, src_height: float, dest_pose: Point, dest_height: float,
                          closest_pt: np.array) -> (bool, str):
        aligned = False
        direction = {-1: "clockwise", 1: "anticlockwise"}
        Orig = np.array([src_pose.x, src_pose.y, src_pose.z + src_height / 2])
        D = np.array([dest_pose.x, dest_pose.y, dest_pose.z + dest_height / 2])

        OA = closest_pt - Orig
        OD = D - Orig
        cross_pdt = np.cross(OA[:2], OD[:2])
        tolerance = 0.001
        if np.abs(cross_pdt) < tolerance:
            aligned = True
        return aligned, direction[np.sign(cross_pdt)]

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_points_on_line(self, lies_along: str, direction: str, obj_pose: Pose, obj_dim: tuple):
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
        A = self.rotate_point(A, rot_matrix)
        B = self.rotate_point(B, rot_matrix)

        return A, B

    def _get_corner_points(self, obj_pose: Quaternion, obj_dim):
        # top lies in XY plane. points in local frame
        # A is bottom left
        A = np.array([obj_dim[0] / 2, -obj_dim[1] / 2, obj_dim[2] / 2])
        B = np.array([obj_dim[0] / 2, obj_dim[1] / 2, obj_dim[2] / 2])
        C = np.array([-obj_dim[0] / 2, obj_dim[1] / 2, obj_dim[2] / 2])
        D = np.array([-obj_dim[0] / 2, -obj_dim[1] / 2, obj_dim[2] / 2])

        rot_matrix = quaternion_matrix(np.array([obj_pose.x, obj_pose.y, obj_pose.z, obj_pose.w]))

        A = self.rotate_point(A, rot_matrix)
        B = self.rotate_point(B, rot_matrix)
        C = self.rotate_point(C, rot_matrix)
        D = self.rotate_point(D, rot_matrix)

        return A, B, C, D

    def _get_edge_points(self, obj_pose: Quaternion, obj_dim):
        A = np.array([0, obj_dim[1] / 2, obj_dim[2] / 2])
        B = np.array([0, -obj_dim[1] / 2, obj_dim[2] / 2])
        C = np.array([obj_dim[0] / 2, 0, obj_dim[2] / 2])
        D = np.array([-obj_dim[0] / 2, 0, obj_dim[2] / 2])

        rot_matrix = quaternion_matrix(np.array([obj_pose.x, obj_pose.y, obj_pose.z, obj_pose.w]))

        A = self.rotate_point(A, rot_matrix)
        B = self.rotate_point(B, rot_matrix)
        C = self.rotate_point(C, rot_matrix)
        D = self.rotate_point(D, rot_matrix)

        return A, B, C, D

    def point_within_bounds(self, ab, ac, ap) -> bool:
        # projecting the point src_opening_point on the ab and ac vectors. ab perpendicular to  ac
        # print((np.dot(ap, ab)), (np.dot(ab, ab)), (np.dot(ap, ac)), (np.dot(ac, ac)))
        return 0 < np.dot(ap, ab) < np.dot(ab, ab) and 0 < np.dot(ap, ac) < np.dot(ac, ac)

    def get_limits(self, length: float, breadth: float, height: float, position: Point, ns=None) -> tuple:
        half_height = height / 2
        half_breadth = breadth / 2
        half_length = length / 2

        ll = (position.x - half_length, position.y - half_breadth, position.z - half_height)
        ul = (position.x + half_length, position.y + half_breadth, position.z + half_height)

        self.test_marker_array.markers.append(self._create_vis_marker(parent_frame='map', ns=ns[0],
                                                                      obj_type=2, action=0, color=(1, 0, 0), lifetime=0,
                                                                      position=ll, size=(0.03, 0.03, 0.03)))
        self.test_marker_array.markers.append(self._create_vis_marker(parent_frame='map', ns=ns[1],
                                                                      obj_type=2, action=0, color=(1, 0, 0), lifetime=0,
                                                                      position=ul, size=(0.03, 0.03, 0.03)))

        return ll, ul

    def _create_vis_marker(self, parent_frame, ns, obj_type, action, color, lifetime, position, size,
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

    def get_transform(self, reference_frame, target_frame):
        self.tf_listener.waitForTransform(reference_frame, target_frame, rospy.Time(), rospy.Duration(5))
        (pos, quat) = self.tf_listener.lookupTransform(reference_frame, target_frame, rospy.Time())
        return pos, quat

    def create_obj_test_alignment(self, obj_names, obj_poses: list):
        for ind, pose in enumerate(obj_poses):
            self.test_marker_array.markers.append(self._create_vis_marker(parent_frame='map', ns=obj_names[ind],
                                                                          obj_type=1, action=0, color=(0, 1, 1),
                                                                          lifetime=0, position=(pose[0][0],
                                                                                                pose[0][1],
                                                                                                pose[0][2]),
                                                                          orientation=Quaternion(pose[1][0],
                                                                                                 pose[1][1],
                                                                                                 pose[1][2],
                                                                                                 pose[1][3]),
                                                                          size=(0.1, 0.1, 0.05)))

    def test_alignment_to_get_direction(self, cont_pose: tuple):
        # Note: You might have to stick with the corner points for objects with edges
        # ToDO : Transform everything to dest frame
        # ToDo: Once there is spilling at t, find by no of particles outside at t - t-1. When it is positive, you need a
        # decrease tilt
        # test cont pose  ([-0.0009452644735574722, -0.10899632424116135, 0.13849832117557526],
        dest_pose = np.array([0, 0, 0])
        print("cont pose ", cont_pose)
        dim = (0.1, 0.1, 0.05)
        # l, d, h = dim[0]/2, dim[1] / 2, dim[2]/2

        q = Quaternion(cont_pose[1][0], cont_pose[1][1], cont_pose[1][2], cont_pose[1][3])
        a, b, c, d = self._get_corner_points(obj_pose=q, obj_dim=(dim[0] * 0.8, dim[1] * 0.8, dim[2]))

        a = a + cont_pose[0]
        b = b + cont_pose[0]
        c = c + cont_pose[0]
        d = d + cont_pose[0]
        corner_points = [a, b, c, d]
        print("points ", a, b, c, d)
        _, _, heights = zip(*corner_points)
        sorted_index = np.argsort(heights)
        lowest_pt = corner_points[sorted_index[0]]
        no_of_low_pts = heights.count(lowest_pt[2])
        interesting_indices = sorted_index[:no_of_low_pts]
        if len(interesting_indices) > 1:
            dist = 0
            for ind in interesting_indices:
                dist_current = self.distance(dest_pose, corner_points[ind])
                if dist == 0:
                    dist = dist_current
                    lowest_pt = corner_points[ind]
                elif dist_current < dist:
                    dist = dist_current
                    lowest_pt = corner_points[ind]
                elif dist_current > dist or dist_current == dist:
                    continue

        print("lowest point ", lowest_pt)
        o = np.array([cont_pose[0][0], cont_pose[0][1], cont_pose[0][2] + dim[2] / 2])
        dest = np.array([dest_pose[0], dest_pose[1], dest_pose[2] + dim[2] / 2])
        print("dest pose, src ", dest, o)
        oa = lowest_pt - o
        od = dest - o
        print("oa, od: ", oa, od)
        cross_pdt = np.cross(oa, od)
        print("cross pdt ", cross_pdt)
        self.test_marker_array.markers.append(
            self._create_vis_marker(parent_frame='map', ns='A', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
                                    position=a, size=(0.01, 0.01, 0.01)))
        self.test_marker_array.markers.append(
            self._create_vis_marker(parent_frame='map', ns='B', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
                                    position=b, size=(0.01, 0.01, 0.01)))
        self.test_marker_array.markers.append(
            self._create_vis_marker(parent_frame='map', ns='C', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
                                    position=c, size=(0.01, 0.01, 0.01)))
        self.test_marker_array.markers.append(
            self._create_vis_marker(parent_frame='map', ns='D', obj_type=2, action=0, color=(1, 1, 0), lifetime=0,
                                    position=d, size=(0.01, 0.01, 0.01)))


# if __name__ == '__main__':
#     rospy.init_node('test_node')
#     u = Utils()
#     pub = rospy.Publisher('/test_marker', MarkerArray, queue_size=1, latch=True)
#
#
#     sp = Pose()
#
#     dim = (0.0646, 0.0646, 0.18)
#     container_obj = ("free_cup2", "free_cup")
#     poses = []
#     for i in container_obj:
#         poses.append(u.get_transform("map", i))
#
#     sp.position.x = poses[0][0][0]
#     sp.position.y = poses[0][0][1]
#     sp.position.z = poses[0][0][2]
#     sp.orientation.x = poses[0][1][0]
#     sp.orientation.y = poses[0][1][1]
#     sp.orientation.z = poses[0][1][2]
#     sp.orientation.w = poses[0][1][3]
#
#     dp = Pose()
#     dp.position.x = poses[1][0][0]
#     dp.position.y = poses[1][0][1]
#     dp.position.z = poses[1][0][2]
#     dp.orientation.x = poses[1][1][0]
#     dp.orientation.y = poses[1][1][1]
#     dp.orientation.z = poses[1][1][2]
#     dp.orientation.w = poses[1][1][3]
#
#     within = u.is_source_opening_within(sp, dim, dp, dim, corner=False)
#     print("openinggg ", within)
#     # opening within
#
#     #
#     # u.create_obj_test_alignment(container_obj, poses)
#     # u.test_alignment_to_get_direction(poses[1])
#     vis_array = u.get_test_visualization_marker_array()
#     # # print(vis_array)
#     pub.publish(vis_array)
#     rospy.sleep(1.0)
#     rospy.spin()
