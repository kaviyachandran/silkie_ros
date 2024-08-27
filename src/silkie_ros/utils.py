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
        A, B, C, _ = self._get_corner_points(container_pose.orientation, container_dim, where="top")
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

    def get_distance_to_retained_object(self, obj_pose: Pose, obj_dim: tuple, ref_name: str, obj_name: list,
                                        lies_along: str, direction: str):
        bigball_dist = []
        for obj in obj_name:
            pot_bigball = self.get_transform(reference_frame=ref_name,
                                             target_frame=obj)
            pot_P_bigball = Point()
            pot_P_bigball.x = pot_bigball[0][0]
            pot_P_bigball.y = pot_bigball[0][1]
            pot_P_bigball.z = pot_bigball[0][2]
            A, B = self._get_points_on_line(lies_along, direction, obj_pose, obj_dim)
            P = np.array([pot_P_bigball.x, pot_P_bigball.y, pot_P_bigball.z])
            bigball_dist.append(self.closest_point_on_line_to_point(A, B, P))
        return np.sort(bigball_dist)[0]

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
            src_A, src_B, src_C, src_D = self._get_corner_points(src_pose.orientation, (src_dim[0] * 0.85,
                                                                                        src_dim[1] * 0.85,
                                                                                        src_dim[2]), where="top")
        elif not corner:
            src_A, src_B, src_C, src_D = self._get_edge_points(src_pose.orientation, (src_dim[0] * 0.85,
                                                                                      src_dim[1] * 0.85,
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
            self._create_vis_marker(parent_frame='map', ns='opening', obj_type=2, action=0, color=(0, 0, 1), lifetime=0,
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

    def is_above(self, src_pose: Pose, src_h: float, dest_pose: Pose, dest_h: float, dest_name: str) -> bool:
        src_bottom_point = np.array([0, 0, -src_h / 2])
        rotated_src_bottom_point = self.rotate_point(src_bottom_point, quaternion_matrix(np.array([
                                                    src_pose.orientation.x, src_pose.orientation.y,
                                                    src_pose.orientation.z, src_pose.orientation.w])))
        map_Point_src_bottom = rotated_src_bottom_point + np.array([src_pose.position.x, src_pose.position.y,
                                                                    src_pose.position.z])
        dest_top_point = np.array([0, 0, dest_h / 2])
        map_Pose_dest = self._get_transform_matrix_from_pose(dest_pose)
        dest_Pose_map = np.linalg.inv(map_Pose_dest)
        # TODO : Use the wrapper in giskard
        # to transform point : https://github.com/SemRoCo/giskardpy/blob/81325ff1c8e41c2392bd10304553341624549cbf/src/giskardpy/utils/tfwrapper.py#L87
        dest_Point_src_bottom = np.dot(dest_Pose_map, np.hstack((map_Point_src_bottom, [1])).reshape(4, 1))
        self.test_marker_array.markers.append(
            self._create_vis_marker(parent_frame=dest_name, ns='src_bottom_pt', obj_type=2, action=0, color=(1, 1, 0),
                                    lifetime=0, position=dest_Point_src_bottom[:3], size=(0.03, 0.03, 0.03)))
        # print("points  ", dest_Point_src_bottom[2][0], dest_top_point[2])
        return bool(dest_Point_src_bottom[2][0] > dest_top_point[2])

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_transform_matrix_from_pose(self, pose: Pose) -> np.array:
        rotation_mat = quaternion_matrix(
            np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                      pose.orientation.w]))

        tf_matrix = np.hstack((rotation_mat[:3, :3],
                                np.array([pose.position.x, pose.position.y,
                                          pose.position.z]).reshape(3, 1)))
        tf_matrix = np.vstack((tf_matrix, np.array([0, 0, 0, 1]).reshape(1, 4)))

        return tf_matrix

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
        # A = self.rotate_point(A, rot_matrix)
        # B = self.rotate_point(B, rot_matrix)

        return A, B

    def _get_corner_points(self, obj_pose: Quaternion, obj_dim, where: str = "top"):
        # top lies in XY plane. points in local frame
        A = np.zeros(3)
        B = np.zeros(3)
        C = np.zeros(3)
        D = np.zeros(3)
        # A is bottom left
        if where == "top":
            A = np.array([-obj_dim[0] / 2, -obj_dim[1] / 2, obj_dim[2] / 2])
            B = np.array([obj_dim[0] / 2, -obj_dim[1] / 2, obj_dim[2] / 2])
            C = np.array([-obj_dim[0] / 2, obj_dim[1] / 2, obj_dim[2] / 2])
            D = np.array([obj_dim[0] / 2, obj_dim[1] / 2, obj_dim[2] / 2])
        elif where == "bottom":
            A = np.array([-obj_dim[0] / 2, -obj_dim[1] / 2, -obj_dim[2] / 2])
            B = np.array([obj_dim[0] / 2, -obj_dim[1] / 2, -obj_dim[2] / 2])
            C = np.array([-obj_dim[0] / 2, obj_dim[1] / 2, -obj_dim[2] / 2])
            D = np.array([obj_dim[0] / 2, obj_dim[1] / 2, -obj_dim[2] / 2])

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

    def points_within_bounds_3d(self, ab, ac, ad, ap) -> bool:
        return self.point_within_bounds(ab, ac, ap) and 0 < np.dot(ap, ad) < np.dot(ad, ad)

    def get_points_to_project(self, dimension: tuple, obj_pose: Pose) -> tuple:
        bottom_points = self._get_corner_points(obj_pose.orientation, dimension, where="bottom")
        z = bottom_points[0][2] + dimension[2]
        D = np.array([bottom_points[0][0], bottom_points[0][1], z])
        D = self.rotate_point(D, quaternion_matrix(np.array([obj_pose.orientation.x, obj_pose.orientation.y,
                                                             obj_pose.orientation.z, obj_pose.orientation.w]))) + \
            np.array([obj_pose.position.x, obj_pose.position.y, obj_pose.position.z])
        bottom_points = bottom_points + np.array([obj_pose.position.x, obj_pose.position.y, obj_pose.position.z])
        self.test_marker_array.markers.append(self._create_vis_marker(parent_frame='map', ns='A',
                                                                      obj_type=2, action=0, color=(1, 0, 0), lifetime=0,
                                                                      position=bottom_points[0],
                                                                      size=(0.03, 0.03, 0.03)))
        self.test_marker_array.markers.append(self._create_vis_marker(parent_frame='map', ns='B',
                                                                      obj_type=2, action=0, color=(1, 0, 0), lifetime=0,
                                                                      position=bottom_points[1],
                                                                      size=(0.03, 0.03, 0.03)))
        self.test_marker_array.markers.append(self._create_vis_marker(parent_frame='map', ns='C',
                                                                      obj_type=2, action=0, color=(1, 0, 0), lifetime=0,
                                                                      position=bottom_points[2],
                                                                      size=(0.03, 0.03, 0.03)))
        self.test_marker_array.markers.append(self._create_vis_marker(parent_frame='map', ns='D',
                                                                      obj_type=2, action=0, color=(1, 0, 0), lifetime=0,
                                                                      position=D, size=(0.03, 0.03, 0.03)))
        return bottom_points[0], bottom_points[1], bottom_points[2], D

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

    def create_obj_test_alignment(self, obj_names, obj_poses: list, dim: tuple):
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
                                                                          size=dim))

    def get_direction_and_orientation(self, height: float, obj_pose: Pose):
        normal = np.array([0, 0, 1])
        point_cup_bottom = np.array([0., 0., -height / 2])

        rot_matrix = quaternion_matrix(np.array([obj_pose.orientation.x,
                                                 obj_pose.orientation.y,
                                                 obj_pose.orientation.z,
                                                 obj_pose.orientation.w]))

        # rotated_point
        point_map_bottom = self.rotate_point(point_cup_bottom, rot_matrix) + np.array([obj_pose.position.x,
                                                                                       obj_pose.position.y,
                                                                                       obj_pose.position.z])

        ob = np.array([obj_pose.position.x, obj_pose.position.y, obj_pose.position.z]) - point_map_bottom
        direction = np.dot(normal, ob)
        orientation = np.degrees(np.arccos(direction / np.linalg.norm(ob)))
        return direction, orientation

    def get_direction_vector(self, dest_point: np.array, src_point: np.array):
        v_src_dest = dest_point - src_point
        return v_src_dest / np.linalg.norm(v_src_dest)

    def get_direction_relative_to_dest(self, direction_vector):
        # ToDo: Wrt robot
        coordinate = np.argsort(direction_vector)[::-1]  # descending order
        location = []
        for index in coordinate:
            if index == 0:  # Along x-axis w.r.t map
                if direction_vector[index] > 0:
                    location.append("inFront")
                elif direction_vector[index] < 0:
                    location.append("behind")
            else:  # Along y-axis w.r.t map
                if direction_vector[index] > 0:
                    location.append("left")
                elif direction_vector[index] < 0:
                    location.append("right")

        return location

    def get_particles_in_src_spilling_boundary(self, src_position: np.array, radius: float, particle_poses: list) \
            -> int:
        num_in_boundary = 0
        for particle in particle_poses:
            if self.distance(src_position, particle[0:2]) < radius:
                num_in_boundary += 1
        return num_in_boundary

    def test_point_within_bounds(self, points, p):
        A, B, C, D = points
        ab = B - A
        ac = C - A
        ad = D - A
        ap = p - A
        is_within = self.points_within_bounds_3d(ab, ac, ad, ap)
        print(is_within)

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
        a, b, c, d = self._get_corner_points(obj_pose=q, obj_dim=(dim[0] * 0.8, dim[1] * 0.8, dim[2]), where="top")

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
#     sp = Pose()
#
#     dim = (0.07, 0.07, 0.18)
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
#     # u.get_limits(dim, sp, ('sl', 'su'))
#     # u.get_limits(dim, dp, ('dl', 'du'))
#     #     within = u.is_source_opening_within(sp, dim, dp, dim, corner=False)
#     #     print("openinggg ", within)
#     #     # opening within
#     #
#     # test_points = u.get_points_to_project(dim, sp)
#     # test = u.get_transform('map', 'p')
#     u.create_obj_test_alignment(container_obj, poses, dim)
#     ab = u.is_above(sp, 0.05, dp, 0.05, "free_cup")
#     # u.test_point_within_bounds(test_points, test[0])
#     print("above ", (ab))
#     #     # u.test_alignment_to_get_direction(poses[1])
#     vis_array = u.get_test_visualization_marker_array()
#
#     pub.publish(vis_array)
#     rospy.sleep(1.0)
#     rospy.spin()
