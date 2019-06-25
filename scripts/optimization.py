#!/usr/bin/env python

"""
Optimzation Module for Processing and Optimizing Raw Vertices and Edges data

by Sherrie Shen, 2018

Last Modified August, 2018

This script will:
- Read in a pickle file from data/raw_data folder. The file contains a pose graph constructed from running
  data_collection.py
- Use breadth first search algorithm defined in pose_graph.py to process the pose graph by starting at the first tag
  vertex seen and delete any vertices or edges that is disconnected from the starting node. Store the result in
  data/process_data folder
- Optimize the pose graph by writing the vertices and edges information to a g2o file. Run g2o and parse the optimized
  result to update the pose graph and stored it in data/optimzed_data folder.
- Plot the optimized and unoptimized pose graph.

Dependencies:
matplotlib: https://matplotlib.org

To use:
- Open Terminal, change the filename to the data that needs to be optimized,  run the code below:

rosrun navigation_prototypes optimization.py

"""
from pose_graph import PoseGraph, Vertex, Edge
import pickle
from os import path
import numpy as np
import math as math
from rospkg import RosPack
import rospy
import geometry_msgs.msg
import tf
from tf.transformations import quaternion_matrix, translation_from_matrix, \
    quaternion_from_matrix, euler_from_quaternion, quaternion_from_euler
from helper_functions import (convert_pose_inverse_transform,
                              convert_translation_rotation_to_pose)
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


class Optimization:
    """
    The Optimization class, the main class of this script, calls methods defined in pose graph to process and optimize
    the pose graph and parses the result
    """

    def __init__(self, filename, location, analyze=False):
        self.package = RosPack().get_path('invisible_map_generation')
        self.raw_data_folder = path.join(self.package, 'data/raw_data')
        self.analyzed_data_folder = path.join(
            self.package, 'data/analyzed_data')
        self.optimized_data_folder = path.join(
            self.package, 'data/optimized_data')
        self.g2o_result_path = path.join(
            self.package, 'data/data_g2o/result.g2o')
        self.location = location

        if analyze:
            self.input_data = path.join(self.analyzed_data_folder, filename)
        else:
            self.input_data = path.join(self.raw_data_folder, filename)

        self.posegraph_g2o = None
        self.posegraph = None
        self.unoptimzied_posegraph = None

        """Interpretation of Optimized Data (Plotting)"""
        self.optimized_pose = None
        self.unoptimized_pose = None
        self.optimized_tag = None
        self.optimized_waypoint = None
        self.xmax = []
        self.xmin = []
        self.ymax = []
        self.ymin = []
        self.zmax = []
        self.zmin = []
        self.index_to_tag_conversion = {}
        self.index_to_waypoint_conversion = {}

    def load_pose_graph(self):
        with open(self.input_data, 'rb') as data:
            # self.posegraph_g2o = pickle.load(data)  # pose graph that will be optimized
            # data.seek(0) # place the handler to the beginning of the pickle file
            # pose graph that will be updated
            self.posegraph = pickle.load(data)

        with open(path.join(self.raw_data_folder, self.location), 'rb') as data:
            # place the handler to the beginning of the pickle file
            data.seek(0)
            self.unoptimzied_posegraph = pickle.load(data)

    def g2o(self, dummy_flag=False, tag_flag=False, waypoint_flag=False):
        """
        Process pose graph with breath first search algorithm traversing from the first tag seen to delete disconnected
        portion.
        Optimize pose graph with g2o optimization.
        :param debug_flag: Flag to debug optimization error by turning on and off dummy and landmark nodes.
        :return: None
        """
        self.load_pose_graph()
        self.posegraph.optimize_pose_without_landmarks_dummy_nodes(dummy_nodes_flag=dummy_flag, tag_flag=tag_flag,
                                                                   waypoint_flag=waypoint_flag)
        if tag_flag:
            # traverse from first pose
            source_node = self.posegraph.odometry_vertices[self.posegraph.num_tags + 1].id
        else:
            source_node = self.posegraph.origin_tag  # traverse from first tag seen
        self.posegraph.process_graph(source_node)
        self.posegraph.optimize_pose()
        # print self.posegraph.odometry_vertices.keys()

    @staticmethod
    def is_integer(num):
        try:
            int(num)
            return True
        except ValueError:
            return False

    @staticmethod
    def update_transformation(dataset, id, translation, rotation):
        """
        Helper function that updates the transform of a node or edge given a certain dataset.
        :param dataset: input dataset
        :param id: id of the node or key of the edge
        :param translation: translation of a node or edge
        :param rotation: rotation in quaternions of a node or edge
        :return: None
        """
        dataset[id].translation = translation
        dataset[id].rotation = rotation

    def update_vertices(self):
        """
        Update the vertices in pose graph from g2o optimization.
        :return: None
        """
        # map string waypoint id to number
        # self.posegraph.map_waypoint_name_to_number()
        with open(self.g2o_result_path, 'rb') as g2o_result:
            for line in g2o_result:
                if line.startswith("VERTEX_SE3:QUAT "):
                    line = line.strip().split()
                    translation = [float(data) for data in line[2:5]]
                    rotation = [float(data) for data in line[5:9]]
                    if int(line[1]) <= self.posegraph.num_tags:  # update tags
                        id = int(line[1])
                        Optimization.update_transformation(
                            self.posegraph.tag_vertices, id, translation, rotation)
                    # update odometry
                    elif self.posegraph.num_tags < int(line[1]) < self.posegraph.waypoint_start_id:
                        id = int(line[1])
                        Optimization.update_transformation(
                            self.posegraph.odometry_vertices, id, translation, rotation)
                    else:
                        id = self.posegraph.waypoint_id_to_name[int(
                            line[1])]  # update waypoints
                        Optimization.update_transformation(
                            self.posegraph.waypoints_vertices, id, translation, rotation)
                        # map waypoint id back to name
                        self.posegraph.waypoints_vertices[id].id = id

    def query_trans_rot_from_calibrated_vertices(self, id):
        """
        Retrun translation, rotation from the corresponding vertices dictionary given a vertex id
        """
        if Optimization.is_integer(id) and int(id) <= self.posegraph.num_tags:
            translation = self.posegraph.tag_vertices[id].translation
            rotation = self.posegraph.tag_vertices[id].rotation
        elif Optimization.is_integer(id) and int(id) > self.posegraph.num_tags:
            translation = self.posegraph.odometry_vertices[id].translation
            rotation = self.posegraph.odometry_vertices[id].rotation
        else:
            translation = self.posegraph.waypoints_vertices[id].translation
            rotation = self.posegraph.waypoints_vertices[id].rotation
        return translation, rotation

    def update_edges_vertices_from_calibration(self, edges):
        for id in edges.keys():
            start_id = edges[id].start.id
            trans_start, rot_start = self.query_trans_rot_from_calibrated_vertices(
                start_id)
            edges[id].start.translation = trans_start
            edges[id].start.rotation = rot_start
            end_id = edges[id].end.id
            trans_end, rot_end = self.query_trans_rot_from_calibrated_vertices(
                end_id)
            edges[id].end.translation = trans_end
            edges[id].end.rotation = rot_end

    def update_all_edges_vertices(self):
        for odom_start_id in self.posegraph.odometry_edges.keys():
            self.update_edges_vertices_from_calibration(
                self.posegraph.odometry_edges[odom_start_id])
        for tag_id in self.posegraph.odometry_tag_edges.keys():
            self.update_edges_vertices_from_calibration(
                self.posegraph.odometry_tag_edges[tag_id])
        for waypoint_id in self.posegraph.odometry_waypoints_edges.keys():
            self.update_edges_vertices_from_calibration(
                self.posegraph.odometry_waypoints_edges[waypoint_id])

    @staticmethod
    def compute_new_edges_math(edges):
        for id in edges.keys():
            # write start vertex as a transformation
            start_trans = edges[id].start.translation
            start_rot = edges[id].start.rotation
            start_pose = convert_translation_rotation_to_pose(
                start_trans, start_rot)
            start_trans_inv, start_rot_inv = convert_pose_inverse_transform(
                start_pose)
            start_transformation = quaternion_matrix(start_rot_inv)
            start_transformation[:-1, -1] = np.array(start_trans_inv).T
            # write end vertex as a transformation
            end_trans = edges[id].end.translation
            end_rot = edges[id].end.rotation
            end_transformation = quaternion_matrix(end_rot)
            end_transformation[:-1, -1] = np.array(end_trans).T
            # compute transformation between two vertices
            start_end_transformation = np.matmul(
                end_transformation, start_transformation)
            translation = translation_from_matrix(start_end_transformation)
            rotation = quaternion_from_matrix(start_end_transformation)
            edges[id].translation_computed = list(translation)
            edges[id].rotation_computed = list(rotation)

    def compute_all_new_edges_math(self):
        for odom_start_id in self.posegraph.odometry_edges.keys():
            Optimization.compute_new_edges_math(
                self.posegraph.odometry_edges[odom_start_id])
        for tag_id in self.posegraph.odometry_tag_edges.keys():
            Optimization.compute_new_edges_math(
                self.posegraph.odometry_tag_edges[tag_id])
        for waypoint_id in self.posegraph.odometry_waypoints_edges.keys():
            Optimization.compute_new_edges_math(
                self.posegraph.odometry_waypoints_edges[waypoint_id])

    @staticmethod
    def write_transform_stamped_msg(transformer, id, trans, rot, parent):
        frame = geometry_msgs.msg.TransformStamped()
        frame.header.frame_id = parent
        frame.child_frame_id = id
        frame.transform.translation.x = trans[0]
        frame.transform.translation.y = trans[1]
        frame.transform.translation.z = trans[2]
        frame.transform.rotation.x = rot[0]
        frame.transform.rotation.y = rot[1]
        frame.transform.rotation.z = rot[2]
        frame.transform.rotation.w = rot[3]
        transformer.setTransform(frame)

    @staticmethod
    def compute_new_edges_transformer(edges):
        for id in edges.keys():
            transform = tf.Transformer(True, rospy.Duration(10))
            start_trans = edges[id].start.translation
            start_rot = edges[id].start.rotation
            end_trans = edges[id].end.translation
            end_rot = edges[id].end.rotation
            Optimization.write_transform_stamped_msg(
                transform, "start", start_trans, start_rot, "map")
            Optimization.write_transform_stamped_msg(
                transform, "end", end_trans, end_rot, "map")
            translation, rotation = transform.lookupTransform(
                "start", "end", rospy.Time(0))
            edges[id].translation_computed = list(translation)
            edges[id].rotation_computed = list(rotation)

    def compute_all_new_edges_transformer(self):
        for odom_start_id in self.posegraph.odometry_edges.keys():
            Optimization.compute_new_edges_transformer(
                self.posegraph.odometry_edges[odom_start_id])
        for tag_id in self.posegraph.odometry_tag_edges.keys():
            Optimization.compute_new_edges_transformer(
                self.posegraph.odometry_tag_edges[tag_id])
        for waypoint_id in self.posegraph.odometry_waypoints_edges.keys():
            Optimization.compute_new_edges_transformer(
                self.posegraph.odometry_waypoints_edges[waypoint_id])

    def update_edges_math(self):
        self.update_all_edges_vertices()
        self.compute_all_new_edges_math()

    def update_edges_transformer(self):
        self.update_all_edges_vertices()
        self.compute_all_new_edges_transformer()

    @staticmethod
    def compute_edges_transformation_difference(edges):
        """
        Given a dictionary of edges, compute the difference between the new edges and old edges
        :param edges:
        :return: None
        """
        for id in edges.keys():
            trans_old = edges[id].translation
            trans_new = edges[id].translation_computed
            edges[id].translation_diff = [trans_new[i] - trans_old[i]
                                          for i in range(3)]
            rot_old = edges[id].rotation
            rot_old_euler = euler_from_quaternion(rot_old)
            rot_new = edges[id].rotation_computed
            rot_new_euler = euler_from_quaternion(rot_new)
            edges[id].rotation_diff = [Optimization.angle_diff(
                rot_new_euler[i], rot_old_euler[i]) for i in range(3)]
            # if edges[id].damping_status:
            #     print id
            #     print rot_old_euler
            #     print rot_new_euler

    @staticmethod
    def angle_normalize(z):
        """ convenience function to map an angle to the range [-pi,pi] """
        return math.atan2(math.sin(z), math.cos(z))

    @staticmethod
    def angle_diff(a, b):
        """ Calculates the difference between angle a and angle b (both should
            be in radians) the difference is always based on the closest
            rotation from angle a to angle b.
            examples:
                angle_diff(.1,.2) -> -.1
                angle_diff(.1, 2*math.pi - .1) -> .2
                angle_diff(.1, .2+2*math.pi) -> -.1
        """
        a = Optimization.angle_normalize(a)
        b = Optimization.angle_normalize(b)
        d1 = a - b
        d2 = 2 * math.pi - math.fabs(d1)
        if d1 > 0:
            d2 *= -1.0
        if math.fabs(d1) < math.fabs(d2):
            return d1
        else:
            return d2

    @staticmethod
    def compute_optimization_cost_for_edges(*edges):
        total_cost = 0
        for edge in edges:
            for start_id in edge.keys():
                for end_id in edge[start_id].keys():
                    edge[start_id][end_id].compute_optimization_cost()
                    total_cost += edge[start_id][end_id].optimization_cost
        return total_cost

    @staticmethod
    def compute_optimization_cost_for_edges_hack(*edges):
        total_cost = 0
        for edge in edges:
            for start_id in edge.keys():
                for end_id in edge[start_id].keys():
                    transformation = np.array(
                        edge[start_id][end_id].translation_diff + edge[start_id][end_id].rotation_diff)
                    optimization_cost = np.matmul(np.matmul(transformation, edge[start_id][end_id].importance_matrix),
                                                  transformation.T)
                    total_cost += optimization_cost
        return total_cost

    def compute_all_edges_transformation_diff_and_cost(self):
        for odom_start_id in self.posegraph.odometry_edges.keys():
            Optimization.compute_edges_transformation_difference(
                self.posegraph.odometry_edges[odom_start_id])
        for tag_id in self.posegraph.odometry_tag_edges.keys():
            Optimization.compute_edges_transformation_difference(
                self.posegraph.odometry_tag_edges[tag_id])
        for waypoint_id in self.posegraph.odometry_waypoints_edges.keys():
            Optimization.compute_edges_transformation_difference(
                self.posegraph.odometry_waypoints_edges[waypoint_id])
        self.posegraph.optimization_cost = Optimization.compute_optimization_cost_for_edges(
            self.posegraph.odometry_edges, self.posegraph.odometry_tag_edges, self.posegraph.odometry_waypoints_edges)
        print "OPTIMIZATION COST:", self.posegraph.optimization_cost

    def parse_g2o_result_math(self):
        self.update_vertices()
        self.update_edges_math()
        self.compute_all_edges_transformation_diff_and_cost()
        with open(path.join(self.optimized_data_folder, "data_optimized.pkl"), 'wb') as f:
            pickle.dump(self.posegraph, f)
            print "OPTIMIZED POSE GRAPH SAVED"

    def parse_g2o_result_transformer(self):
        self.update_vertices()
        self.update_edges_transformer()
        self.compute_all_edges_transformation_diff_and_cost()
        with open(path.join(self.optimized_data_folder, "data_optimized.pkl"), 'wb') as f:
            pickle.dump(self.posegraph, f)
            print "OPTIMIZED POSE GRAPH SAVED"

    @staticmethod
    def multiply_transform(tr1, tr2):
        T = quaternion_matrix(tr1[1])
        T[:-1, -1] = np.squeeze(np.asarray(tr1[0]))

        T2 = quaternion_matrix(tr2[1])
        T2[:-1, -1] = np.squeeze(np.asarray(tr2[0]))

        tres = np.matmul(T, T2)
        trans = translation_from_matrix(tres)
        rot = quaternion_from_matrix(tres)

        return trans, rot

    @staticmethod
    def process_vertices_for_plot(vertices):
        optimized_vertices = []
        for key in sorted(vertices):
            if vertices[key].translation != [0, 0, 0]:  # remove dummy vertices
                optimized_vertices.append(vertices[key].translation)
        optimized_vertices = np.array(optimized_vertices)
        return optimized_vertices

    @staticmethod
    def map_index_to_landmark_id(landmarks, stored_loc):
        sorted_landmarks_id = sorted(landmarks.keys())
        for landmark_id in sorted(landmarks):
            id_index = sorted_landmarks_id.index(landmark_id)
            stored_loc[id_index] = landmark_id

    def plot_extreme_vertices_for_pose(self, ax):
        all_translations = []
        for id in self.posegraph.odometry_vertices.keys():
            # not including dummy vertices
            if not self.posegraph.odometry_vertices[id].fix_status:
                translation = self.posegraph.odometry_vertices[id].translation
                all_translations.append(
                    (id, translation[0], translation[1], translation[2]))
        trans_sort_x = sorted(all_translations, key=lambda vertex: vertex[1])
        trans_sort_y = sorted(all_translations, key=lambda vertex: vertex[2])
        trans_sort_z = sorted(all_translations, key=lambda vertex: vertex[3])
        self.xmax = trans_sort_x[-1]
        self.xmin = trans_sort_x[0]
        self.ymax = trans_sort_y[-1]
        self.ymin = trans_sort_y[0]
        self.zmax = trans_sort_z[-1]
        self.zmin = trans_sort_z[0]
        for extreme_pt in [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]:
            print "id:", extreme_pt[0]
            plt.plot((extreme_pt[1],), (extreme_pt[2],),
                     (extreme_pt[3],), 'rx')
            ax.text(extreme_pt[1], extreme_pt[2],
                    extreme_pt[3], str(extreme_pt[0]))

    @staticmethod
    def plot_landmarks(ax, landmarks, landmarks_index_to_id):
        for row in range(np.shape(landmarks)[0]):
            landmark = landmarks[row, :]
            plt.plot((landmark[0],), (landmark[1],), (landmark[2],), 'go')
            ax.text(landmark[0], landmark[1], landmark[2],
                    str(landmarks_index_to_id[row]))

    def plot_g2o_trajectory(self):
        self.optimized_pose = Optimization.process_vertices_for_plot(
            self.posegraph.odometry_vertices)
        self.unoptimized_pose = Optimization.process_vertices_for_plot(
            self.unoptimzied_posegraph.odometry_vertices)
        self.optimized_tag = Optimization.process_vertices_for_plot(
            self.posegraph.tag_vertices)
        self.optimized_waypoint = Optimization.process_vertices_for_plot(
            self.posegraph.waypoints_vertices)
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        # plot optimized pose vertices
        optimized_pose, = plt.plot(self.optimized_pose[:, 0], self.optimized_pose[:, 1], self.optimized_pose[:, 2],
                                   'bo',
                                   label='corrected trajectory')

        # plot unoptimized pose vertices
        unoptimized_pose, = plt.plot(self.unoptimized_pose[:, 0], self.unoptimized_pose[:, 1],
                                     self.unoptimized_pose[:, 2], 'm--', label='uncorrected trajectory')

        # plot extreme pose vertices
        self.plot_extreme_vertices_for_pose(ax)

        # plot tag vertices
        Optimization.map_index_to_landmark_id(
            self.posegraph.tag_vertices, self.index_to_tag_conversion)
        Optimization.plot_landmarks(
            ax, self.optimized_tag, self.index_to_tag_conversion)

        # plot waypoint vertices
        Optimization.map_index_to_landmark_id(
            self.posegraph.waypoints_vertices, self.index_to_waypoint_conversion)
        Optimization.plot_landmarks(
            ax, self.optimized_waypoint, self.index_to_waypoint_conversion)

        plt.legend(handles=[optimized_pose, unoptimized_pose])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('G2O Trajectory Plot')
        plt.grid(True)
        plt.show()

    def run(self, dummy_flag=False, tag_flag=False, waypoint_flag=False, plot=True):
        self.g2o(dummy_flag=dummy_flag, tag_flag=tag_flag,
                 waypoint_flag=waypoint_flag)
        self.parse_g2o_result_transformer()
        # self.parse_g2o_result_math() # currently inaccurate
        if plot:
            self.plot_g2o_trajectory()


if __name__ == "__main__":
    process = Optimization("academic_center.pkl", "academic_center.pkl")
    #process = Optimization("data_analyzed.pkl", "academic_center.pkl", analyze=True)
    process.run(dummy_flag=False, tag_flag=False,
                waypoint_flag=False)  # False is do optimize
