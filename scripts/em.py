#!/usr/bin/env python
import pickle
from pose_graph import Edge
from os import path
from rospkg import RosPack
import numpy as np
from maximization_model import maxweights


def edge2vectors(edge, mode):
    observations = np.zeros([6, 18])
    observations[:, mode*6:(mode+1)*6] = np.eye(6)
    errors = edge.translation_diff + edge.rotation_diff
    return (observations.tolist(), errors)


class Analysis:
    def __init__(self, filename):
        self.package = RosPack().get_path('navigation_prototypes')
        self.optimized_data_folder = path.join(
            self.package, 'data/optimized_data')
        self.analyzed_data_folder = path.join(
            self.package, 'data/analyzed_data')
        with open(path.join(self.optimized_data_folder, filename), 'rb') as data:
            self.posegraph = pickle.load(data)

        self.get_observations_and_errors()
        self.success = False
        self.w = np.zeros(18)

    def get_observations_and_errors(self):
        observations = []
        errors = []

        for startid in self.posegraph.odometry_edges:
            for endid in self.posegraph.odometry_edges[startid]:
                edge = self.posegraph.odometry_edges[startid][endid]
                if edge.damping_status:
                    obs, err = edge2vectors(edge, 2)
                    observations.extend(obs)
                    errors.extend(err)
                else:
                    obs, err = edge2vectors(edge, 0)
                    observations.extend(obs)
                    errors.extend(err)

        for startid in self.posegraph.odometry_tag_edges:
            for endid in self.posegraph.odometry_tag_edges[startid]:
                edge = self.posegraph.odometry_tag_edges[startid][endid]
                obs, err = edge2vectors(edge, 1)
                observations.extend(obs)
                errors.extend(err)

        self.observations = np.array(observations)
        self.errors = np.array(errors)

    def getvariance(self):
        res = maxweights(self.observations, self.errors, self.w)
        success = res.success

        self.success = success
        self.w = np.array(res.x)
        self.variance = np.exp(self.w)

    def updateEdges(self):
        importance = 1 / np.sqrt(self.variance)
        indices = ([3, 3, 3, 4, 4, 5], [3, 4, 5, 4, 5, 5])

        for startid in self.posegraph.odometry_edges:
            for endid in self.posegraph.odometry_edges[startid]:
                edge = self.posegraph.odometry_edges[startid][endid]

                if self.posegraph.odometry_edges[startid][endid].damping_status:
                    basis = Edge.compute_basis_vector(
                        edge.start.rotation, importance[15], importance[16], importance[17])
                    importanceMatrix = np.diag(importance[12:18])
                    # self.posegraph.odometry_edges[startid][endid].importance_matrix = importanceMatrix
                else:
                    importanceMatrix = np.diag(importance[:6])
                    importanceMatrix[[3, 3, 3, 4, 4, 5], [
                        3, 4, 5, 4, 5, 5]] = basis[np.triu_indices(3)]
                    self.posegraph.odometry_edges[startid][endid].importance_matrix = importanceMatrix

        for startid in self.posegraph.odometry_tag_edges:
            for endid in self.posegraph.odometry_tag_edges[startid]:
                importanceMatrix = np.diag(importance[6:12])
                lower = importanceMatrix.T
                importanceMatrix[np.tril_indices(
                    6, 1)] = lower[np.tril_indices(6, 1)]
                self.posegraph.odometry_tag_edges[startid][endid].importance_matrix = importanceMatrix

    def writePosegraph(self):
        self.posegraph.write_g2o_data()

        with open(path.join(self.analyzed_data_folder, 'data_analyzed.pkl'), 'wb') as data:
            pickle.dump(self.posegraph, data)


def main():
    analysis = Analysis('data_optimized.pkl')
    analysis.getvariance()
    analysis.updateEdges()
    analysis.writePosegraph()
    print(analysis.posegraph.odometry_edges.values()[0].values()[0].end.rotation)
    return analysis


if __name__ == '__main__':
    x = main()
