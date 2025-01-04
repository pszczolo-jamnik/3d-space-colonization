import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from numba import njit, prange

from numba.core.errors import NumbaWarning
import warnings
warnings.filterwarnings('ignore', category=NumbaWarning)

from collections import defaultdict

import plyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--attraction-distance", help="", type=float, required=True)
parser.add_argument("--kill-distance", help="", type=float, required=True)
parser.add_argument("--segment-length", help="", type=float, required=True)
args = parser.parse_args()

n_attractors = 10000
attraction_distance = 3.0
kill_distance = 0.5
step = 0.1

BIG_NUMBER = n_attractors * 10
INF = float('inf')

def plot_3d(points):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z)

    plt.show()

@njit(parallel=True)
def grow(nodes, attractors, node_attractors_list, nodes_len, step):
    output = np.zeros_like(nodes)
    output_normals = np.zeros_like(nodes)

    for i in prange(nodes_len):
        node = nodes[i]
        idx = node_attractors_list[i]

        if idx[0] != BIG_NUMBER:
            idx = idx[idx < n_attractors]
            vectors = attractors[idx] - node

            squared = vectors * vectors
            summed = np.sum(squared, axis=1)
            norms =  np.sqrt(summed)

            direction_vectors = vectors / norms.reshape(-1, 1)
            n = direction_vectors.shape[0]
            average_direction = np.sum(direction_vectors, axis=0) / n
            average_direction /= np.linalg.norm(average_direction)

            output[i] = node + average_direction * step
            output_normals[i] = average_direction

    return (output, output_normals)

def write_xyz_to_ply(nodes, normals, age, output_filepath):
    nodes = np.hstack((nodes, normals, age[:, np.newaxis]))

    vertex = np.array([(n[0], n[1], n[2],
                        n[3], n[4], n[5],
                        n[6]) for n in nodes],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                               ('age', 'f4')])

    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el]).write(output_filepath)

def random_sphere_points(n, r=1):
    theta = np.random.uniform(0, 2 * np.pi, n)  # Azimuthal angle
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n))  # Polar angle
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.vstack((x, y, z)).T


# attractors = np.random.uniform(0.0, 20.0, (n_attractors, 3))
attractors = random_sphere_points(5000, 10)
nodes = random_sphere_points(15, 12)

normals = np.ones_like(nodes)

age = np.zeros((nodes.shape[0],))

# nodes = np.zeros((1, 3))
# nodes[0] = (8, 8, 8)

def update(nodes, normals, attractors):
    node_tree = KDTree(nodes)

    nodes_len = nodes.shape[0]

    (kill_dist, kill_idx) = node_tree.query(x=attractors, k=1, distance_upper_bound=kill_distance)
    kill_idx = [i for i, item in enumerate(kill_idx) if item != nodes_len]
    attractors = np.delete(attractors, kill_idx, axis=0)

    # Associate each attractor with the single closest node
    # within the pre-defined attraction distance
    (nearest_dist, nearest_idx) = node_tree.query(x=attractors, k=1, distance_upper_bound=attraction_distance)

    node_attractors_dict = defaultdict(list)

    for j, item in enumerate(nearest_idx):
        if item != nodes_len:
            node_attractors_dict[item].append(j)

    node_attractors_list = [node_attractors_dict[i] for i in range(nodes_len)]

    max_length = max(len(lst) for lst in node_attractors_list)
    node_attractors_list = [lst + [BIG_NUMBER] * (max_length - len(lst)) for lst in node_attractors_list]

    node_attractors_list = np.array(node_attractors_list)

    (new_nodes, new_normals) = grow(nodes, attractors, node_attractors_list, nodes_len, step)

    new_nodes = new_nodes[~np.all(new_nodes == 0, axis=1)]
    new_normals = new_normals[~np.all(new_normals == 0, axis=1)]

    new_count = new_nodes.shape[0]

    return (np.concatenate((nodes, new_nodes)),
            np.concatenate((normals, new_normals)),
            attractors,
            new_count)

iterations = 100

for cycle in range(iterations):
    (nodes, normals, attractors, new_count) = update(nodes, normals, attractors)
    age = np.concatenate((age, np.tile((cycle + 1) / iterations, new_count)))

    print(f"nodes {nodes.shape[0]}")
    print(f"attractors {attractors.shape[0]}")

write_xyz_to_ply(nodes, normals, age, "tree.ply")

# plot_3d(nodes)
# plot_3d(normals)
# plot_3d(attractors)

