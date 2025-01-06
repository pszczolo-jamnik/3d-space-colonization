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

from opensimplex import OpenSimplex

parser = argparse.ArgumentParser()
parser.add_argument("--attraction_distance", help="", type=float, required=True)
parser.add_argument("--kill_distance", type=float, required=True)
parser.add_argument("--segment_length", type=float, required=True)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--file_in", required=True)
parser.add_argument("--dir_out", required=True)
args = parser.parse_args()

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
    output_parent = np.full((nodes.shape[0],), INT_MAX, dtype=int)

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

            # might experiment with how far it grows
            output[i] = node + average_direction * step
            output_normals[i] = average_direction
            output_parent[i] = i

    return (output, output_normals, output_parent)

def write_ply(nodes, normals, thickness, output_filepath):
    nodes = np.hstack((nodes, normals, thickness[:, np.newaxis]))

    vertex = np.array([(n[0], n[1], n[2],
                        n[3], n[4], n[5],
                        n[6]) for n in nodes],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                               ('thickness', 'f4')])

    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el]).write(output_filepath)

def read_ply(filename):
    plydata = plyfile.PlyData.read(filename)
    vertex_data = plydata['vertex']

    vertices = np.vstack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ]).T
    return vertices

def random_sphere_points(n, r=1):
    theta = np.random.uniform(0, 2 * np.pi, n)  # Azimuthal angle
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n))  # Polar angle
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.vstack((x, y, z)).T

def noise_probability(attractors, noise_grid, noise_start, noise_size):
    simplex = OpenSimplex(seed=2)

    noise_res = noise_size / noise_grid

    noise_xyz = np.arange(noise_start, noise_start + noise_size, noise_res)

    noise = simplex.noise3array(noise_xyz, noise_xyz, noise_xyz)

    # normalize
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    # grid
    attractor_indices = np.floor(attractors * noise_grid).astype(int)
    attractor_indices = np.clip(attractor_indices, 0, noise_grid-1)

    attractor_counts = np.zeros_like(noise)
    np.add.at(attractor_counts, (attractor_indices[:, 0],
                                attractor_indices[:, 1],
                                attractor_indices[:, 2]), 1)

    target_counts = np.round((noise / noise.sum()) * len(attractors)).astype(int)

    # Generate new points
    attractors_new = []
    n = noise_grid
    for i in range(n):
        for j in range(n):
            for k in range(n):
                num_points = target_counts[i, j, k]
                if num_points > 0:
                    # Generate random points within the cell
                    cell_points = np.random.uniform(
                        low=[i/n, j/n, k/n],
                        high=[(i+1)/n, (j+1)/n, (k+1)/n],
                        size=(num_points, 3)
                    )
                    attractors_new.append(cell_points)

    return np.vstack(attractors_new)

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

    (new_nodes, new_normals, new_parent) = grow(nodes,
                                                attractors,
                                                node_attractors_list,
                                                nodes_len,
                                                step)

    new_parent = new_parent.astype(int)

    new_nodes = new_nodes[~np.all(new_nodes == 0, axis=1)]
    new_normals = new_normals[~np.all(new_normals == 0, axis=1)]
    new_parent = new_parent[new_parent != INT_MAX]
    new_count = new_nodes.shape[0]

    return (np.concatenate((nodes, new_nodes)),
            np.concatenate((normals, new_normals)),
            attractors,
            new_parent,
            new_count)


n_attractors = 40000
attraction_distance = args.attraction_distance
kill_distance = args.kill_distance
step = args.segment_length

BIG_NUMBER = n_attractors * 10
INF = float('inf')
INT_MAX = np.iinfo(int).max

# attractors = random_sphere_points(5000, 10)
# nodes = random_sphere_points(15, 12)

attractors = read_ply(args.file_in)

nodes = attractors[np.argmin(attractors[:, 2])]

nodes = nodes[np.newaxis, :]

normals = np.ones_like(nodes)

age = np.zeros((nodes.shape[0],))

parent = np.full((nodes.shape[0],), INT_MAX, dtype=int)

iterations = args.iterations

for cycle in range(iterations):
    (nodes,
     normals,
     attractors,
     new_parent,
     new_count) = update(nodes, normals, attractors)

    age = np.concatenate((age, np.tile(cycle, new_count)))

    parent = np.concatenate((parent, new_parent))

    print(f"nodes {nodes.shape[0]}; attractors {attractors.shape[0]}", end="\r")

has_children = np.full((nodes.shape[0],), False)

for i in parent:
    if i != INT_MAX:
        has_children[int(i)] = True

min_thickness = 0.001
max_thickness = 0.1
thickness_a = 0.05
thickness_b = 0.02

thickness = np.full((nodes.shape[0],), min_thickness)

idx = np.where(has_children == False)[0]

for i in idx:
    node = i
    while parent[node] != INT_MAX:
        node_thickness = thickness[node]
        parent_thickness = thickness[parent[node]]

        if (parent_thickness < (node_thickness + thickness_a)):
            thickness[parent[node]] = node_thickness + thickness_b
            # print(f"node {parent[node]}: {thickness[parent[node]]}")
        node = parent[node]

# export frame by frame
for i in range(iterations):
    nodes_frame = np.zeros_like(nodes)
    idx = np.where(age <= i)[0]
    nodes_select = nodes[idx]
    nodes_frame[:nodes_select.shape[0]] = nodes_select
    # folder dir_out must exist
    write_ply(nodes_frame, normals, thickness, args.dir_out + f"/{i}.ply")

# plot_3d(nodes)
# plot_3d(normals)
# plot_3d(attractors)

