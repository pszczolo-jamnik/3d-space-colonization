extern crate clap;
extern crate kdtree;
extern crate num_traits;
extern crate ply_rs_bw;

mod ply_utils;

use clap::Parser;
use kdtree::KdTree;
// use kdtree::ErrorKind;
use std::ops::Sub;
use std::thread;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    file_in: String,
    #[arg(long)]
    file_out: String,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec3(f32, f32, f32);

#[derive(Debug)]
struct Node {
    point: Vec3,
    vector: Vec3,
    thiccness: f32,
    age: u32,
}

impl Vec3 {
    fn to_array(&self) -> [f32; 3] {
        [self.0, self.1, self.2]
    }

    fn norm(&self) -> f32 {
        (self.0.powi(2) + self.1.powi(2) + self.2.powi(2)).sqrt()
    }

    fn normalize(&self) -> Self {
        match self.norm() {
            // edge case
            0.0 => Self(0.0, 0.0, 0.0),
            x => Self(self.0 / x, self.1 / x, self.2 / x),
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

fn euclidean<T: num_traits::Float>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(T::zero(), ::std::ops::Add::add)
        .sqrt()
}

fn grow(node_pt: &Vec3, attractors: &Vec<&Vec3>, step: f32) -> (Vec3, Vec3) {
    let distances: Vec<Vec3> = attractors
        .iter()
        .map(|attractor| **attractor - *node_pt)
        .collect();

    let distances_norm: Vec<Vec3> = distances.iter().map(|v| v.normalize()).collect();

    let distances_norm_sum: Vec3 = distances_norm
        .iter()
        .copied()
        .reduce(|acc, v| Vec3(acc.0 + v.0, acc.1 + v.1, acc.2 + v.2))
        .unwrap();

    let distances_norm_mean_norm = Vec3(
        distances_norm_sum.0 / (distances_norm.len() as f32),
        distances_norm_sum.1 / (distances_norm.len() as f32),
        distances_norm_sum.2 / (distances_norm.len() as f32),
    )
    .normalize();

    (
        Vec3(
            node_pt.0 + distances_norm_mean_norm.0 * step,
            node_pt.1 + distances_norm_mean_norm.1 * step,
            node_pt.2 + distances_norm_mean_norm.2 * step,
        ),
        distances_norm_mean_norm,
    )
}

/// First, associate each attractor with the single closest node
/// within pre-defined attraction distance. Then, get all influencing
/// attractors for each node
fn get_nodes_attractors<'a>(
    nodes_tree: &KdTree<f32, u32, [f32; 3]>,
    nodes: &Vec<Node>,
    attractors: &'a Vec<Vec3>,
    attraction_distance: f32,
) -> Vec<Option<Vec<&'a Vec3>>> {
    let attractor_node: Vec<Option<u32>> = attractors
        .iter()
        .map(|a| {
            let nearest = nodes_tree.nearest(&a.to_array(), 1, &euclidean).unwrap();
            assert_eq!(nearest.len(), 1);
            if nearest[0].0 <= attraction_distance {
                Some(*nearest[0].1)
            } else {
                None
            }
        })
        .collect();

    nodes
        .iter()
        .enumerate()
        .map(|(node_idx, _)| {
            attractor_node
                .iter()
                .enumerate()
                .filter_map(|(attractor_idx, node_idx_)| match node_idx_ {
                    None => None,
                    Some(i) if (*i as usize) == node_idx => Some(attractor_idx),
                    _ => None,
                })
                .map(|i| &attractors[i])
                .collect()
        })
        .map(|v: Vec<&Vec3>| if v.is_empty() { None } else { Some(v) })
        .collect()
}

fn main() {
    const ATTRACTION_DISTANCE: f32 = 0.02;
    const KILL_DISTANCE: f32 = 0.01;
    const SEGMENT_LENGTH: f32 = 0.002;
    const ITERATIONS: u32 = 150;
    const ZERO: Vec3 = Vec3(0.0, 0.0, 0.0);

    let args = Args::parse();

    let mut attractors = ply_utils::read_ply(&args.file_in);

    // mother node (min Z)
    let origin = attractors
        .iter()
        .min_by(|x, y| x.2.total_cmp(&y.2))
        .unwrap();
    let mut nodes = vec![Node {
        point: *origin,
        vector: ZERO,
        thiccness: 0.001,
        age: 0,
    }];

    println!("🌱 {:?}", nodes);

    let mut kdtree = KdTree::new(3);
    kdtree.add(nodes[0].point.to_array(), 0).unwrap();

    for iteration in 1..=ITERATIONS {
        println!(
            "⌛ {} nodes {} attractors {}",
            iteration,
            nodes.len(),
            attractors.len()
        );
        let nodes_attractors =
            get_nodes_attractors(&kdtree, &nodes, &attractors, ATTRACTION_DISTANCE);

        let mut thread_result = Vec::new();

        // Run grow() threaded
        thread::scope(|s| {
            let mut thread_handles = Vec::new();
            for (i, node_attractors) in nodes_attractors.into_iter().enumerate() {
                if let Some(growing_node_attractors) = node_attractors {
                    let nodes_i = &nodes[i].point;
                    let handle =
                        s.spawn(move || grow(nodes_i, &growing_node_attractors, SEGMENT_LENGTH));
                    thread_handles.push(handle);
                }
            }
            for handle in thread_handles {
                thread_result.push(handle.join().unwrap());
            }
        });

        // Append nodes
        for (i, (new_node, _)) in thread_result.iter().enumerate() {
            kdtree
                .add(new_node.to_array(), (nodes.len() + i) as u32)
                .unwrap();
        }
        let (new_points, parent_vectors): (Vec<_>, Vec<_>) = thread_result.into_iter().unzip();
        nodes.append(
            &mut new_points
                .iter()
                // !
                .map(|p| Node {
                    point: *p,
                    vector: ZERO,
                    thiccness: 0.0,
                    age: iteration,
                })
                .collect(),
        );

        // Attractors are pruned as soon as any node enters its kill distance
        let mut victims = Vec::new();
        for (i, attractor) in attractors.iter().enumerate() {
            let within = kdtree
                .within(&attractor.to_array(), KILL_DISTANCE, &euclidean)
                .unwrap();
            if !within.is_empty() {
                // println!("🔪 {:?}", within);
                victims.push(i);
            }
        }
        for v in victims {
            attractors.swap_remove(v);
        }
    }
    ply_utils::write_ply(&args.file_out, nodes.iter().map(|n| &n.point).collect());
    // ply_utils::write_ply(&args.file_out, attractors.iter().collect());
}
