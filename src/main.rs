extern crate clap;
extern crate kdtree;
extern crate num_traits;
extern crate ply_rs_bw;
extern crate rand;
extern crate rayon;

mod ply_utils;

use clap::{Args, Parser};
use kdtree::KdTree;
// use kdtree::ErrorKind;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::ops::Sub;

// use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Input PLY file
    #[arg(long)]
    file_in: String,
    /// Output PLY file
    #[arg(long)]
    file_out: String,
    #[arg(long)]
    iterations: u32,
    #[command(flatten)]
    origin: CliOrigin,
}

#[derive(Args, Debug)]
#[group(required = true, multiple = false)]
struct CliOrigin {
    /// Start from N random points
    #[arg(long, value_name = "N")]
    origin_random: Option<u8>,
    /// Start from minimum along (x | y | z)
    #[arg(long, value_name = "AXIS")]
    origin_min: Option<char>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec3(f32, f32, f32);

#[derive(Debug)]
struct Node {
    point: Vec3,
    vector: Vec3,
    thiccness: f32,
    generation: u32,
    // ! -> &Node
    parent: Option<usize>,
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

/// Returns new point and vector for parent
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
    nodes_tree: &KdTree<f32, usize, [f32; 3]>,
    nodes: &Vec<Node>,
    attractors: &'a Vec<Vec3>,
    attraction_distance: f32,
) -> Vec<Option<Vec<&'a Vec3>>> {
    let attractor_node: Vec<Option<usize>> = attractors
        .par_iter()
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
        .par_iter()
        .enumerate()
        .map(|(node_idx, _)| {
            attractor_node
                .iter()
                .enumerate()
                .filter_map(|(attractor_idx, node_idx_)| match node_idx_ {
                    None => None,
                    Some(i) if *i == node_idx => Some(attractor_idx),
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
    const KILL_DISTANCE: f32 = 0.005;
    const SEGMENT_LENGTH: f32 = 0.0015;
    const ZERO: Vec3 = Vec3(0.0, 0.0, 0.0);

    let args = Cli::parse();

    let mut attractors = ply_utils::read_ply(&args.file_in);
    let mut nodes = Vec::new();
    let mut kdtree = KdTree::new(3);

    // init nodes
    if let Some(n) = args.origin.origin_random {
        let mut rng = thread_rng();
        for _ in 0..n {
            let point = attractors[rng.gen_range(0..attractors.len())];
            nodes.push(Node {
                point,
                vector: ZERO,
                thiccness: 0.0,
                generation: 0,
                parent: None,
            });
        }
    } else {
        let axis = args.origin.origin_min.unwrap();
        let point = attractors
            .iter()
            .min_by(|x, y| match axis {
                'x' => x.0.total_cmp(&y.0),
                'y' => x.1.total_cmp(&y.1),
                'z' => x.2.total_cmp(&y.2),
                _ => panic!("wrong axis"),
            })
            .unwrap();
        nodes.push(Node {
            point: *point,
            vector: ZERO,
            thiccness: 0.0,
            generation: 0,
            parent: None,
        });
    }

    for (i, node) in nodes.iter().enumerate() {
        kdtree.add(node.point.to_array(), i).unwrap();
    }

    println!("🌱 {:?}", nodes);

    for iteration in 1..=args.iterations {
        println!(
            "🔄 {} -> nodes: {} attractors: {}",
            iteration,
            nodes.len(),
            attractors.len()
        );
        let nodes_attractors =
            get_nodes_attractors(&kdtree, &nodes, &attractors, ATTRACTION_DISTANCE);

        let mut grow_result = Vec::new();

        for (i, node_attractors) in nodes_attractors.into_iter().enumerate() {
            if let Some(growing_node_attractors) = node_attractors {
                let growing = grow(&nodes[i].point, &growing_node_attractors, SEGMENT_LENGTH);
                // i is new node's parent
                grow_result.push((i, growing));
            }
        }

        // Append nodes
        for (i, (_, (new_node, _))) in grow_result.iter().enumerate() {
            kdtree.add(new_node.to_array(), nodes.len() + i).unwrap();
        }

        let new_nodes: Vec<Node> = grow_result
            .iter()
            .map(|(parent, (new_point, _))| Node {
                point: *new_point,
                vector: ZERO,
                thiccness: 0.0,
                generation: iteration,
                parent: Some(*parent),
            })
            .collect();

        for (parent, (_, parent_vector)) in &grow_result {
            nodes[*parent].vector = *parent_vector;
        }

        nodes.extend(new_nodes);

        // Attractors are pruned as soon as any node enters its kill distance
        let mut victims: Vec<usize> = attractors
            .par_iter()
            .enumerate()
            .filter_map(|(i, attractor)| {
                let within = kdtree
                    .within(&attractor.to_array(), KILL_DISTANCE, &euclidean)
                    .unwrap();
                if !within.is_empty() {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        victims.sort_unstable();

        // remove in descending order
        for v in victims.iter().rev() {
            attractors.swap_remove(*v);
        }
    }

    let nodes_last_gen: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            if node.generation == args.iterations {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    const THICCNESS_A: f32 = 0.01;
    const THICCNESS_B: f32 = 0.07;

    // Experiment here
    for node_i in nodes_last_gen {
        let mut node_ref = &mut nodes[node_i];
        node_ref.thiccness = 0.01;
        while let Some(node_parent) = node_ref.parent {
            let thiccness = node_ref.thiccness;
            node_ref = &mut nodes[node_parent];
            if node_ref.thiccness < (thiccness + THICCNESS_B) {
                node_ref.thiccness = thiccness + THICCNESS_A;
            }
        }
    }

    ply_utils::write_ply(&args.file_out, nodes.iter().collect());
}
