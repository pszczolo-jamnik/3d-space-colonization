extern crate clap;
extern crate kdtree;
extern crate num_traits;
extern crate ply_rs_bw;
extern crate rand;
extern crate rayon;

mod ply_utils;

use clap::{Args, Parser};
use kdtree::KdTree;
use rand::prelude::*;
use rayon::prelude::*;
use std::ops::Sub;

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
    /// Only nodes within this distance around an attractor can be associated
    /// with that attractor. Large attraction distances mean smoother and more
    /// subtle branch curves, but at a performance cost
    #[arg(long)]
    attraction_distance: f32,
    /// An attractor may be removed if one or more nodes are within
    /// this distance around it
    #[arg(long)]
    kill_distance: f32,
    /// The distance between nodes as the network grows. Larger values
    /// mean better performance, but choppier and sharper branch curves
    #[arg(long)]
    segment_length: f32,
    #[command(flatten)]
    origin: CliOrigin,
    /// Randomize growth 0.1 - 1.0
    #[arg(long, value_name = "R")]
    random: Option<f32>,
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
    thickness: f32,
    generation: u32,
    // ! -> &Node
    parent: Option<usize>,
    children: u8,
}

impl Vec3 {
    fn to_array(self) -> [f32; 3] {
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

    fn is_finite(&self) -> bool {
        self.0.is_finite() && self.1.is_finite() && self.2.is_finite()
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
fn grow(
    node_pt: &Vec3,
    attractors: &Vec<&Vec3>,
    step: f32,
    random: Option<f32>,
    rng: &mut ThreadRng,
) -> (Vec3, Vec3) {
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

    let (rx, ry, rz): (f32, f32, f32) = if let Some(r) = random {
        (
            rng.gen_range(0.0..r),
            rng.gen_range(0.0..r),
            rng.gen_range(0.0..r),
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    (
        Vec3(
            node_pt.0 + (distances_norm_mean_norm.0 + rx) * step,
            node_pt.1 + (distances_norm_mean_norm.1 + ry) * step,
            node_pt.2 + (distances_norm_mean_norm.2 + rz) * step,
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
        .map(|(node_idx, node)| {
            if node.children < 5 {
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
            } else {
                vec![]
            }
        })
        .map(|v: Vec<&Vec3>| if v.is_empty() { None } else { Some(v) })
        .collect()
}

fn main() {
    const ZERO: Vec3 = Vec3(0.0, 0.0, 0.0);

    let args = Cli::parse();

    let mut attractors: Vec<Option<Vec3>> = ply_utils::read_ply(&args.file_in)
        .iter()
        .inspect(|a| {
            if !a.is_finite() {
                println!("⚠️ input contains NaN of inf values");
            }
        })
        .filter(|a| a.is_finite())
        .map(|a| Some(*a))
        .collect();
    let mut nodes = Vec::new();
    let mut kdtree_n = KdTree::new(3);
    let mut kdtree_a = KdTree::new(3);

    // init nodes
    if let Some(n) = args.origin.origin_random {
        let mut rng = thread_rng();
        for _ in 0..n {
            let point = attractors[rng.gen_range(0..attractors.len())].unwrap();
            nodes.push(Node {
                point,
                vector: ZERO,
                thickness: 0.0,
                generation: 0,
                parent: None,
                children: 0,
            });
        }
    } else {
        let axis = args.origin.origin_min.unwrap();
        let point = attractors
            .iter()
            .map(|a| match a {
                Some(p) => p,
                None => unreachable!(),
            })
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
            thickness: 0.0,
            generation: 0,
            parent: None,
            children: 0,
        });
    }

    for (i, node) in nodes.iter().enumerate() {
        kdtree_n.add(node.point.to_array(), i).unwrap();
    }

    // init attractors
    for (i, attractor) in attractors.iter().enumerate() {
        if let Some(point) = attractor {
            kdtree_a.add(point.to_array(), i).unwrap();
        } else {
            unreachable!();
        }
    }

    let mut rng = thread_rng();

    println!("🌱 {:?}", nodes);

    for iteration in 1..=args.iterations {
        println!("🔄 {} -> nodes: {}", iteration, nodes.len(),);

        const GEN_BEFORE: u32 = 5;
        let mut attractors_within: Vec<usize> = nodes
            .par_iter()
            .filter_map(|node| {
                if (node.generation >= iteration - GEN_BEFORE) || iteration < GEN_BEFORE {
                    Some(node)
                } else {
                    None
                }
            })
            .map(|node| {
                let (_, i): (Vec<_>, Vec<&usize>) = kdtree_a
                    .within(&node.point.to_array(), args.attraction_distance, &euclidean)
                    .unwrap()
                    .into_iter()
                    .unzip();
                i
            })
            .flatten()
            .map(|i| *i)
            .collect();

        attractors_within.sort_unstable();
        attractors_within.dedup();

        let attractors_within: Vec<Vec3> = attractors_within
            .par_iter()
            .filter_map(|i| attractors[*i])
            .collect();

        let nodes_attractors = get_nodes_attractors(
            &kdtree_n,
            &nodes,
            &attractors_within,
            args.attraction_distance,
        );

        assert_eq!(nodes.len(), nodes_attractors.len());

        let mut grow_result = Vec::new();

        for (i, node_attractors) in nodes_attractors.into_iter().enumerate() {
            if let Some(growing_node_attractors) = node_attractors {
                let growing = grow(
                    &nodes[i].point,
                    &growing_node_attractors,
                    args.segment_length,
                    args.random,
                    &mut rng,
                );
                // i is new node's parent
                grow_result.push((i, growing));
            }
        }

        // Append nodes
        for (i, (_, (new_node, _))) in grow_result.iter().enumerate() {
            kdtree_n.add(new_node.to_array(), nodes.len() + i).unwrap();
        }

        let new_nodes: Vec<Node> = grow_result
            .iter()
            .map(|(parent, (new_point, _))| Node {
                point: *new_point,
                vector: ZERO,
                thickness: 0.0,
                generation: iteration,
                parent: Some(*parent),
                children: 0,
            })
            .collect();

        for (parent, (_, parent_vector)) in &grow_result {
            nodes[*parent].vector = *parent_vector;
            nodes[*parent].children += 1;
        }

        nodes.extend(new_nodes);

        // Attractors are pruned as soon as any node enters its kill distance
        let victims: Vec<usize> = attractors
            .par_iter()
            .enumerate()
            .filter_map(|(i, attractor)| {
                if let Some(a) = attractor {
                    let within = kdtree_n
                        .within(&a.to_array(), args.kill_distance, &euclidean)
                        .unwrap();
                    if !within.is_empty() {
                        Some(i)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        for v in victims.iter() {
            attractors[*v] = None;
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

    const THICKNESS_A: f32 = 0.01;
    const THICKNESS_B: f32 = 0.05;

    // Experiment here
    for node_i in nodes_last_gen {
        let mut node_ref = &mut nodes[node_i];
        node_ref.thickness = 0.01;
        while let Some(node_parent) = node_ref.parent {
            let thickness = node_ref.thickness;
            node_ref = &mut nodes[node_parent];
            if node_ref.thickness < (thickness + THICKNESS_B) {
                node_ref.thickness = thickness + THICKNESS_A;
            }
        }
    }

    ply_utils::write_ply(&args.file_out, nodes.iter().collect());
}
