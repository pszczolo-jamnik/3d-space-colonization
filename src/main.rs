extern crate clap;
extern crate kdtree;
extern crate num_traits;
extern crate ply_rs_bw;

mod ply_utils;

use clap::Parser;
use kdtree::KdTree;
// use kdtree::ErrorKind;
use std::ops::Sub;

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

impl Vec3 {
    fn to_array(&self) -> [f32; 3] {
        [self.0, self.1, self.2]
    }

    fn norm(&self) -> f32 {
        (self.0.powi(2) + self.1.powi(2) + self.2.powi(2)).sqrt()
    }

    fn normalize(&self) -> Self {
        Self(
            self.0 / self.norm(),
            self.1 / self.norm(),
            self.2 / self.norm(),
        )
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

fn grow(node: Vec3, attractors: Vec<&Vec3>, step: f32) -> (Vec3, Vec3) {
    let distances: Vec<Vec3> = attractors
        .into_iter()
        .map(|attractor| *attractor - node)
        .collect();

    let distances_norm: Vec<Vec3> = distances.iter().map(|v| v.normalize()).collect();

    let distances_norm_sum: Vec3 = distances_norm
        .iter()
        .copied()
        .reduce(|acc, v| Vec3(acc.0 + v.0, acc.1 + v.1, acc.2 + v.2))
        .unwrap();

    // println!("{:?}", distances_norm_sum);

    let distances_norm_mean_norm = Vec3(
        distances_norm_sum.0 / (distances_norm.len() as f32),
        distances_norm_sum.1 / (distances_norm.len() as f32),
        distances_norm_sum.2 / (distances_norm.len() as f32),
    )
    .normalize();
    // println!("{:?}", distances_norm_mean_norm);

    (
        Vec3(
            node.0 + distances_norm_mean_norm.0 * step,
            node.1 + distances_norm_mean_norm.1 * step,
            node.2 + distances_norm_mean_norm.2 * step,
        ),
        distances_norm_mean_norm,
    )
}

fn main() {
    const ATTRACTION_DISTANCE: f32 = 0.3;
    const KILL_DISTANCE: f32 = 0.03;
    const SEGMENT_LENGTH: f32 = 0.01;
    const ITERATIONS: u32 = 10;

    let args = Args::parse();

    let mut attractors = ply_utils::read_ply(&args.file_in);

    // mother node (min Z)
    let nodes = attractors
        .iter()
        .min_by(|x, y| x.2.total_cmp(&y.2))
        .unwrap();
    let mut nodes = vec![nodes.clone()];
    println!("starting from {:?}", nodes);

    let mut kdtree = KdTree::new(3);

    kdtree.add(nodes[0].to_array(), 0).unwrap();
    kdtree.add(nodes[0].to_array(), 1).unwrap();
    kdtree.add(nodes[0].to_array(), 2).unwrap();

    let mut attractor_nodes: Vec<Option<&u32>> = attractors
        .iter()
        .map(|a| {
            let nearest = kdtree.nearest(&a.to_array(), 1, &euclidean).unwrap();
            assert_eq!(nearest.len(), 1);
            if nearest[0].0 <= ATTRACTION_DISTANCE {
                Some(nearest[0].1)
            } else {
                None
            }
        })
        .collect();

    for (i, node) in nodes.iter().enumerate() {
        let node_attractors: Vec<usize> = attractor_nodes
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| match value {
                None => None,
                Some(val) if *val == (i as u32) => Some(index),
                _ => None,
            })
            .collect();
        // println!("{:?}", node_attractors);
        if !node_attractors.is_empty() {
            // ! spawn thread
            let node_attractors: Vec<&Vec3> =
                node_attractors.iter().map(|&i| &attractors[i]).collect();
            ply_utils::write_ply(&args.file_out, node_attractors);
        }
    }
}
