extern crate clap;
extern crate kdtree;
extern crate num_traits;
extern crate ply_rs_bw;

use clap::Parser;

use ply_rs_bw::parser;
use ply_rs_bw::ply;

use ply_rs_bw::ply::{
    Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
    ScalarType,
};
use ply_rs_bw::writer::Writer;

use kdtree::KdTree;
// use kdtree::ErrorKind;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    file_in: String,
    #[arg(long)]
    file_out: String,
}

#[derive(Debug, Clone)]
struct Point(f32, f32, f32);

impl Point {
    fn to_array(&self) -> [f32; 3] {
        [self.0, self.1, self.2]
    }
}

impl ply::PropertyAccess for Point {
    fn new() -> Self {
        Point(0.0, 0.0, 0.0)
    }
    fn set_property(&mut self, key: &String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.0 = v,
            ("y", ply::Property::Float(v)) => self.1 = v,
            ("z", ply::Property::Float(v)) => self.2 = v,
            // (k, _) => panic!("Point: Unexpected key/value combination: key: {}", k),
            (_, _) => {}
        }
    }
}

fn read_ply(file_name: &str) -> Vec<Point> {
    let f = std::fs::File::open(file_name).unwrap();
    let mut f = std::io::BufReader::new(f);

    let point_parser = parser::Parser::<Point>::new();

    let header = point_parser.read_header(&mut f).unwrap();

    let mut point_list = Vec::new();

    for (_ignore_key, element) in &header.elements {
        match element.name.as_ref() {
            "vertex" => {
                point_list = point_parser
                    .read_payload_for_element(&mut f, &element, &header)
                    .unwrap();
            }
            _ => panic!("Unexpected element!"),
        }
    }
    point_list
}

fn write_ply(file_name: &str, points_in: Vec<&Point>) {
    let mut buf = std::fs::File::create(file_name).unwrap();

    let mut ply = {
        let mut ply = Ply::<DefaultElement>::new();
        ply.header.encoding = Encoding::BinaryLittleEndian;
        ply.header.comments.push("Ferris".to_string());

        let mut point_element = ElementDef::new("point".to_string());
        let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        ply.header.elements.add(point_element);

        let mut points = Vec::new();

        for p in points_in {
            let mut point = DefaultElement::new();
            point.insert("x".to_string(), Property::Float(p.0));
            point.insert("y".to_string(), Property::Float(p.1));
            point.insert("z".to_string(), Property::Float(p.2));
            points.push(point);
        }

        ply.payload.insert("point".to_string(), points);

        ply.make_consistent().unwrap();
        ply
    };

    let w = Writer::new();
    let written = w.write_ply(&mut buf, &mut ply).unwrap();
    println!("{} bytes written", written);
}

fn euclidean<T: num_traits::Float>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(T::zero(), ::std::ops::Add::add)
        .sqrt()
}

fn main() {
    const ATTRACTION_DISTANCE: f32 = 0.3;
    const KILL_DISTANCE: f32 = 0.03;
    const SEGMENT_LENGTH: f32 = 0.01;
    const ITERATIONS: u32 = 10;

    let args = Args::parse();

    let mut attractors = read_ply(&args.file_in);

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
            let node_attractors: Vec<&Point> =
                node_attractors.iter().map(|&i| &attractors[i]).collect();
            write_ply(&args.file_out, node_attractors);
        }
    }
}
