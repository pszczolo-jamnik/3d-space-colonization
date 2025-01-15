use ply_rs_bw::parser;
use ply_rs_bw::ply;

use ply_rs_bw::ply::{
    Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
    ScalarType,
};
use ply_rs_bw::writer::Writer;

use super::{Node, Vec3};

impl ply::PropertyAccess for Vec3 {
    fn new() -> Self {
        Vec3(0.0, 0.0, 0.0)
    }
    fn set_property(&mut self, key: &String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.0 = v,
            ("y", ply::Property::Float(v)) => self.1 = v,
            ("z", ply::Property::Float(v)) => self.2 = v,
            // (k, _) => panic!("Vec3: Unexpected key/value combination: key: {}", k),
            (_, _) => {}
        }
    }
}

pub fn read_ply(file_name: &str) -> Vec<Vec3> {
    let f = std::fs::File::open(file_name).unwrap();
    let mut f = std::io::BufReader::new(f);

    let point_parser = parser::Parser::<Vec3>::new();

    let header = point_parser.read_header(&mut f).unwrap();

    let mut point_list = Vec::new();

    for (_ignore_key, element) in &header.elements {
        match element.name.as_ref() {
            "vertex" => {
                point_list = point_parser
                    .read_payload_for_element(&mut f, element, &header)
                    .unwrap();
            }
            _ => panic!("Unexpected element!"),
        }
    }
    point_list
}

pub fn write_ply(file_name: &str, nodes_in: Vec<&Node>) {
    let mut buf = std::fs::File::create(file_name).unwrap();

    let mut ply = {
        let mut ply = Ply::<DefaultElement>::new();
        ply.header.encoding = Encoding::BinaryLittleEndian;
        ply.header
            .comments
            .push("https://github.com/pszczolo-jamnik/3d-space-colonization".to_string());

        let mut point_element = ElementDef::new("vertex".to_string());
        for name in [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "thickness",
            "generation",
            "children",
        ] {
            let p = PropertyDef::new(name.to_string(), PropertyType::Scalar(ScalarType::Float));
            point_element.properties.add(p);
        }
        ply.header.elements.add(point_element);

        let mut points = Vec::new();

        for node in nodes_in {
            let mut point = DefaultElement::new();
            point.insert("x".to_string(), Property::Float(node.point.0));
            point.insert("y".to_string(), Property::Float(node.point.1));
            point.insert("z".to_string(), Property::Float(node.point.2));
            point.insert("nx".to_string(), Property::Float(node.vector.0));
            point.insert("ny".to_string(), Property::Float(node.vector.1));
            point.insert("nz".to_string(), Property::Float(node.vector.2));
            point.insert("thickness".to_string(), Property::Float(node.thickness));
            point.insert(
                "generation".to_string(),
                Property::Float(node.generation as f32),
            );
            point.insert(
                "children".to_string(),
                Property::Float(node.children as f32),
            );
            points.push(point);
        }

        ply.payload.insert("vertex".to_string(), points);

        ply.make_consistent().unwrap();
        ply
    };

    let w = Writer::new();
    let _written = w.write_ply(&mut buf, &mut ply).unwrap();
    // println!("{} bytes written", written);
}
