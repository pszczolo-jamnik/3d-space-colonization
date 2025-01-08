use ply_rs_bw::parser;
use ply_rs_bw::ply;

use ply_rs_bw::ply::{
    Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
    ScalarType,
};
use ply_rs_bw::writer::Writer;

use super::Vec3;

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
                    .read_payload_for_element(&mut f, &element, &header)
                    .unwrap();
            }
            _ => panic!("Unexpected element!"),
        }
    }
    point_list
}

pub fn write_ply(file_name: &str, points_in: Vec<&Vec3>) {
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
