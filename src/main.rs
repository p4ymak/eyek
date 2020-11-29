use bvh::aabb::{Bounded, AABB};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::BVH;
use bvh::nalgebra::{Point3, Vector3};
use bvh::ray::Ray;
use obj;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone)]
struct Tris {
    v_3d: [Point3<f32>; 3],
    v_uv: [Point3<f32>; 3],
    min: Point3<f32>,
    max: Point3<f32>,
    node_index: usize,
}
impl Bounded for Tris {
    fn aabb(&self) -> AABB {
        AABB::with_bounds(self.min, self.max)
    }
}
impl BHShape for Tris {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[derive(Debug)]
struct Mesh {
    tris: Vec<Tris>,
    //aabb: [[f32; 3]; 2],
}

fn load_meshes(path_from: &str) -> Vec<Tris> {
    let data = obj::Obj::load(path_from).unwrap().data;

    let mut tris = Vec::<Tris>::new();
    let mut tris_id: usize = 0;
    for obj in data.objects {
        for group in obj.groups {
            for poly in group.polys {
                let mut tr_min_x = f32::MAX;
                let mut tr_max_x = f32::MIN;
                let mut tr_min_y = f32::MAX;
                let mut tr_max_y = f32::MIN;
                let mut tr_min_z = f32::MAX;
                let mut tr_max_z = f32::MIN;
                let mut vs_pos = Vec::<Point3<f32>>::new();
                let mut vs_uv = Vec::<Point3<f32>>::new();

                for vert in poly.0 {
                    let x = data.position[vert.0][0];
                    let y = data.position[vert.0][1];
                    let z = data.position[vert.0][2];
                    let u = data.texture[vert.1.unwrap()][0];
                    let v = data.texture[vert.1.unwrap()][1];

                    vs_pos.push(Point3::new(x, y, z));
                    vs_uv.push(Point3::new(u, v, 0.0));

                    tr_min_x = tr_min_x.min(x);
                    tr_max_x = tr_max_x.max(x);
                    tr_min_y = tr_min_y.min(y);
                    tr_max_y = tr_max_y.max(y);
                    tr_min_z = tr_min_z.min(z);
                    tr_max_z = tr_max_z.max(z);
                }
                if vs_pos.len() >= 3 {
                    tris.push(Tris {
                        v_3d: [vs_pos[0], vs_pos[1], vs_pos[2]],
                        v_uv: [vs_uv[0], vs_uv[1], vs_uv[2]],
                        min: Point3::new(tr_min_x, tr_min_y, tr_min_z),
                        max: Point3::new(tr_max_x, tr_max_y, tr_max_z),
                        node_index: tris_id,
                    });
                    tris_id += 1;
                }
            }
        }
    }
    return tris;
}

fn main() {
    let path_from = "/home/p4ymak/Work/Phygitalism/201109_Projector/from_und3ve10p3d/test_0/Scan/TestScan42Scan.obj";
    let mut faces: Vec<Tris> = load_meshes(path_from);
    let bvh = BVH::build(&mut faces);

    println!("{:?}", faces.len());
}
