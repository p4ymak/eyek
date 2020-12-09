use bvh::aabb::{Bounded, AABB};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::BVH;
use bvh::nalgebra::distance;
use bvh::nalgebra::geometry::{Isometry3, Perspective3, Quaternion, Translation3, UnitQuaternion};
use bvh::nalgebra::{Point3, Vector3};
use bvh::ray::Ray;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use obj;
use rayon::prelude::*;
use serde_derive::Deserialize;
use serde_json;
use std::env;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
struct Tris2D {
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,
}
impl Tris2D {
    fn has_point(&self, pt: Point3<f32>) -> bool {
        fn sign(a: Point3<f32>, b: Point3<f32>, c: Point3<f32>) -> f32 {
            (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y)
        }
        let d1 = sign(pt, self.a, self.b);
        let d2 = sign(pt, self.b, self.c);
        let d3 = sign(pt, self.c, self.a);
        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        !(has_neg && has_pos)
    }
    fn bounds(&self) -> [f32; 4] {
        let mut coords_x = [self.a.x, self.b.x, self.c.x];
        let mut coords_y = [self.a.y, self.b.y, self.c.y];
        coords_x.sort_by(|i, j| i.partial_cmp(j).unwrap());
        coords_y.sort_by(|i, j| i.partial_cmp(j).unwrap());
        //return min_x, min_y, max_x, max_y of triangle
        [coords_x[0], coords_y[0], coords_x[2], coords_y[2]]
    }
    fn cartesian_to_barycentric(&self, pt: Point3<f32>) -> Point3<f32> {
        let v0 = self.b - self.a;
        let v1 = self.c - self.a;
        let v2 = pt - self.a;
        let den = 1.0 / (v0.x * v1.y - v1.x * v0.y);
        let v = (v2.x * v1.y - v1.x * v2.y) * den;
        let w = (v0.x * v2.y - v2.x * v0.y) * den;
        let u = 1.0 - v - w;
        Point3::new(u, v, w)
    }
    fn barycentric_to_cartesian(&self, pt: Point3<f32>) -> Point3<f32> {
        let x = pt.x * self.a.x + pt.y * self.b.x + pt.z * self.c.x;
        let y = pt.x * self.a.y + pt.y * self.b.y + pt.z * self.c.y;
        let z = 0.0; //pt.x * self.a.z + pt.y * self.b.z + pt.z * self.c.z;
        Point3::new(x, y, z)
    }
}

#[derive(Debug, Clone)]
struct Tris3D {
    v_3d: [Point3<f32>; 3],
    v_uv: Tris2D,
    min: Point3<f32>,
    mid: Point3<f32>,
    max: Point3<f32>,
    node_index: usize,
}
impl Bounded for Tris3D {
    fn aabb(&self) -> AABB {
        AABB::with_bounds(self.min, self.max)
    }
}
impl BHShape for Tris3D {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[derive(Debug)]
struct Mesh {
    tris: Vec<Tris3D>,
    //aabb: [[f32; 3]; 2],
}

#[derive(Debug, Deserialize)]
struct VecCameraJSON {
    data: Vec<CameraJSON>,
}
#[derive(Debug, Deserialize)]
struct CameraJSON {
    cameraPosition: Vec<f32>,
    cameraRotation: Vec<f32>,
    imageName: String,
}
#[derive(Debug)]
struct CameraRaw {
    id: usize,
    pos: [f32; 3],
    rot: UnitQuaternion<f32>,
    img_path: String,
}

fn load_meshes(path_obj: &str) -> Vec<Tris3D> {
    let data = obj::Obj::load(path_obj).unwrap().data;
    let mut tris = Vec::<Tris3D>::new();
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
                    let tr_mid_x = (tr_min_x + tr_max_x) / 2.0;
                    let tr_mid_y = (tr_min_y + tr_max_y) / 2.0;
                    let tr_mid_z = (tr_min_z + tr_max_z) / 2.0;
                    tris.push(Tris3D {
                        v_3d: [vs_pos[0], vs_pos[1], vs_pos[2]],
                        v_uv: Tris2D {
                            a: vs_uv[0],
                            b: vs_uv[1],
                            c: vs_uv[2],
                        },
                        min: Point3::new(tr_min_x, tr_min_y, tr_min_z),
                        mid: Point3::new(tr_mid_x, tr_mid_y, tr_mid_z),
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

fn load_cameras(path_json_imgs: &str) -> Vec<CameraRaw> {
    let file_json = fs::File::open(Path::new(path_json_imgs).join("imageData.json")).unwrap();
    let cameras_json: VecCameraJSON = serde_json::from_reader(file_json).unwrap();
    let mut cameras = Vec::<CameraRaw>::new();
    let mut id = 0;
    for cam in cameras_json.data {
        id += 1;
        let pos = [
            cam.cameraPosition[0],
            cam.cameraPosition[1],
            -cam.cameraPosition[2],
        ];
        //This quaternion as a 4D vector of coordinates in the [ x, y, z, w ] storage order.
        let rot = UnitQuaternion::from_quaternion(Quaternion::new(
            //KOCTbIJIb
            //cam.cameraRotation[0],
            -cam.cameraRotation[3],
            -cam.cameraRotation[2],
            cam.cameraRotation[1],
            cam.cameraRotation[0],
        ));

        let img_path = Path::new(path_json_imgs)
            .join(cam.imageName)
            .to_string_lossy()
            .into_owned();

        cameras.push(CameraRaw {
            id,
            pos,
            rot,
            img_path,
        });
    }

    cameras
}

fn cast_pixels_rays(
    camera_raw: CameraRaw,
    faces: &Vec<Tris3D>,
    bvh: &BVH,
    mut texture: &mut RgbaImage,
) {
    let img = image::open(camera_raw.img_path).unwrap();
    let width = img.dimensions().0 as usize;
    let height = img.dimensions().1 as usize;
    let ratio = width as f32 / height as f32;
    //let fovx = 0.541 as f32;
    let fovy = 0.72; //((fovx).atan() * ratio).tan();
    let [cam_x, cam_y, cam_z] = camera_raw.pos;
    let rot = camera_raw.rot;
    let cam_tr = Translation3::new(cam_x, cam_y, cam_z);
    let cam_pos = Point3::new(cam_x, cam_y, cam_z);
    let iso_targ = Isometry3::from_parts(cam_tr, rot);
    let cam_target = iso_targ.transform_point(&Point3::new(0.0, 0.0, 1.0));
    let iso = Isometry3::face_towards(&cam_pos, &cam_target, &Vector3::y());
    let perspective = Perspective3::new(ratio, fovy, 0.1, 100.0);
    let mut checked_pixels: Vec<Vec<bool>> = Vec::with_capacity(width);
    for _ in 0..width {
        checked_pixels.push(vec![false; height]);
    }

    //DEBUG
    //let polycount = faces.len();
    //let mut ray_casts = 0;
    //let mut test_img = RgbaImage::new(width as u32, height as u32);
    //let ray_target_test =
    //    iso.transform_point(&perspective.unproject_point(&Point3::new(0.0, 0.0, 1.0)));
    //println!("CAM_POS:\n{:?}\nCAM_TAR:{:?}", &cam_pos, ray_target_test);

    for y in 0..height {
        for x in 0..width {
            if !checked_pixels[x][y] {
                let ray_origin = iso.transform_point(&perspective.unproject_point(&Point3::new(
                    (x as f32 / width as f32) * 2.0 - 1.0,
                    (y as f32 / height as f32) * 2.0 - 1.0,
                    -1.0,
                )));

                let ray_target_pt =
                    iso.transform_point(&perspective.unproject_point(&Point3::new(
                        (x as f32 / width as f32) * 2.0 - 1.0,
                        (y as f32 / height as f32) * 2.0 - 1.0,
                        1.0,
                    )));
                //println!("{:?}\n{:?}\n{:?}\n", pos_pt, ray_origin, ray_target_pt);
                let ray = Ray::new(
                    ray_origin,
                    Vector3::new(
                        ray_target_pt.x, // - ray_origin.x,
                        ray_target_pt.y, // - ray_origin.y,
                        ray_target_pt.z, // - ray_origin.z,
                    ),
                );

                let collisions = bvh.traverse(&ray, &faces);
                //if collisions.len() == 0 {
                //    checked_pixels[x][y] = true;
                //    continue;
                //}

                //test_img.put_pixel(x as u32, height as u32 - y as u32 - 1, Rgba([0, 0, 0, 255]));
                //ray_casts += 1;
                for face in closest_faces(collisions, cam_pos) {
                    face_img_to_uv(
                        &face,
                        &iso,
                        &perspective,
                        &mut checked_pixels,
                        &img,
                        &mut texture,
                    );
                }
            }
        }
    }
    //test_img
    //    .save("/home/p4/Work/Phygitalism/201127_Raskrasser/tests/test_1/dumpIot/test.png")
    //    .unwrap();
    //println!("Collisions: {:?}/{:?}", ray_casts, polycount);
}

fn closest_faces(faces: Vec<&Tris3D>, pt: Point3<f32>) -> Vec<&Tris3D> {
    if faces.len() <= 1 {
        return faces;
    }
    let mut closest = Vec::<(f32, &Tris3D)>::new();
    for f in faces {
        closest.push((distance(&f.mid, &pt), f));
    }
    closest.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut range = closest[0]
        .1
        .v_3d
        .iter()
        .map(|&p| distance(&p, &closest[0].1.mid))
        .collect::<Vec<f32>>();
    range.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    let epsilon = range.first().unwrap();

    closest
        .iter()
        .filter(|f| f.0 - closest[0].0 <= *epsilon)
        .map(|f| f.1)
        .collect()
}

fn _closest_faces(faces: Vec<&Tris3D>, _pt: Point3<f32>) -> Vec<&Tris3D> {
    if faces.len() == 1 {
        return faces;
    }
    return vec![faces[0], faces[1]];
}

fn mix_colors(source: Rgba<u8>, target: &Rgba<u8>) -> Rgba<u8> {
    let sr = source[0];
    let sg = source[1];
    let sb = source[2];
    let tr = target[0];
    let tg = target[1];
    let tb = target[2];
    let ta = target[3];

    if ta == 0 {
        return source;
    } else {
        return Rgba([sr / 2 + tr / 2, sg / 2 + tg / 2, sb / 2 + tb / 2, 255]);
    }
}

fn face_img_to_uv(
    face: &Tris3D,
    iso: &Isometry3<f32>,
    perspective: &Perspective3<f32>,
    checked_pixels: &mut Vec<Vec<bool>>,
    img: &DynamicImage,
    texture: &mut RgbaImage,
) {
    let uv_width = texture.dimensions().0 as f32;
    let uv_height = texture.dimensions().1 as f32;
    let uv_min_x = (face.v_uv.bounds()[0] * uv_width).floor() as usize;
    let uv_min_y = (face.v_uv.bounds()[1] * uv_height).floor() as usize;
    let uv_max_x = (face.v_uv.bounds()[2] * uv_width).ceil() as usize;
    let uv_max_y = (face.v_uv.bounds()[3] * uv_height).ceil() as usize;

    let cam_width = img.dimensions().0 as f32;
    let cam_height = img.dimensions().1 as f32;

    let face_cam = Tris2D {
        a: perspective.project_point(&iso.inverse_transform_point(&face.v_3d[0])),
        b: perspective.project_point(&iso.inverse_transform_point(&face.v_3d[1])),
        c: perspective.project_point(&iso.inverse_transform_point(&face.v_3d[2])),
    };
    // println!("{:?}", face_cam);
    for v in uv_min_y..uv_max_y {
        for u in uv_min_x..uv_max_x {
            let p_uv = Point3::new(u as f32 / uv_width as f32, v as f32 / uv_height as f32, 0.0);
            if face.v_uv.has_point(p_uv) {
                let p_bary = face.v_uv.cartesian_to_barycentric(p_uv);
                let p_cam = face_cam.barycentric_to_cartesian(p_bary);

                if face_cam.has_point(p_cam)
                    && p_cam.x > -1.0
                    && p_cam.y > -1.0
                    && p_cam.x < 1.0
                    && p_cam.y < 1.0
                {
                    // println!("{:?}", p_cam);
                    let cam_x = (cam_width * (p_cam.x + 1.0) / 2.0) as u32;
                    let cam_y = (cam_height * (p_cam.y + 1.0) / 2.0) as u32;
                    // println!("x: {:?}\ny: {:?}\n", cam_x, cam_y);
                    if cam_x < cam_width as u32
                        && cam_x > 0
                        && cam_y < cam_height as u32
                        && cam_y > 0
                    {
                        let source_color = img.get_pixel(cam_x, (cam_height - 1.0) as u32 - cam_y);
                        let current_color =
                            texture.get_pixel(u as u32, uv_height as u32 - v as u32);
                        let target_color = mix_colors(source_color, current_color);

                        texture.put_pixel(u as u32, uv_height as u32 - v as u32, target_color);
                        checked_pixels[cam_x as usize][cam_y as usize] = true;
                    }
                }
            }
        }
    }
}

fn blend_pixels(texture: &RgbaImage, x: u32, y: u32) -> Rgba<u8> {
    let ways = [
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
    ];
    let bx = texture.dimensions().0 as i32;
    let by = texture.dimensions().1 as i32;
    let mut neibs_count = 0;
    let mut r = 0 as u32;
    let mut g = 0 as u32;
    let mut b = 0 as u32;

    for way in ways.iter() {
        let col = texture.get_pixel(
            ((x as i32 + way[0] % bx) as u32).min(bx as u32 - 1),
            ((y as i32 + way[1] % by) as u32).min(by as u32 - 1),
        );
        if col[3] != 0 {
            neibs_count += 1;
            r += col[0] as u32;
            g += col[1] as u32;
            b += col[2] as u32;
        }
    }
    if neibs_count != 0 {
        r /= neibs_count;
        g /= neibs_count;
        b /= neibs_count;
    }
    Rgba([r as u8, g as u8, b as u8, 255])
}

fn fill_empty_pixels(texture: &mut RgbaImage) {
    let (width, height) = texture.dimensions();
    for v in 0..(height as usize) {
        for u in 0..(width as usize) {
            let current_color = *texture.get_pixel(u as u32, v as u32);
            if current_color[3] == 0 {
                let blended_color = blend_pixels(&texture, u as u32, v as u32);
                if blended_color[3] != 0 {
                    texture.put_pixel(u as u32, v as u32, blended_color)
                }
            }
        }
    }
    //texture
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let path_obj = &args[1];
    let path_json_imgs = &args[2];
    let path_texture = &args[3];
    let img_res = args[4].parse::<u32>().unwrap();

    let mut faces: Vec<Tris3D> = load_meshes(path_obj);
    let cameras = load_cameras(path_json_imgs);
    let bvh = BVH::build(&mut faces);

    //let mut ccount = 0;

    let textures: Vec<RgbaImage> = cameras
        .into_par_iter()
        .map(|cam| {
            let mut texture = RgbaImage::new(img_res, img_res);
            let id = cam.id;
            cast_pixels_rays(cam, &faces, &bvh, &mut texture);
            //ccount += 1;
            println!("Finished cam: {:?}", id);
            //println!("Finished Cam: {:?}", ccount);
            texture
        })
        .collect();

    let mut texture = RgbaImage::new(img_res, img_res);
    for y in 0..img_res {
        for x in 0..img_res {
            let mut n = 0;
            let mut r: usize = 0;
            let mut g: usize = 0;
            let mut b: usize = 0;

            for part in &textures {
                let col = part.get_pixel(x, y);
                if col[3] != 0 {
                    n += 1;
                    r += col[0] as usize;
                    g += col[1] as usize;
                    b += col[2] as usize;
                }
            }
            if n > 0 {
                r /= n;
                g /= n;
                b /= n;
                texture.put_pixel(x, y, Rgba([r as u8, g as u8, b as u8, 255]))
            }
        }
    }

    fill_empty_pixels(&mut texture);
    println!("Filled empty pixels");

    texture.save(Path::new(path_texture)).unwrap();
    println!("Texture saved!");
}
