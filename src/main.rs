use bvh::aabb::{Bounded, AABB};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::BVH;
use bvh::nalgebra::geometry::{
    Isometry3, Orthographic3, Perspective3, Translation3, UnitQuaternion,
};
use bvh::nalgebra::{Point3, Vector3};
use bvh::ray::Ray;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use rayon::prelude::*;
use serde_derive::Deserialize;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::Path;
use triangle::{Point, Triangle};

enum Projection {
    Persp(Perspective3<f32>),
    Ortho(Orthographic3<f32>),
}

fn point3_to_point(pt: Point3<f32>) -> Point<f32> {
    Point {
        x: pt.x,
        y: pt.y,
        z: pt.z,
    }
}
fn point_to_point3(pt: Point<f32>) -> Point3<f32> {
    Point3::new(pt.x, pt.y, pt.z)
}

fn project_point_to_cam(
    pt: Point<f32>,
    iso: &Isometry3<f32>,
    projection: &Projection,
) -> Point<f32> {
    match projection {
        Projection::Persp(pr) => {
            point3_to_point(pr.project_point(&iso.inverse_transform_point(&point_to_point3(pt))))
        }
        Projection::Ortho(pr) => {
            point3_to_point(pr.project_point(&iso.inverse_transform_point(&point_to_point3(pt))))
        }
    }
}

fn uv_udim(u: f32, v: f32) -> u32 {
    1000 + u.ceil() as u32 + 10 * v.floor() as u32
}
#[derive(Debug, Clone)]
struct Tris3D {
    v_3d: Triangle<f32>,
    v_uv: Triangle<f32>,
    min: Point<f32>,
    max: Point<f32>,
    node_index: usize,
    id: usize,
}
impl Bounded for Tris3D {
    fn aabb(&self) -> AABB {
        AABB::with_bounds(point_to_point3(self.min), point_to_point3(self.max))
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
impl PartialEq for Tris3D {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[derive(Debug)]
struct UDIMs(HashMap<u32, Vec<Tris3D>>);

#[derive(Debug, Deserialize)]
struct Coords {
    x: f32,
    y: f32,
    z: f32,
}
#[derive(Debug, Deserialize)]
struct VecCameraJSON {
    data: Vec<CameraJSON>,
}
#[derive(Debug, Deserialize)]
struct CameraJSON {
    location: Coords,
    rotation_euler: Coords,
    scale: Coords,
    fov_x: f32,
    limit_near: f32,
    limit_far: f32,
    image_path: String,
}
#[derive(Debug, Clone)]
struct CameraRaw {
    id: usize,
    pos: [f32; 3],
    rot: UnitQuaternion<f32>,
    scale: [f32; 3],
    fov_x: f32,
    limit_near: f32,
    limit_far: f32,
    image_path: String,
}

struct Properties {
    path_data: String,
    path_texture: String,
    img_res_x: u32,
    img_res_y: u32,
    clip_uv: bool,
    blending: Blending,
    backface_culling: bool,
    occlude: bool,
    bleed: u8,
    // upscale: u8,
}

fn load_meshes(path_data: &str) -> (HashMap<u32, Vec<Tris3D>>, Vec<Tris3D>) {
    let data = obj::Obj::load(Path::new(path_data).join("mesh.obj"))
        .unwrap()
        .data;
    let mut udims_tris = HashMap::<u32, Vec<Tris3D>>::new();
    let mut all_tris = Vec::<Tris3D>::new();
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
                let mut vs_pos = Vec::<Point<f32>>::new();
                let mut vs_uv = Vec::<Point<f32>>::new();
                let mut udims = HashSet::<u32>::new();
                for vert in poly.0 {
                    let x = data.position[vert.0][0];
                    let y = data.position[vert.0][1];
                    let z = data.position[vert.0][2];
                    let uv = match vert.1 {
                        Some(i) => match data.texture.get(i) {
                            Some(uv) => uv,
                            _ => continue,
                        },
                        _ => continue,
                    };

                    let u = uv[0];
                    let v = uv[1];
                    vs_pos.push(Point { x, y, z });
                    vs_uv.push(Point { x: u, y: v, z: 0.0 });

                    tr_min_x = tr_min_x.min(x);
                    tr_max_x = tr_max_x.max(x);
                    tr_min_y = tr_min_y.min(y);
                    tr_max_y = tr_max_y.max(y);
                    tr_min_z = tr_min_z.min(z);
                    tr_max_z = tr_max_z.max(z);
                    if u % 1.0 != 0.0 && v % 1.0 != 0.0 {
                        udims.insert(uv_udim(u, v).max(1001));
                    }
                }

                if vs_pos.len() >= 3 {
                    let poly = Tris3D {
                        v_3d: Triangle {
                            a: vs_pos[0],
                            b: vs_pos[1],
                            c: vs_pos[2],
                        },
                        v_uv: Triangle {
                            a: vs_uv[0],
                            b: vs_uv[1],
                            c: vs_uv[2],
                        },
                        min: Point {
                            x: tr_min_x,
                            y: tr_min_y,
                            z: tr_min_z,
                        },
                        max: Point {
                            x: tr_max_x,
                            y: tr_max_y,
                            z: tr_max_z,
                        },
                        node_index: tris_id,
                        id: tris_id,
                    };
                    if poly.v_3d.is_collinear() {
                        continue;
                    }
                    for u_id in udims {
                        if let Entry::Vacant(entry) = udims_tris.entry(u_id) {
                            entry.insert(vec![poly.to_owned()]);
                        } else {
                            udims_tris.get_mut(&u_id).unwrap().push(poly.to_owned());
                        }
                    }
                    all_tris.push(poly);
                    tris_id += 1;
                }
            }
        }
    }
    (udims_tris, all_tris)
}

fn load_cameras(path_data: &str) -> Vec<CameraRaw> {
    let file_json = fs::File::open(Path::new(path_data).join("cameras.json")).unwrap();
    let cameras_json: VecCameraJSON = serde_json::from_reader(file_json).unwrap();
    let mut cameras = Vec::<CameraRaw>::new();
    for (id, cam) in cameras_json.data.into_iter().enumerate() {
        let pos = [cam.location.x, cam.location.y, cam.location.z];
        let rot = UnitQuaternion::from_euler_angles(
            cam.rotation_euler.x,
            cam.rotation_euler.y,
            cam.rotation_euler.z,
        );
        let scale = [cam.scale.x, cam.scale.y, cam.scale.z];
        let fov_x = cam.fov_x;
        let limit_near = cam.limit_near;
        let limit_far = cam.limit_far;
        let image_path = cam.image_path;

        cameras.push(CameraRaw {
            id,
            pos,
            rot,
            scale,
            fov_x,
            limit_near,
            limit_far,
            image_path,
        });
    }

    cameras
}

fn cast_pixels_rays(
    all_tris: &[Tris3D],
    camera_raw: CameraRaw,
    faces: &[Tris3D],
    bvh: &BVH,
    texture: &mut RgbaImage,
    properties: &Properties,
) {
    let img = image::open(camera_raw.image_path).unwrap();
    let width = img.dimensions().0 as usize;
    let height = img.dimensions().1 as usize;
    // if properties.upscale > 0 {
    //     width *= 2_usize.pow(properties.upscale as u32);
    //     height *= 2_usize.pow(properties.upscale as u32);

    //     img.resize(width as u32, height as u32, FilterType::CatmullRom);
    // }

    let ratio = width as f32 / height as f32;
    let fov_y = 2.0 * ((camera_raw.fov_x / 2.0).tan() / ratio).atan();
    let limit_near = camera_raw.limit_near;
    let limit_far = camera_raw.limit_far;
    let [cam_x, cam_y, cam_z] = camera_raw.pos;
    let rot = camera_raw.rot;
    let [sc_x, sc_y, _] = camera_raw.scale;
    let cam_tr = Translation3::new(cam_x, cam_y, cam_z);
    let iso = Isometry3::from_parts(cam_tr, rot);
    let projection = match camera_raw.fov_x > 0.0 {
        true => Projection::Persp(Perspective3::new(ratio, fov_y, limit_near, limit_far)),
        false => Projection::Ortho(Orthographic3::new(
            -sc_x / 2.0,
            sc_x / 2.0,
            -sc_y / 2.0,
            sc_y / 2.0,
            limit_near,
            limit_far,
        )),
    };

    let faces_visible = match properties.backface_culling {
        true => faces
            .iter()
            .filter(|f| !backface(f, &iso, &projection))
            .collect::<Vec<&Tris3D>>(),
        false => faces.iter().collect::<Vec<&Tris3D>>(),
    };
    for face in faces_visible {
        face_img_to_uv(
            all_tris,
            bvh,
            face,
            &iso,
            &projection,
            &img,
            texture,
            properties,
        );
    }
}

fn is_face_closest(
    face: &Tris3D,
    faces_visible: Vec<&Tris3D>,
    ray: Ray,
    near: f32,
    far: f32,
) -> bool {
    if faces_visible.is_empty() {
        return false;
    }
    let ray_orig = point3_to_point(ray.origin);
    let ray_dir = Point {
        x: ray.direction.x,
        y: ray.direction.y,
        z: ray.direction.z,
    };
    let mut closest: Vec<(Option<f32>, &Tris3D)> = faces_visible
        .into_iter()
        .map(|f| (f.v_3d.ray_intersection(&ray_orig, &ray_dir), f))
        .filter(|h| h.0.is_some() && !h.0.unwrap().is_nan())
        .collect();
    if closest.is_empty() {
        return false;
    }
    if closest.len() > 1 {
        closest.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap()
            //.expect(&format!("{:?} {:?}", &a.0, &b.0))
        });
    }
    let dist_to_first = closest[0].0.unwrap();
    if dist_to_first >= near && dist_to_first <= far {
        return closest[0].1 == face;
    }
    false
    // let dist_to_first = closest[0].0.unwrap();
    // let epsilon = (dist_to_first - near) / ((far - near).max(f32::MIN));
    // let closest_faces = closest
    //     .iter()
    //     .filter(|f| {
    //         ((f.0.unwrap() - dist_to_first).abs() <= epsilon)
    //             && f.0.unwrap() >= near
    //             && f.0.unwrap() <= far
    //     })
    //     .map(|f| f.1)
    //     .collect::<Vec<&Tris3D>>();
    // closest_faces.contains(&face)
}

fn _mix_colors(source: Rgba<u8>, target: &Rgba<u8>) -> Rgba<u8> {
    let sr = source[0];
    let sg = source[1];
    let sb = source[2];
    let tr = target[0];
    let tg = target[1];
    let tb = target[2];
    let ta = target[3];

    if ta == 0 {
        source
    } else {
        Rgba([sr / 2 + tr / 2, sg / 2 + tg / 2, sb / 2 + tb / 2, 255])
    }
}

fn repeat_bounds(x: isize, dim: f32) -> u32 {
    let rep = x as f32 % dim;
    if rep < 0.0 {
        (dim + rep) as u32
    } else {
        rep as u32
    }
}

fn _conform_uv(w: f32) -> f32 {
    match w {
        w if w > 1.0 => w % 1.0,
        _ => w,
    }
}
fn backface(face: &Tris3D, iso: &Isometry3<f32>, projection: &Projection) -> bool {
    let normal = face.v_3d.normal();
    match normal {
        None => true,
        Some(n) => {
            let ray_origin_pt =
                Point3::new(iso.translation.x, iso.translation.y, iso.translation.z);
            let ray_target_pt = match projection {
                Projection::Persp(pr) => {
                    iso.transform_point(&pr.unproject_point(&Point3::new(0.0, 0.0, 1.0)))
                }
                Projection::Ortho(pr) => {
                    iso.transform_point(&pr.unproject_point(&Point3::new(0.0, 0.0, 1.0)))
                }
            };
            let cam_vec = Vector3::new(
                ray_target_pt.x - ray_origin_pt.x,
                ray_target_pt.y - ray_origin_pt.y,
                ray_target_pt.z - ray_origin_pt.z,
            );
            let normal_vec = Vector3::new(n.x, n.y, n.z);
            let dot = cam_vec.dot(&normal_vec);
            if dot > 0.0 {
                return true;
            }
            false
        }
    }
}

fn face_img_to_uv(
    all_tris: &[Tris3D],
    bvh: &BVH,
    face: &Tris3D,
    iso: &Isometry3<f32>,
    projection: &Projection,
    img: &DynamicImage,
    texture: &mut RgbaImage,
    properties: &Properties,
) {
    let clip_uv = properties.clip_uv;
    let uv_width = texture.dimensions().0 as f32;
    let uv_height = texture.dimensions().1 as f32;
    let tris_bounds = face.v_uv.aabb();
    let uv_min_u = (tris_bounds[0].x * uv_width).floor() as isize;
    let uv_min_v = (tris_bounds[0].y * uv_height).floor() as isize;
    let uv_max_u = (tris_bounds[1].x * uv_width).ceil() as isize;
    let uv_max_v = (tris_bounds[1].y * uv_height).ceil() as isize;

    let cam_width = img.dimensions().0 as f32;
    let cam_height = img.dimensions().1 as f32;

    let face_cam = Triangle {
        a: project_point_to_cam(face.v_3d.a, iso, projection),
        b: project_point_to_cam(face.v_3d.b, iso, projection),
        c: project_point_to_cam(face.v_3d.c, iso, projection),
    };
    if face_cam.is_collinear() {
        return;
    }
    for v in uv_min_v..=uv_max_v {
        for u in uv_min_u..=uv_max_u {
            let uv_u = match clip_uv {
                true => u as u32,
                false => repeat_bounds(u, uv_width),
            };
            let uv_v = match clip_uv {
                true => v as u32,
                false => repeat_bounds(v, uv_height),
            };
            let ray_disp = [[0.0, 0.0], [0.99, 0.99], [0.0, 0.99], [0.99, 0.0]];
            let mut colors_to_mix = Vec::<Color>::new();
            for d in ray_disp.iter() {
                let p_uv = Point {
                    x: (u as f32 + d[0]) / uv_width,
                    y: (v as f32 + d[1]) / uv_height,
                    z: 0.0,
                };
                if face.v_uv.has_point(p_uv) {
                    let p_bary = face.v_uv.cartesian_to_barycentric(&p_uv);
                    let p_cam = face_cam.barycentric_to_cartesian(&p_bary);

                    if face_cam.has_point(p_cam)
                        && p_cam.x >= -1.0
                        && p_cam.y >= -1.0
                        && p_cam.x <= 1.0
                        && p_cam.y <= 1.0
                    {
                        let cam_x = (cam_width * (p_cam.x + 1.0) / 2.0) as u32;
                        let cam_y = (cam_height * (p_cam.y + 1.0) / 2.0) as u32;
                        if cam_x < cam_width as u32
                            && cam_y < cam_height as u32
                            && uv_u < uv_width as u32
                            && uv_v < uv_height as u32
                        {
                            let face_is_visible = match properties.occlude {
                                true => {
                                    let ray_origin_pt = match projection {
                                        Projection::Persp(_) => Point3::new(
                                            iso.translation.x,
                                            iso.translation.y,
                                            iso.translation.z,
                                        ),
                                        Projection::Ortho(pr) => {
                                            iso.transform_point(&pr.unproject_point(&Point3::new(
                                                p_cam.x, p_cam.y, -1.0,
                                            )))
                                        }
                                    };
                                    let ray_target_pt = match projection {
                                        Projection::Persp(pr) => {
                                            iso.transform_point(&pr.unproject_point(&Point3::new(
                                                p_cam.x, p_cam.y, 1.0,
                                            )))
                                        }
                                        Projection::Ortho(pr) => {
                                            iso.transform_point(&pr.unproject_point(&Point3::new(
                                                p_cam.x, p_cam.y, 1.0,
                                            )))
                                        }
                                    };

                                    let ray = Ray::new(
                                        ray_origin_pt,
                                        Vector3::new(
                                            ray_target_pt.x - ray_origin_pt.x,
                                            ray_target_pt.y - ray_origin_pt.y,
                                            ray_target_pt.z - ray_origin_pt.z,
                                        ),
                                    );

                                    let [znear, zfar] = match projection {
                                        Projection::Persp(pr) => [pr.znear(), pr.zfar()],
                                        Projection::Ortho(pr) => [pr.znear(), pr.zfar()],
                                    };

                                    is_face_closest(
                                        face,
                                        bvh.traverse(&ray, all_tris),
                                        ray,
                                        znear,
                                        zfar,
                                    )
                                }
                                false => true,
                            };

                            if face_is_visible {
                                let source_color =
                                    img.get_pixel(cam_x, cam_height as u32 - cam_y - 1);
                                colors_to_mix.push(source_color);
                            }
                        }
                    }
                }
            }
            if !colors_to_mix.is_empty() {
                texture.put_pixel(uv_u, uv_height as u32 - uv_v - 1, average(colors_to_mix));
            }
        }
    }
}

fn blend_pixel_with_neigbhours(texture: &RgbaImage, x: u32, y: u32, limit: u8) -> Rgba<u8> {
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
    let mut r = 0;
    let mut g = 0;
    let mut b = 0;
    let mut a = 0;
    for way in ways.iter() {
        let xp = x as i32 + way[0];
        let yp = y as i32 + way[1];
        if xp >= 0 && xp < bx && yp >= 0 && yp < by {
            let col = texture.get_pixel(xp as u32, yp as u32);
            if col[3] != 0 {
                neibs_count += 1;
                r += col[0] as u32;
                g += col[1] as u32;
                b += col[2] as u32;
                a += col[3] as u32;
            }
        }
    }
    if neibs_count > limit as u32 {
        r /= neibs_count;
        g /= neibs_count;
        b /= neibs_count;
        a /= neibs_count;
        Rgba([r as u8, g as u8, b as u8, a as u8])
    } else {
        return *texture.get_pixel(x, y);
    }
}

type Color = Rgba<u8>;

enum Blending {
    Average,
    Median,
    Mode,
    Overlay,
}

fn average(colors: Vec<Color>) -> Color {
    let mut sum_r: usize = 0;
    let mut sum_g: usize = 0;
    let mut sum_b: usize = 0;
    let mut sum_a: usize = 0;
    colors.iter().for_each(|c| {
        sum_r += c[0] as usize;
        sum_g += c[1] as usize;
        sum_b += c[2] as usize;
        sum_a += c[3] as usize;
    });
    let r = (sum_r / colors.len()) as u8;
    let g = (sum_g / colors.len()) as u8;
    let b = (sum_b / colors.len()) as u8;
    let a = (sum_a / colors.len()) as u8;
    Rgba([r, g, b, a])
}

fn median(colors: &mut Vec<Color>) -> Color {
    colors.sort_by_key(col_len);
    colors[colors.len() / 2]
}

fn mode(colors: Vec<Color>) -> Color {
    let mut vec_mode = Vec::new();
    let mut seen_map = HashMap::new();
    let mut max_val = 0;
    for c in colors {
        let ctr = seen_map.entry(c).or_insert(0);
        *ctr += 1;
        if *ctr > max_val {
            max_val = *ctr;
        }
    }
    for (key, val) in seen_map {
        if val == max_val {
            vec_mode.push(key);
        }
    }
    vec_mode[0]
}

fn overlay(colors: Vec<Color>) -> Color {
    let mut bg = Rgba([0, 0, 0, 0]);
    for fg in colors {
        let bga = bg[3] as f32;
        let bgr = bg[0] as f32;
        let bgg = bg[1] as f32;
        let bgb = bg[2] as f32;

        let fga = fg[3] as f32;
        let fgr = fg[0] as f32;
        let fgg = fg[1] as f32;
        let fgb = fg[2] as f32;

        let d = (255.0 - fga) * bga;
        let a = d + fga;
        let r = (d * bgr + fgr * fga) / a;
        let g = (d * bgg + fgg * fga) / a;
        let b = (d * bgb + fgb * fga) / a;

        bg = Rgba([r as u8, g as u8, b as u8, a as u8]);
    }
    bg
}

fn combine_layers(textures: Vec<(usize, RgbaImage)>, blending: &Blending) -> (RgbaImage, bool) {
    let (img_res_x, img_res_y) = textures[0].1.dimensions();
    let mut mono_texture = RgbaImage::new(img_res_x, img_res_y);
    let mut texture_is_empty = true;
    for y in 0..img_res_y {
        for x in 0..img_res_x {
            let mut colors = Vec::<Color>::new();
            for (_, part) in &textures {
                let col = part.get_pixel(x, y);
                if col[3] != 0 {
                    colors.push(*col);
                }
            }
            if !colors.is_empty() {
                texture_is_empty = false;
                let m = match &blending {
                    Blending::Average => average(colors),
                    Blending::Median => median(&mut colors),
                    Blending::Mode => mode(colors),
                    Blending::Overlay => overlay(colors),
                };
                mono_texture.put_pixel(x, y, m)
            }
        }
    }
    (mono_texture, texture_is_empty)
}

fn expand_pixels(texture: &mut RgbaImage, limit: u8) {
    let (width, height) = texture.dimensions();
    let mut future_pixels = Vec::<(u32, u32, Rgba<u8>)>::new();
    for v in 0..(height as usize) {
        for u in 0..(width as usize) {
            let current_color = *texture.get_pixel(u as u32, v as u32);
            if current_color[3] == 0 {
                let blended_color = blend_pixel_with_neigbhours(texture, u as u32, v as u32, limit);
                if blended_color[3] != 0 {
                    future_pixels.push((u as u32, v as u32, blended_color));
                }
            }
        }
    }
    for p in future_pixels {
        texture.put_pixel(p.0, p.1, p.2);
    }
}

fn col_len(c: &Color) -> usize {
    (((c[0] as usize).pow(2)
        + (c[1] as usize).pow(2)
        + (c[2] as usize).pow(2)
        + (c[3] as usize).pow(2)) as f32)
        .sqrt() as usize
}

fn parse_arguments(args: Vec<String>) -> Option<Properties> {
    if args.len() < 10 {
        println!("Arguments are insufficient.");
        return None;
    }

    let properties = Properties {
        path_data: args[1].to_string(),
        path_texture: args[2].to_string(),
        img_res_x: args[3].parse::<u32>().unwrap(),
        img_res_y: args[4].parse::<u32>().unwrap(),
        clip_uv: matches![args[5].parse::<u8>(), Ok(1)],
        blending: match args[6].parse::<u8>() {
            Ok(0) => Blending::Average,
            Ok(1) => Blending::Median,
            Ok(2) => Blending::Mode,
            Ok(3) => Blending::Overlay,
            _ => Blending::Overlay,
        },

        backface_culling: match args[7].parse::<u8>() {
            Ok(0) => false,
            Ok(1) => true,
            _ => false,
        },

        occlude: match args[8].parse::<u8>() {
            Ok(0) => false,
            Ok(1) => true,
            _ => false,
        },
        bleed: match args[9].parse::<u8>() {
            Ok(n) => n,
            _ => 0,
        },
        // upscale: match args[9].parse::<u8>() {
        //     Ok(n) => n,
        //     _ => 0,
        // },
    };

    Some(properties)
}

fn main() {
    //CLI
    println!("\nEyek welcomes you!");
    let args: Vec<_> = env::args().collect();
    let properties = match parse_arguments(args) {
        Some(props) => props,
        None => {
            println!("Can't parse arguments.\nEyek out.");
            return;
        }
    };

    //Loading
    let (udims_tris, mut all_tris) = load_meshes(&properties.path_data);
    let udims_num = udims_tris.len();
    println!("OBJ loaded.");
    println!("UDIMs: {}", udims_num);
    let cameras = load_cameras(&properties.path_data);
    let cam_num = cameras.len();
    let cameras_loaded = match cam_num {
        1 => "Camera loaded.".to_string(),
        _ => format!("{:?} cameras loaded.", cam_num),
    };
    println!("{}", cameras_loaded);
    println!("Puny humans are instructed to wait.");
    let bvh = BVH::build(&mut all_tris);
    for (id, faces) in udims_tris {
        println!("Started UDIM: {}", id);
        //Parallel execution
        let mut textures: Vec<(usize, RgbaImage)> = cameras
            .to_owned()
            .into_par_iter()
            .map(|cam| {
                let mut texture = RgbaImage::new(properties.img_res_x, properties.img_res_y);
                let id = cam.id;
                cast_pixels_rays(&all_tris, cam, &faces, &bvh, &mut texture, &properties);
                println!("Finished cam: #{:?} / {:?}", id, cam_num);
                // if properties.bleed == 0 {
                //     expand_pixels(&mut texture, 2);
                // }
                (id, texture)
            })
            .collect();

        //Combining images
        if let Blending::Overlay = &properties.blending {
            textures.sort_by(|a, b| a.0.cmp(&b.0));
        };

        let (mut mono_texture, texture_is_empty) = combine_layers(textures, &properties.blending);

        //Color empty pixels around polygons edges
        for _ in 0..properties.bleed {
            expand_pixels(&mut mono_texture, 0);
        }

        //Export texture
        let file_name = match &udims_num {
            1 => format!("{}.png", &properties.path_texture),
            _ => format!("{}.{}.png", &properties.path_texture, id),
        };
        if !texture_is_empty {
            mono_texture.save(Path::new(&file_name)).unwrap();
            println!("Finished UDIM: {}\n", id);
        }
    }
    fs::remove_dir_all(properties.path_data).unwrap();
    println!("Texture saved!\nEyek out. See you next time.");
}
