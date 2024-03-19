

try:
    import pyrender
    import pytorch3d
    from pytorch3d.renderer import (
        look_at_view_transform,
    )
except:
    pass
import trimesh, math, glob, os
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
import numpy as np
import os, cv2, json, shutil, time, scipy
    
def render_mesh_pyrender(mesh, dist, elev, azim, image_width=256, image_height=256, is_render = True):

    def _render_mesh(mesh, cam_pose, K, image_width, image_height):

        [fx, fy, cx, cy] = K

        scene = pyrender.Scene.from_trimesh_scene(mesh, ambient_light=(1, 1, 1))
        camera = pyrender.camera.IntrinsicsCamera(fx,fy,cx,cy)
        scene.add(camera, pose = cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene.add(light, pose = cam_pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)
        color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.VERTEX_NORMALS)

        return color, depth

    
    T = np.array([[(dist * math.cos(math.radians(elev))) * math.cos(math.radians(azim)), \
                        dist * math.sin(math.radians(elev)),
                        (dist * math.cos(math.radians(elev))) * math.sin(math.radians(azim)), \
                        ]])

    R = pytorch3d.renderer.cameras.look_at_rotation([mesh.centroid], at = T)
    RT = np.eye(4,4)
    RT[:3,:3] = R
    RT[:3,3] = T

    fx = 355  # Focal length along the x-axis
    fy = 355  # Focal length along the y-axis
    cx = image_width / 2  # Principal point's x-coordinate
    cy = image_height / 2  # Principal point's y-coordinate

    K = np.array([[fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]] )

    if is_render: im, depth = _render_mesh(mesh, RT, [fx, fy, cx, cy],  image_width, image_height) 
    else: im, depth = None, None

    R, T = RT[:3,:3], RT[:3,3:]
    RT_ = RT.copy()
    T1 = np.concatenate([np.eye(3,3), -T], -1)
    RT = np.eye(4,4)
    RT[:3,:4] = R.T@T1


    return im, depth, RT, RT_, K


def main(obj_path, num_views, out_path, dist = 3, elev = 20, image_width = 256, image_height = 256, gray_scale = True, obj_id = None):

    """
    Description:
    This function generates multiple views of a 3D object and saves them as images. It uses the provided obj_path to load the 3D object, generates the specified number of views around the object, and saves the rendered images to the specified out_path.

    Parameters:
    - obj_path (str): The path to the 3D object file.
    - num_views (int): The number of views to generate around the object.
    - out_path (str): The directory path where the rendered images will be saved.
    - dist (float, optional): The distance from the object while capturing the views. Default is 3 units.
    - elev (float, optional): The elevation angle (in degrees) for the camera. Default is 20 degrees.
    - image_width (int, optional): The width of the output images. Default is 256 pixels.
    - image_height (int, optional): The height of the output images. Default is 256 pixels.
    - gray_scale (bool, optional): Whether to render images in grayscale or color. Default is True (grayscale).
    - obj_id (str or None, optional): The identifier for the object. Default is None.

    Returns:
    None

    Usage:
    # Example 1: Generate 10 views of a 3D object in grayscale
    main(obj_path="path/to/object.obj", num_views=10, out_path="output/directory", gray_scale=True)

    """

    out_dir = out_path  
    os.makedirs(out_dir, exist_ok = True)

    mesh_s = trimesh.load(obj_path, force="scene")
    mesh = trimesh.load(obj_path, force="mesh")
    
    if gray_scale:
        for msh in mesh_s.geometry.values():
            if hasattr(msh, 'visual') and hasattr(msh.visual, 'material'):
                msh.visual = trimesh.visual.color.ColorVisuals()

    points_3d = trimesh.sample.sample_surface(mesh, 2000)[0]
    ray_tracer = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    data = []

    for azim in range(0,360+1,360//num_views):

        image, depth, RT, RT_, K = render_mesh_pyrender(mesh=mesh_s, dist=dist, elev=elev, azim=azim, image_width = image_width, image_height = image_height, is_render = True) 
        ray_origins = np.linalg.inv(RT) @ np.array([[0, 0, 0, 1]]).T
        intersections,_,_ = ray_tracer.intersects_location(
                ray_origins=ray_origins[:3, :].T-points_3d+points_3d,
                ray_directions=points_3d-ray_origins[:3, :].T,
                multiple_hits = False
            )
        is_visible = np.sqrt(np.sum((points_3d[:, np.newaxis] - intersections) ** 2, axis=-1)).min(1)<1e-8
        x,y,z = points_3d[:,0], points_3d[:,1], points_3d[:,2]
        P = K@RT[:3]
        xy = np.stack([x,y,z, np.ones_like(z)],-1)@P.T
        xy = xy/xy[:,2:]
        xy = np.array([(int(i[1].round()), 256 - int(i[0].round())) for i in xy])

        output_path = os.path.join(out_dir, f'{obj_id}_{dist}_{azim}_{elev}.png')

        proj_matrix = {'RT':np.array(RT).tolist(), 'RT_':np.array(RT_).tolist(),'K':np.array(K).tolist()}

        with open(output_path.replace('.png', '.json'), 'w+') as f:
            json.dump(proj_matrix, f, sort_keys=True)

        cv2.imwrite(output_path, image)

        np.save(output_path.replace('.png', '.npy'), {'pxy': xy, 'is_visible':is_visible})
        np.save(output_path.replace('.png', '_depth.npy'), depth)

    create_pos3d_encoding(obj_path, out_path)


def create_pos3d_encoding(obj_path, out_path):

    stride = 8
    out_dim = 256//stride
    paths = glob.glob(out_path+'/*.png')

    mesh = trimesh.load(obj_path, force = 'mesh')
    ray_tracer = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    pnts3d = trimesh.sample.sample_surface(mesh, 10000)[0]
    x,y,z = pnts3d[:,0], pnts3d[:,1], pnts3d[:,2]

    for pi in paths:
        KRT = json.load(open(pi.replace('.png', '.json'), 'r'))
        start_time = time.time()
        P = np.array(KRT['K'])@np.array(KRT['RT'])[:3]

        P[:2] /= stride
        xy = np.stack([x,y,z, np.ones_like(z)],-1)@P.T
        xy = xy/xy[:,2:]
        xy = np.array([(int(i[1].round()), out_dim - int(i[0].round())) for i in xy])
        ray_origins = np.linalg.inv(np.array(KRT['RT'])) @ np.array([[0, 0, 0, 1]]).T
        intersections,_,_ = ray_tracer.intersects_location(
                ray_origins=ray_origins[:3, :].T-pnts3d+pnts3d,
                ray_directions=pnts3d-ray_origins[:3, :].T,
                multiple_hits = False
            )
        is_visible = scipy.spatial.distance.cdist(pnts3d,intersections, 'euclidean').min(1)<1e-8
        ps = -1*np.ones((out_dim,out_dim,3))
        for i, j in zip(xy[is_visible], pnts3d[is_visible]):
            if i[0] < out_dim and i[1] < out_dim:
                ps[i[0],i[1]] = j

        my_dict = np.load(pi.replace('.png', '.npy'), allow_pickle = True).item()
        my_dict['xyz'] = ps
        np.save(pi.replace('.png', '.npy'), my_dict)
        

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--obj_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--dist_from_object', type=float, default=2)
    parser.add_argument('--elev_angle', type=int, default=20)
    parser.add_argument('--image_height', type=int, default=256)
    parser.add_argument('--image_width', type=int, default=256)
    parser.add_argument('--gray_scale', action='store_true')
    parser.add_argument('--num_imgs', type=int, default=20)
    parser.add_argument('--obj_id', type=str, default='obj')


    args = parser.parse_args()
    main(args.obj_path,  args.num_imgs, args.out_path, \
        dist = args.dist_from_object, elev = args.elev_angle, \
            image_width = args.image_height, image_height = args.image_width, gray_scale = args.gray_scale, obj_id = args.obj_id)
