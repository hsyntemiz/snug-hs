import os

import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


def load_obj(filename, tex_coords=False):
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()

            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces


def save_obj(filename, vertices, faces):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if tf.is_tensor(vertices):
        vertices = vertices.numpy()

    if tf.is_tensor(faces):
        faces = faces.numpy()

    vertices = vertices.squeeze()
    faces = faces.squeeze()

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print("Saved:", filename)


def _todict(matobj):
    '''
	A recursive function which constructs from matobjects nested dictionaries
	'''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray) and np.any(
                [isinstance(item, sio.matlab.mio5_params.mat_struct) for item in elem]):
            dict[strg] = [None] * len(elem)
            for i, item in enumerate(elem):
                if isinstance(item, sio.matlab.mio5_params.mat_struct):
                    dict[strg][i] = _todict(item)
                else:
                    dict[strg][i] = item
        else:
            dict[strg] = elem
    return dict


def _check_keys(dict):
    '''
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	'''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def loadInfo(filename):
    '''
	this function should be called instead of direct sio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	'''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    del data['__globals__']
    del data['__header__']
    del data['__version__']
    return _check_keys(data)


def load_motion(path):
    motion = np.load(path, mmap_mode='r')

    reduce_factor = int(motion['mocap_framerate'] // 30)
    pose = motion['poses'][::reduce_factor, :72]
    trans = motion['trans'][::reduce_factor, :]

    separate_arms(pose)

    # Swap axes
    swap_rotation = R.from_euler('zx', [-90, 270], degrees=True)
    root_rot = R.from_rotvec(pose[:, :3])
    pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
    trans = swap_rotation.apply(trans)

    # Center model in first frame
    trans = trans - trans[0]

    # Compute velocities
    trans_vel = finite_diff(trans, 1 / 30)

    return pose.astype(np.float32), trans.astype(np.float32), trans_vel.astype(np.float32)


def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
    num_joints = poses.shape[-1] // 3

    poses = poses.reshape((-1, num_joints, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

    poses[:, 23] *= 0.1
    poses[:, 22] *= 0.1

    return poses.reshape((poses.shape[0], -1))


def finite_diff(x, h, diff=1):
    if diff == 0:
        return x

    v = np.zeros(x.shape, dtype=x.dtype)
    v[1:] = (x[1:] - x[0:-1]) / h

    return finite_diff(v, h, diff - 1)


def get_model_path(garment):
    garments = {
        "tshirt": (
            "models/SNUG-Tshirt",
            "assets/meshes/tshirt.obj"
        ),
        "tank": (
            "models/SNUG-Tank",
            "assets/meshes/tank.obj"
        ),
        "top": (
            "models/SNUG-Top",
            "assets/meshes/long_sleeve_top.obj"
        ),
        "pants": (
            "models/SNUG-Pants",
            "assets/meshes/pants.obj"
        ),
        "shorts": (
            "models/SNUG-Shorts",
            "assets/meshes/shorts.obj"
        )
    }

    assert garment in garments, f"'{garment}' is not a valid option. Valid options: {list(garments.keys())}"

    return garments[garment]


def weights_prior(T, B, weights):
    tree = cKDTree(B)
    _, idx = tree.query(T)
    return weights[idx]


def my_weights_prior(T, B, weights):
    t_l = T.shape[0]
    idx = np.zeros(t_l,dtype=int)
    for i, t in enumerate(T):
        dist = tf.math.reduce_euclidean_norm(B-t,axis=-1)
        idx[i]=tf.argmin(dist).numpy().astype(int)
    print(idx)
    return idx, weights[idx]