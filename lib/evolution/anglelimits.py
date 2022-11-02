import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================#
# Load the joint angle constraints
# These files are directly converted from .mat to .npy
# The MATLAB implementation of the CVPR 15 paper has detailed documentation.

root = "resources/constraints"
# logging.info("Loading files from " + root)
model_path = os.path.join(root, "jointAngleModel_v2.npy")
joint_angle_limits = np.load(model_path, allow_pickle=True).item()
angle_spread = joint_angle_limits['angleSprd']
# separation plane for conditional joint angle
sepPlane = joint_angle_limits['sepPlane']
E2 = joint_angle_limits['E2']
bounds = joint_angle_limits['bounds']
# static pose and parameters used in coordinate transformation
static_pose_path = os.path.join(root, "staticPose.npy")
static_pose = np.load(static_pose_path, allow_pickle=True).item()
di = static_pose['di']
a = static_pose['a'].reshape(3)
# # load the pre-computed conditinal distribution
# con_dis_path = os.path.join(root, "conditional_dis.npy")
# con_dis = np.load(con_dis_path, allow_pickle=True).item()
# =============================================================================#
# joint names of the CVPR 15 paper
PRIOR_NAMES = ['back-bone',
               'R-shldr',
               'R-Uarm',
               'R-Larm',
               'L-shldr',
               'L-Uarm',
               'L-Larm',
               'head',
               'R-hip',
               'R-Uleg',
               'R-Lleg',
               'R-feet',
               'L-hip',
               'L-Uleg',
               'L-Lleg',
               'L-feet'
               ]
# Human 3.6M joint names are slightly different from the above
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'
# correspondence of the joints
# (key, value) -> (index in prior_names, index in H36M names)
correspondence = {0: 12, 1: 13, 2: 25, 3: 26, 4: 27, 5: 17, 6: 18, 7: 19, 8: 15, 9: 1,
                  10: 2, 11: 3, 13: 6, 14: 7, 15: 8}
# number of bone vectors attached to a torso
num_of_bones = 9
# descretization of spherical coordinates
# bin edges for theta
theta_edges = np.arange(0.5, 122, 1)  # theta values: 1 to 121 (integer)
# bin edges for phi
phi_edges = np.arange(0.5, 62, 1)  # phi values: 1 to 61
# color map used for visualization
cmap = plt.cm.RdYlBu
# indices used for computing bone vectors for non-torso bones
nt_parent_indices = [8, 11, 12, 14, 15, 4, 5, 1, 2]
nt_child_indices = [10, 12, 13, 15, 16, 5, 6, 2, 3]
# map from bone index to the parent's di index
di_indices = {2: 5, 4: 2, 6: 13, 8: 9}
# map from angle index to record index
record_indices = {0: 4, 1: 2, 3: 0, 5: 8, 7: 5, 2: 3, 4: 1, 6: 9, 8: 6}
# name for the bone vectors
bone_name = {
    1: 'thorax to head top',
    2: 'left shoulder to left elbow',
    3: 'left elbow to left wrist',
    4: 'right shoulder to right elbow',
    5: 'right elbow to right wrist',
    6: 'left hip to left knee',
    7: 'left knee to left ankle',
    8: 'right hip to right knee',
    9: 'right knee to right ankle'
}


# =============================================================================#
def is_valid_local(skeleton_local, return_ang=False):
    """
    Check if the limbs represented in local coordinate system are valid or not.
    """
    valid_vec = np.ones((num_of_bones), dtype=np.bool)
    angles = to_spherical(skeleton_local)
    angles[:, 1:] *= 180 / np.pi
    # convert to valid range and discretize
    # theta: -180~180 degrees discretized into 120 bins
    # phi: -90~90 degrees discretized into 60 bins
    angles[:, 1] = np.floor((angles[:, 1] + 180) / 3 + 1)
    angles[:, 2] = np.floor((angles[:, 2] + 90) / 3 + 1)
    # go through each bone and check the angle-limits
    for angle_idx in range(len(angles)):
        angle = angles[angle_idx]
        record_idx = record_indices[angle_idx]
        theta, phi = int(angle[1]), int(angle[2])
        if angle_idx in [0, 1, 3, 5, 7]:
            test_value = angle_spread[0, record_idx][theta - 1, phi - 1]
            if test_value == 0:
                valid_vec[angle_idx] = False
        else:
            angle_parent = angles[angle_idx - 1]
            theta_p, phi_p = int(angle_parent[1]), int(angle_parent[2])
            vector = normalize(sepPlane[0, record_idx][theta_p - 1, phi_p - 1])
            for value in vector:
                if np.isnan(value):
                    valid_vec[angle_idx] = False
                    continue
            if np.dot(np.hstack([skeleton_local[angle_idx], 1]), vector) > 0:
                valid_vec[angle_idx] = False
            else:
                e1 = vector[:-1]
                e2 = E2[0, record_idx][theta_p - 1, phi_p - 1]
                T = gram_schmidt_columns(np.hstack([e1.reshape(3, 1),
                                                    e2.reshape(3, 1),
                                                    np.cross(e1, e2).reshape(3, 1)]))
                bnd = bounds[0, record_idx][theta_p - 1, phi_p - 1]
                u = (T[:, 1:]).T @ skeleton_local[angle_idx]
                if u[0] < bnd[0] or u[0] > bnd[1] or u[1] < bnd[2] or u[1] > bnd[3]:
                    valid_vec[angle_idx] = False
    if return_ang:
        return valid_vec, angles
    else:
        return valid_vec


def is_valid(skeleton, return_ang=False, camera=False):
    """
    args:
        skeleton: input skeleton of shape [num_joints, 3] use the annotation
        of Human 3.6M dataset
    return:
        valid_vec: boolean vector specifying the validity for each bone.
        return 0 for invalid bones.
        camera: relative orientation of camera and human
    """

    skeleton = skeleton.reshape(17, -1)
    # the ordering of coordinate used by the Prior was x,z and y
    skeleton = skeleton[:, [0, 2, 1]]
    # convert bone vectors into local coordinate
    skeleton_local = to_local(skeleton)
    ret = is_valid_local(skeleton_local, return_ang=return_ang)
    if return_ang:
        return ret[0], ret[1]
    else:
        return ret


def normalize(vector):
    """
    Normalize a vector.
    """
    return vector / np.linalg.norm(vector)


def to_spherical(xyz):
    """
    Convert from Cartisian coordinate to spherical coordinate
    theta: [-pi, pi]
    phi: [-pi/2, pi/2]
    note that xyz should be float number
    """
    # return in r, phi, and theta (elevation angle from z axis down)
    return_value = np.zeros(xyz.shape, dtype=xyz.dtype)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    return_value[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)  # r
    return_value[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])  # theta
    return_value[:, 2] = np.arctan2(xyz[:, 2], np.sqrt(xy))  # phi
    return return_value


def to_xyz(rthetaphi):
    """
    Convert from spherical coordinate to Cartisian coordinate
    theta: [0, 2*pi] or [-pi, pi]
    phi: [-pi/2, pi/2]
    """
    return_value = np.zeros(rthetaphi.shape, dtype=rthetaphi.dtype)
    sintheta = np.sin(rthetaphi[:, 1])
    costheta = np.cos(rthetaphi[:, 1])
    sinphi = np.sin(rthetaphi[:, 2])
    cosphi = np.cos(rthetaphi[:, 2])
    return_value[:, 0] = rthetaphi[:, 0] * costheta * cosphi  # x
    return_value[:, 1] = rthetaphi[:, 0] * sintheta * cosphi  # y
    return_value[:, 2] = rthetaphi[:, 0] * sinphi  # z
    return return_value


def test_coordinate_conversion():
    # theta: [-pi, pi] reference
    xyz = np.random.rand(1, 3) * 2 - 1
    rthetaphi = to_spherical(xyz)
    xyz2 = to_xyz(rthetaphi)
    print('maximum error:', np.max(np.abs(xyz - xyz2)))
    # theta: [0, 2*pi] reference
    xyz = np.random.rand(1, 3) * 2 - 1
    rthetaphi = to_spherical(xyz)
    indices = rthetaphi[:, 1] < 0
    rthetaphi[:, 1][indices] += 2 * np.pi
    xyz2 = to_xyz(rthetaphi)
    print('maximum error:', np.max(np.abs(xyz - xyz2)))
    return


def gram_schmidt_columns(X):
    """
    Apply Gram-Schmidt orthogonalization to obtain basis vectors.
    """
    B = np.zeros(X.shape)
    B[:, 0] = (1 / np.linalg.norm(X[:, 0])) * X[:, 0]
    for i in range(1, 3):
        v = X[:, i]
        U = B[:, 0:i]  # subspace basis which has already been orthonormalized
        pc = U.T @ v  # orthogonal projection coefficients of v onto U
        p = U @ pc
        v = v - p
        if np.linalg.norm(v) < 2e-16:
            # vectors are not linearly independent!
            raise ValueError
        else:
            v = normalize(v)
            B[:, i] = v
    return B


def direction_check(system, v1, v2, v3):
    if system[:, 0].dot(v1) < 0:
        system[:, 0] *= -1
    if system[:, 1].dot(v2) < 0:
        system[:, 1] *= -1
    if system[:, 2].dot(v3) < 0:
        system[:, 2] *= -1
    return system


def get_normal(x1, a, x):
    """
    Get normal vector.
    """
    nth = 1e-4
    # x and a are parallel
    if np.linalg.norm(x - a) < nth or np.linalg.norm(x + a) < nth:
        n = np.cross(x, x1)
        flag = True
    else:
        n = np.cross(a, x)
        flag = False
    return normalize(n), flag


def get_basis1(skeleton):
    """
    Compute local coordinate system from 3D joint positions.
    This system is used for upper-limbs.
    """
    # compute the vector from the left shoulder to the right shoulder
    left_shoulder = skeleton[11]
    right_shoulder = skeleton[14]
    v1 = normalize(right_shoulder - left_shoulder)
    # compute the backbone vector from the thorax to the spine
    thorax = skeleton[8]
    spine = skeleton[7]
    v2 = normalize(spine - thorax)
    # v3 is the cross product of v1 and v2 (front-facing vector for upper-body)
    v3 = normalize(np.cross(v1, v2))
    return v1, v2, v3


def to_local(skeleton):
    """
    Represent the bone vectors in the local coordinate systems.
    """
    v1, v2, v3 = get_basis1(skeleton)
    # compute the vector from the left hip to the right hip
    left_hip = skeleton[4]
    right_hip = skeleton[1]
    v4 = normalize(right_hip - left_hip)
    # v5 is the cross product of v4 and v2 (front-facing vector for lower-body)
    v5 = normalize(np.cross(v4, v2))
    # compute orthogonal coordinate systems using GramSchmidt
    # for upper body, we use v1, v2 and v3
    system1 = gram_schmidt_columns(np.hstack([v1.reshape(3, 1),
                                              v2.reshape(3, 1),
                                              v3.reshape(3, 1)]))
    # make sure the directions rougly align
    # system1 = direction_check(system1, v1, v2, v3)
    # for lower body, we use v4, v2 and v5
    system2 = gram_schmidt_columns(np.hstack([v4.reshape(3, 1),
                                              v2.reshape(3, 1),
                                              v5.reshape(3, 1)]))
    # system2 = direction_check(system2, v4, v2, v5)

    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]
    # convert bone vector to local coordinate system
    bones_local = np.zeros(bones.shape, dtype=bones.dtype)
    for bone_idx in range(len(bones)):
        # only compute bone vectors for non-torsos
        # the order of the non-torso bone vector is:
        # bone vector1: thorax to head top
        # bone vector2: left shoulder to left elbow
        # bone vector3: left elbow to left wrist
        # bone vector4: right shoulder to right elbow
        # bone vector5: right elbow to right wrist
        # bone vector6: left hip to left knee
        # bone vector7: left knee to left ankle
        # bone vector8: right hip to right knee
        # bone vector9: right knee to right ankle
        bone = normalize(bones[bone_idx])
        if bone_idx in [0, 1, 3, 5, 7]:
            # bones that are directly connected to the torso
            if bone_idx in [0, 1, 3]:
                # upper body
                bones_local[bone_idx] = system1.T @ bone
            else:
                # lower body
                bones_local[bone_idx] = system2.T @ bone
        else:
            if bone_idx in [2, 4]:
                parent_R = system1
            else:
                parent_R = system2
            # parent bone index is smaller than 1
            vector_u = normalize(bones[bone_idx - 1])
            di_index = di_indices[bone_idx]
            vector_v, flag = get_normal(parent_R @ di[:, di_index],
                                        parent_R @ a,
                                        vector_u
                                        )
            vector_w = np.cross(vector_u, vector_v)
            local_system = gram_schmidt_columns(np.hstack([vector_u.reshape(3, 1),
                                                           vector_v.reshape(3, 1),
                                                           vector_w.reshape(3, 1)]
                                                          )
                                                )
            bones_local[bone_idx] = local_system.T @ bone
    return bones_local


def to_global(skeleton, bones_local, cache=False):
    """
    Convert local coordinate back into global coordinate system.
    cache: return intermeadiate results
    """
    return_value = {}
    v1, v2, v3 = get_basis1(skeleton)
    # compute the vector from the left hip to the right hip
    left_hip = skeleton[6]
    right_hip = skeleton[1]
    v4 = normalize(right_hip - left_hip)
    # v5 is the cross product of v4 and v2 (front-facing vector for lower-body)
    v5 = normalize(np.cross(v4, v2))
    # compute orthogonal coordinate systems using GramSchmidt
    # for upper body, we use v1, v2 and v3
    system1 = gram_schmidt_columns(np.hstack([v1.reshape(3, 1),
                                              v2.reshape(3, 1),
                                              v3.reshape(3, 1)]))
    # make sure the directions rougly align
    # system1 = direction_check(system1, v1, v2, v3)
    # for lower body, we use v4, v2 and v5
    system2 = gram_schmidt_columns(np.hstack([v4.reshape(3, 1),
                                              v2.reshape(3, 1),
                                              v5.reshape(3, 1)]))
    # system2 = direction_check(system2, v4, v2, v5)
    if cache:
        return_value['cache'] = [system1, system2]
        return_value['bl'] = bones_local

    bones_global = np.zeros(bones_local.shape)
    # convert bone vector to local coordinate system
    for bone_idx in [0, 1, 3, 5, 7, 2, 4, 6, 8]:
        # the indices follow the order from torso to limbs
        # only compute bone vectors for non-torsos
        bone = normalize(bones_local[bone_idx])
        if bone_idx in [0, 1, 3, 5, 7]:
            # bones that are directly connected to the torso
            if bone_idx in [0, 1, 3]:
                # upper body
                # this is the inverse transformation compared to the to_local
                # function
                bones_global[bone_idx] = system1 @ bone
            else:
                # lower body
                bones_global[bone_idx] = system2 @ bone
        else:
            if bone_idx in [2, 4]:
                parent_R = system1
            else:
                parent_R = system2
            # parent bone index is smaller than 1
            vector_u = normalize(bones_global[bone_idx - 1])
            di_index = di_indices[bone_idx]
            vector_v, flag = get_normal(parent_R @ di[:, di_index],
                                        parent_R @ a,
                                        vector_u)
            vector_w = np.cross(vector_u, vector_v)
            local_system = gram_schmidt_columns(np.hstack([vector_u.reshape(3, 1),
                                                           vector_v.reshape(3, 1),
                                                           vector_w.reshape(3, 1)]))
            if cache:
                return_value['cache'].append(local_system)
            bones_global[bone_idx] = local_system @ bone
    return_value['bg'] = bones_global
    return return_value


# sampling utilities: sample 3D human skeleton from a pre-computed distribution
template = np.load(os.path.join(root, 'template.npy'), allow_pickle=True).reshape(32, -1)
template_bones = template[nt_parent_indices, :] - template[nt_child_indices, :]
template_bone_lengths = to_spherical(template_bones)[:, 0]

nt_parent_indices = [8, 11, 12, 14, 15, 4, 5, 1, 2]
nt_child_indices = [10, 12, 13, 15, 16, 5, 6, 2, 3]


def get_skeleton(bones, pose, bone_length=template_bone_lengths):
    """
    Update the non-torso limb of a skeleton by specifying bone vectors.
    """
    new_pose = pose.copy()
    for bone_idx in [0, 1, 3, 5, 7, 2, 4, 6, 8]:
        new_pose[nt_child_indices[bone_idx]] = new_pose[nt_parent_indices[bone_idx]] \
                                               - bones[bone_idx] * bone_length[bone_idx]
    return new_pose
