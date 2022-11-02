"""
Utility functions for genetic evolution.
"""
from lib.evolution.anglelimits import \
    to_spherical, \
    nt_parent_indices, nt_child_indices, to_local, to_global, get_skeleton, is_valid_local


import matplotlib.pyplot as plt
import os
import logging

import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

root = "resources/constraints"
# Joints in H3.6M -- data has 32 joints, but only 17 that move
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'  # 0
H36M_NAMES[1] = 'RHip'  # 1
H36M_NAMES[2] = 'RKnee'  # 2
H36M_NAMES[3] = 'RFoot'  # 3
H36M_NAMES[6] = 'LHip'  # 4
H36M_NAMES[7] = 'LKnee'  # 5
H36M_NAMES[8] = 'LFoot'  # 6
H36M_NAMES[12] = 'Spine'  # 7
H36M_NAMES[13] = 'Thorax'  # 8
H36M_NAMES[14] = 'Neck/Nose'  # 9
H36M_NAMES[15] = 'Head'  # 10
H36M_NAMES[17] = 'LShoulder'  # 11
H36M_NAMES[18] = 'LElbow'  # 12
H36M_NAMES[19] = 'LWrist'  # 13
H36M_NAMES[25] = 'RShoulder'  # 14
H36M_NAMES[26] = 'RElbow'  # 15
H36M_NAMES[27] = 'RWrist'  # 16
total_joints_num = 17

# this dictionary stores the parent indice for each joint
# key:value -> child joint index:its parent joint index
parent_idx = {1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 7: 0, 8: 7, 9: 8, 10: 9, 11: 8,
              12: 11, 13: 12, 14: 8, 15: 14, 16: 15
              }

# this dictionary stores the children indices for each parent joint
# key:value -> parent index: joint indices for its children as a list
children_idx = {
    0: [1, 4],
    1: [2], 2: [3],
    4: [5], 5: [6],
    8: [9, 11, 14],
    9: [10], 11: [12], 12: [13],
    14: [15], 15: [16]
}

# used roots for random selection
root_joints = [0, 1, 2, 4, 5, 8, 11, 12, 14, 15]

# names of the bone vectors attached on the human torso
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
# this dictionary stores the sub-tree rooted at each root joint
# key:value->root joint index:list of bone vector indices
bone_indices = {0: [5, 6, 7, 8],
                1: [7, 8],
                2: [8],
                4: [5, 6],
                5: [6],
                8: [1, 2, 3, 4],  # thorax
                11: [1, 2],
                12: [2],
                14: [3, 4],
                15: [4]
                }

# load template bone lengths that can be used during mutation
# you can prepare your own bone length templates to represent
# subjects with different size
bl_templates = np.load(os.path.join(root, "bones.npy"), allow_pickle=True)

# pre-compute the sub-tree joint indices for each joint
subtree_indices = {}


def get_subtree(joint_idx, children_idx):
    if joint_idx not in children_idx:
        return None
    subtree = set()
    for child_idx in children_idx[joint_idx]:
        subtree.add(child_idx)
        offsprings = get_subtree(child_idx, children_idx)
        if offsprings is not None:
            subtree = subtree.union(offsprings)
    return subtree


for joint_idx in range(total_joints_num):
    if H36M_NAMES[joint_idx] != '':
        subtree_indices[joint_idx] = get_subtree(joint_idx, children_idx)


def swap_bones(bones_father, bones_mother, root_idx):
    swap_indices = bone_indices[root_idx]
    temp = bones_father.copy()
    bones_father[swap_indices] = bones_mother[swap_indices].copy()
    bones_mother[swap_indices] = temp[swap_indices].copy()
    del temp
    return bones_father, bones_mother, swap_indices


def get_bone_length(skeleton):
    """
    Compute limb length for a given skeleton.
    """
    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]
    bone_lengths = to_spherical(bones)[:, 0]
    return bone_lengths


def get_random_rotation(sigma=60.):
    angle = np.random.normal(scale=sigma)
    axis_idx = np.random.choice(3, 1)
    if axis_idx == 0:
        r = R.from_euler('xyz', [angle, 0., 0.], degrees=True)
    elif axis_idx == 1:
        r = R.from_euler('xyz', [0., angle, 0.], degrees=True)
    else:
        r = R.from_euler('xyz', [0., 0., angle], degrees=True)
    return r


def rotate_bone_random(bone, sigma=10.):
    r = get_random_rotation(sigma)
    bone_rot = r.as_matrix() @ bone.reshape(3, 1)
    return bone_rot.reshape(3)


def rotate_pose_random(pose=None, sigma=60.):
    # pose shape: [n_joints, 3]
    if pose is None:
        result = None
    else:
        r = get_random_rotation()
        pose = pose.reshape(total_joints_num, 3)
        # rotate around hip
        hip = pose[0].reshape(1, 3)
        relative_pose = pose - hip
        rotated = r.as_matrix() @ relative_pose.T
        result = rotated.T + hip
    return result


def re_order(skeleton):
    # the ordering of coordinate used by the Prior was x,z and y
    return skeleton[:, [0, 2, 1]]


def set_z(pose, target):
    if pose is None:
        return None
    original_shape = pose.shape
    pose = pose.reshape(total_joints_num, 3)
    min_val = pose[:, 2].min()
    pose[:, 2] -= min_val - target
    return pose.reshape(original_shape)


def modify_pose(skeleton, local_bones, bone_length, ro=False):
    # get a new pose by modify an existing pose with input local bone vectors
    # and bone lengths
    new_bones = to_global(skeleton, local_bones)['bg']
    new_pose = get_skeleton(new_bones, skeleton, bone_length=bone_length)
    if ro:
        new_pose = re_order(new_pose)
    return new_pose.reshape(-1)


def exploration(father, mother, opt, post_processing=True):
    """
    Produce novel data by exploring the data space with evolutionary operators.
    cross over operator in the local coordinate system
    mutation: perturb the local joint angle
    """
    # get local coordinate for each bone vector
    father = re_order(father.reshape(total_joints_num, -1))
    father_bone_length = get_bone_length(father)
    mother = re_order(mother.reshape(total_joints_num, -1))
    mother_bone_length = get_bone_length(mother)
    bones_father = to_local(father)
    bones_mother = to_local(mother)
    if opt.CV:
        # crossover: exchange random sub-trees of two kinematic trees
        root_idx = np.random.randint(0, len(root_joints))
        root_selected = root_joints[root_idx]
        bones_father, bones_mother, indices = swap_bones(bones_father,
                                                         bones_mother,
                                                         root_selected)
    if opt.M:
        # local mutation: apply random rotation to local limb
        for bone_idx in indices:
            if np.random.rand() <= opt.MRL:
                bones_father[bone_idx] = rotate_bone_random(bones_father[bone_idx], sigma=opt.SDL)
                bones_mother[bone_idx] = rotate_bone_random(bones_mother[bone_idx], sigma=opt.SDL)

    son_pose, daughter_pose = None, None
    if opt.C:
        # apply joint angle constraint as the fitness function
        valid_vec_fa = is_valid_local(bones_father)
        valid_vec_mo = is_valid_local(bones_mother)

    if not opt.C or valid_vec_fa.sum() >= opt.Th:
        son_pose = modify_pose(father, bones_father, mother_bone_length, ro=True)
    if not opt.C or valid_vec_mo.sum() >= opt.Th:
        daughter_pose = modify_pose(mother, bones_mother, father_bone_length, ro=True)
    if opt.M:
        # global mutation: rotate the whole 3D skeleton
        if np.random.rand() <= opt.MRG:
            son_pose = rotate_pose_random(son_pose, sigma=opt.SDG)
        if np.random.rand() <= opt.MRG:
            daughter_pose = rotate_pose_random(daughter_pose, sigma=opt.SDG)
    if post_processing:
        # move the poses to the ground plane
        set_z(son_pose, np.random.normal(loc=20.0, scale=3.0))
        set_z(daughter_pose, np.random.normal(loc=20.0, scale=3.0))

    return son_pose, daughter_pose


def choose_best(population, fraction=0.02, method='random'):
    """
    Choose the best candidates to produce descendents.
    """
    if method == 'random':
        # this is a simple implementation by random sampling
        num_total = len(population)
        num_to_choose = int(fraction * num_total)
        chosen_indices = np.random.choice(num_total, num_to_choose * 2, replace=False)
        father_indices = chosen_indices[:num_to_choose]
        mother_indices = chosen_indices[num_to_choose:]
    else:
        raise NotImplementedError
    return father_indices, mother_indices


def normalize(data, mean=None, std=None):
    if mean is not None and std is not None:
        pass
    elif mean is None and std is None:
        mean = np.mean(data, axis=0).reshape(1, data.shape[1])
        std = np.std(data, axis=0).reshape(1, data.shape[1])
    else:
        raise ValueError
    return (data - mean) / std


def unnormalize(data, mean, std):
    return (data * std) + mean


def postprocess_3d(poses):
    return poses - np.tile(poses[:, :3], [1, total_joints_num])  # TODO: check this


def calc_errors(pred_poses, gt_poses, protocol='mpjpe'):
    # error after a regid alignment, corresponding to protocol #2 in the paper
    # Compute Euclidean distance error per joint
    sqerr = (pred_poses - gt_poses) ** 2  # Squared error between prediction and expected output
    sqerr = sqerr.reshape(len(sqerr), -1, 3)
    sqerr = np.sqrt(sqerr.sum(axis=2))
    if protocol == 'mpjpe':
        ret = sqerr.mean(axis=1)
        ret = ret.reshape(len(ret), 1)
    else:
        raise NotImplementedError
    return ret


def to_numpy(tensor):
    return tensor.data.cpu().numpy()


def get_prediction(cascade, data):
    data = torch.from_numpy(data.astype(np.float32))
    if torch.cuda.is_available():
        data = data.cuda()
    # forward pass to get prediction for the first stage
    prediction = cascade[0](data)
    # prediction for later stages
    for stage_idx in range(1, len(cascade)):
        prediction += cascade[stage_idx](data)
    return prediction


def cast_to_float(dic, dtype=np.float32):
    # cast to float 32 for space saving
    for key in dic.keys():
        dic[key] = dic[key].astype(dtype)
    return dic


def xyz2spherical(xyz):
    # convert cartesian coordinate to spherical coordinate
    # return in r, phi, and theta (elevation angle from z axis down)
    return_value = np.zeros(xyz.shape, dtype=xyz.dtype)
    xy = xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2
    return_value[:, :, 0] = np.sqrt(xy + xyz[:, :, 2] ** 2)  # r
    return_value[:, :, 1] = np.arctan2(np.sqrt(xy), xyz[:, :, 2])  # phi
    return_value[:, :, 2] = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])  # theta
    return return_value


def spherical2xyz(rphitheta):
    return_value = np.zeros(rphitheta.shape, dtype=rphitheta.dtype)
    sinphi = np.sin(rphitheta[:, :, 1])
    cosphi = np.cos(rphitheta[:, :, 1])
    sintheta = np.sin(rphitheta[:, :, 2])
    costheta = np.cos(rphitheta[:, :, 2])
    return_value[:, :, 0] = rphitheta[:, :, 0] * sinphi * costheta  # x
    return_value[:, :, 1] = rphitheta[:, :, 0] * sinphi * sintheta  # y
    return_value[:, :, 2] = rphitheta[:, :, 0] * cosphi  # z
    return return_value


# global variables
parent_idx = [0, 4, 5,
              0, 1, 3,
              0, 7, 8, 9,
              8, 11, 12,
              8, 14, 15]
child_idx = [4, 5, 6,
             3, 2, 3,
             7, 8, 9, 10,
             11, 12, 13,
             14, 15, 16]


def position_to_angle(skeletons):
    # transform 3d positions to joint angle representation

    # first compute the bone vectors
    # a bone vector is the vector from on parent joint to one child joint
    # hip->left hip->left knee->left foot,
    # hip->right hip-> right knee-> right foot
    # hip -> spine->thorax->nose->head
    # thorax -> left shoulder->left elbow->left wrist
    # thorax -> right shoulder-> right elbow->right wrist
    num_sample = skeletons.shape[0]
    skeletons = skeletons.reshape(num_sample, -1, 3)

    parent_joints = skeletons[:, parent_idx, :]
    child_joints = skeletons[:, child_idx, :]
    bone_vectors = child_joints - parent_joints
    # now compute the angles and bone lengths
    rphitheta = xyz2spherical(bone_vectors)
    return rphitheta


def angle_to_position(rphitheta, skeletons):
    # transform joint angle representation to 3d positions
    # starting from the root, create joint one by one according to predefined
    # hierarchical relation
    num_sample = skeletons.shape[0]
    skeletons = skeletons.reshape(num_sample, -1, 3)
    for bone_idx in range(len(parent_idx)):
        offset = spherical2xyz(np.expand_dims(rphitheta[:, bone_idx, :], axis=1))
        offset = offset[:, 0, :]
        skeletons[:, child_idx[bone_idx], :] = \
            skeletons[:, parent_idx[bone_idx], :] + offset
    return skeletons


def mutate_bone_length(population, opt, gen_idx, method='simple'):
    """
    Randomly mutate bone length in a population to increase variation in
    subject size.
    For example, H36M only contains adults yet you can modify bone
    length to represent children. Since the posture and subject size are
    independent, you can synthetize data for dancing kids for free if you already
    have data for dancing adults. You only need little prior knowledge on human
    bone length.
    """
    # the camera parameters in H36M correspond to the five subjects
    # Rename the synthetic population as these subjects so that the camera
    # parameters can be used
    psuedo_subject_names = [1, 5, 6, 7, 8]
    dict_3d = {}
    for i in range(len(population)):
        if np.random.rand() > opt.MBLR:
            angles = position_to_angle(population[i].reshape(1, -1))
            if method == 'simple':
                # The simplest way is to change to bone length to some value
                # according to prior knowledge about human bone size.
                # In our experiment, we collect these values manually from our
                # interactive visualization tool as well as cross validation.
                idx = np.random.randint(0, len(bl_templates))
                angles[0, :, 0] = bl_templates[idx]
                population[i] = (angle_to_position(angles, population[i].reshape(1, -1))).reshape(-1)
            elif method == 'addnoise':
                # add Gaussian noise to current bone length to obtain new bone length
                raise ValueError('Deprecated')
            else:
                raise NotImplementedError
    poses_list = np.array_split(population, len(psuedo_subject_names))
    for subject_idx in range(len(psuedo_subject_names)):
        dict_3d[(psuedo_subject_names[subject_idx], 'n/a', 'n/a')] = \
            poses_list[subject_idx]
    save_path = get_save_path(opt, gen_idx)
    np.save(save_path, cast_to_float(dict_3d))
    logging.info('file saved at ' + save_path)
    return


def one_iteration(population, opt, model_file=None):
    """
    Run one iteration to produce the next generation.
    """
    # select the best individuals
    father_indices, mother_indices = choose_best(population, fraction=opt.F)
    # produce next generation by evolutionary operators
    offsprings = []
    for idx in tqdm(range(len(father_indices))):
        son, daughter = exploration(population[father_indices[idx]],
                                    population[mother_indices[idx]],
                                    opt)
        if son is not None:
            offsprings.append(son.reshape(1, -1))
        if daughter is not None:
            offsprings.append(daughter.reshape(1, -1))
    offsprings = np.concatenate(offsprings, axis=0)
    logging.info('{:d} out of {:d} poses survived.'.format(len(offsprings),
                                                           len(father_indices) * 2))

    if opt.Mer:
        # merge the offsprings with the parents
        population = np.vstack([population, offsprings])
    else:
        population = offsprings
    return population


def get_save_path(opt, gen_idx):
    if opt.WS:
        save_path = os.path.join(opt.SD, opt.SS, opt.SN)
    else:
        save_path = os.path.join(opt.SD, 'S15678', opt.SN)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'generation_{:d}.npy'.format(gen_idx))
    return save_path


def split_and_save(final_poses, parameters, gen_idx):
    temp_subject_list = [1, 5, 6, 7, 8]
    train_set_3d = {}
    poses_list = np.array_split(final_poses, len(temp_subject_list))
    for subject_idx in range(len(temp_subject_list)):
        train_set_3d[(temp_subject_list[subject_idx], 'n/a', 'n/a')] = \
            poses_list[subject_idx]
    save_path = get_save_path(parameters, gen_idx)
    np.save(save_path, cast_to_float(train_set_3d))
    print('file saved at {:s}!'.format(save_path))
    return


def save_results(poses, opt, gen_idx):
    # get save path
    if opt.MBL:
        mutate_bone_length(poses, opt, gen_idx)
    else:
        split_and_save(poses, opt, gen_idx)
    return


def evolution(initial_population, opt, model_file=None):
    """
    Dataset evolution.
    """
    population = initial_population
    initial_num = len(initial_population)
    for gen_idx in range(1, opt.G + 1):
        population = one_iteration(population, opt, model_file=model_file)
    # if not enough
    if opt.E and len(population) < initial_num * opt.T:
        while len(population) < initial_num * opt.T:
            gen_idx += 1
            population = one_iteration(population, opt, model_file=model_file)
            if opt.I:
                save_results(population.copy(), opt, gen_idx)
    return population
