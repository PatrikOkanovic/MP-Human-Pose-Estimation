import cv2
import torch
import pickle as pkl
import json
import random
import os
import logging
import numpy as np
import copy
import re

from lib.utils.prep_h36m import CamBackProj  # , compute_similarity_transform
from lib.dataset.JointDataset import JointDataset, H36M_NAMES
from lib.utils.data_utils import define_actions
from lib.utils.img_utils import get_single_patch_sample
from lib.utils.utils import calc_kpt_bound

logger = logging.getLogger(__name__)

H36M_NAMES
header_pred = 'Id,'
header_gt = 'Id,'
for idx, name in enumerate(H36M_NAMES):
    header_pred += name + '_x,'
    header_pred += name + '_y,'
    header_pred += name + '_z'

    if idx < len(H36M_NAMES) - 1:
        header_pred += ','

    header_gt += name + '_x_gt,'
    header_gt += name + '_y_gt,'
    header_gt += name + '_z_gt,'

header_gt += 'Split'

fmt_pred = ['%d'] + ['%.4f'] * 51
fmt_gt = ['%d'] + ['%.4f'] * 51 + ['%d']


class H36M(JointDataset):
    def __init__(self, cfg, root, image_set, is_train, always_flip=False):
        super().__init__(cfg, root, image_set, is_train, always_flip)

        self.parent_ids = np.array([0, 0, 1, 2, 0, 4, 5, 0, 8, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int)

        self.flip_pairs = np.array([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int)

        self.num_joints = 17
        self.is_train = is_train

        self.db = self._get_db()
        image_names = [self.db[i].get("image") for i in range(self.db_length)]
        image_names = [re.search('/\w\d\d?/', image_names[i]) for i in range(0, self.db_length)]
        logger.info(
            f'=> set contains the following subjects: {sorted(set([image_names[i][0] for i in range(0, self.db_length)]))}')

        logger.info('=> load {} samples'.format(self.db_length))

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        return self.get_data(db_rec)

    def get_data(self, the_db):

        if the_db['is_h36m']:
            image_file = os.path.join(self.root, the_db['image'])
        else:
            image_file = the_db['image']

        if the_db['is_h36m']:
            cam = the_db['cam']
        else:
            cam = None

        if 'joints_3d_vis' in the_db.keys() and 'joints_3d' in the_db.keys():
            joints = the_db['joints_3d'].copy()
            joints_vis = the_db['joints_3d_vis'].copy()
            joints_vis[:, 2] *= self.cfg.DATASET.Z_WEIGHT
        else:
            joints = joints_vis = None

        img_patch, label, label_weight, scale, rot = get_single_patch_sample(image_file, the_db['center_x'],
                                                                             the_db['center_y'], the_db['width'],
                                                                             the_db['height'], self.flip_pairs,
                                                                             self.patch_width, self.patch_height,
                                                                             self.rect_3d_width, self.mean, self.std,
                                                                             self.cfg.DATASET.AUGMENT, self.label_func,
                                                                             occluder=self.occluders, joints=joints,
                                                                             joints_vis=joints_vis,
                                                                             is_train=self.is_train,
                                                                             always_flip=self.always_flip)

        meta = {
            'image': image_file,
            'center_x': the_db['center_x'],
            'center_y': the_db['center_y'],
            'width': the_db['width'],
            'height': the_db['height'],
            'scale': float(scale),
            'rot': float(rot),
            'R': cam.R if cam is not None else np.zeros((3, 3), dtype=np.float64),
            'T': cam.T if cam is not None else np.zeros((3, 1), dtype=np.float64),
            'f': cam.f if cam is not None else np.zeros((2, 1), dtype=np.float64),
            'c': cam.c if cam is not None else np.zeros((2, 1), dtype=np.float64),
            'projection_matrix': cam.projection_matrix if cam is not None else np.zeros((3, 4), dtype=np.float64)
        }

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32), meta

    def _get_h36m_db(self):
        # create train/val/test split for H36M
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set + '.pkl')

        with open(file_name, 'rb') as anno_file:
            anno = pkl.load(anno_file)

        if isinstance(anno, dict):
            gt_db = [[] for i in range(self.num_cams)]  # for each cameras construct a database
            # rnd_subset = np.random.permutation(len(anno[1]))
            # for idx in rnd_subset:
            for idx in range(len(anno[1])):
                for cid in range(self.num_cams):
                    a = anno[cid + 1][idx]
                    a['is_h36m'] = True
                    gt_db[cid].append(a)

            # self.db_length = len(gt_db[0])

            temp_db = []
            for db in gt_db:
                temp_db += db
            gt_db = temp_db
        else:
            gt_db = []

            for idx in range(len(anno)):
                a = anno[idx]
                a['is_h36m'] = True
                gt_db.append(a)

        return gt_db

    def _get_db(self):
        gt_db = self._get_h36m_db()

        self.db_length = len(gt_db)

        return gt_db

    def evaluate(self, preds, save_path=None, debug=False, writer_dict=None, time_str=''):
        preds = preds[:, :, 0:3]

        gt_poses_glob = []
        pred_poses_glob = []
        pred_2d_poses = []
        all_images = []

        gts = self.db

        sample_num = preds.shape[0]
        root = 0
        pred_to_save = []
        gt_to_save = []

        j14 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]

        dist = []
        dist_align = []
        dist_norm = []
        dist_14 = []
        dist_14_align = []
        dist_14_norm = []
        dist_x = []
        dist_y = []
        dist_z = []
        dist_per_joint = []
        pck = []

        for n_sample in range(0, sample_num):
            gt = gts[n_sample]
            # Org image info
            fl = gt['fl'][0:2]
            c_p = gt['c_p'][0:2]

            gt_3d_root = np.reshape(gt['pelvis'], (1, 3))

            if 'joints_3d' in gt.keys() and 'joints_3d_vis' in gt.keys():
                gt_2d_kpt = gt['joints_3d'].copy()
                has_gt = True
            else:
                gt_2d_kpt = np.zeros((17, 3))
                has_gt = False

            # get camera depth from root joint
            pre_2d_kpt = preds[n_sample].copy()

            pre_2d_kpt[:, 2] = pre_2d_kpt[:, 2] + gt_3d_root[0, 2]
            gt_2d_kpt[:, 2] = gt_2d_kpt[:, 2] + gt_3d_root[0, 2]
            joint_num = pre_2d_kpt.shape[0]

            # back project
            pre_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
            gt_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)

            for n_jt in range(0, joint_num):
                pre_3d_kpt[n_jt, 0], pre_3d_kpt[n_jt, 1], pre_3d_kpt[n_jt, 2] = \
                    CamBackProj(pre_2d_kpt[n_jt, 0], pre_2d_kpt[n_jt, 1], pre_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])
                gt_3d_kpt[n_jt, 0], gt_3d_kpt[n_jt, 1], gt_3d_kpt[n_jt, 2] = \
                    CamBackProj(gt_2d_kpt[n_jt, 0], gt_2d_kpt[n_jt, 1], gt_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])

            # This is computing the PA-MPJPE metric. We are not using this for benchmark, but leave the code
            # here for people who are interested.
            # _, Z, T, b, c = compute_similarity_transform(gt_3d_kpt, pre_3d_kpt, compute_optimal_scale=True)
            # pre_3d_kpt_align = (b * pre_3d_kpt.dot(T)) + c
            # pre_3d_kpt_norm = b * pre_3d_kpt

            # should align root, required by protocol #1
            pre_3d_kpt = pre_3d_kpt - pre_3d_kpt[root]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[root]
            # pre_3d_kpt_align = pre_3d_kpt_align - pre_3d_kpt_align[root]
            # pre_3d_kpt_norm = pre_3d_kpt_norm - pre_3d_kpt_norm[root]

            diff = (gt_3d_kpt - pre_3d_kpt)
            # diff_align = (gt_3d_kpt - pre_3d_kpt_align)
            # diff_norm = (gt_3d_kpt - pre_3d_kpt_norm)

            e_jt = []
            # e_jt_align = []
            # e_jt_norm = []
            e_jt_14 = []
            # e_jt_14_align = []
            # e_jt_14_norm = []
            e_jt_x = []
            e_jt_y = []
            e_jt_z = []

            for n_jt in range(0, joint_num):
                e_jt.append(np.linalg.norm(diff[n_jt]))
                # e_jt_align.append(np.linalg.norm(diff_align[n_jt]))
                # e_jt_norm.append(np.linalg.norm(diff_norm[n_jt]))
                e_jt_x.append(np.sqrt(diff[n_jt][0] ** 2))
                e_jt_y.append(np.sqrt(diff[n_jt][1] ** 2))
                e_jt_z.append(np.sqrt(diff[n_jt][2] ** 2))

                if np.linalg.norm(diff[n_jt]) >= 150:
                    pck.append(0)
                else:
                    pck.append(1)

            for jt in j14:
                e_jt_14.append(np.linalg.norm(diff[jt]))
                # e_jt_14_align.append(np.linalg.norm(diff_align[jt]))
                # e_jt_14_norm.append(np.linalg.norm(diff_norm[jt]))

            if self.cfg.DEBUG.DEBUG and n_sample % 100 == 0:
                cam = gt['cam']
                pred = cam.camera_to_world_frame(pre_3d_kpt)
                gt_pt = cam.camera_to_world_frame(gt_3d_kpt)

                gt_poses_glob.append(gt_pt)
                pred_poses_glob.append(pred)
                pred_2d_poses.append(pre_2d_kpt)
                all_images.append(self.root + gts[n_sample]['image'])
                import cv2
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                import matplotlib.pyplot as plt
                from lib.utils.vis import drawskeleton, show3Dpose

                m = 2  # 1 for MPII joints, 2 for H36M joints

                img = cv2.imread(self.root + gts[n_sample]['image'])
                fig = plt.figure(figsize=(19.2, 10.8))
                canvas = FigureCanvas(fig)
                lc = (255, 0, 0), '#ff0000'
                rc = (0, 0, 255), '#0000ff'
                ax = fig.add_subplot(131)
                drawskeleton(img, pre_2d_kpt, thickness=3, lcolor=lc[0], rcolor=rc[0], mpii=m)
                drawskeleton(img, gt_2d_kpt, thickness=3, lcolor=(0, 255, 0), rcolor=(0, 255, 0), mpii=m)
                ax.imshow(img[:, :, ::-1])
                ax.set_title('Predictions in 2D')

                ax = fig.add_subplot(132, projection='3d')
                show3Dpose(gt_pt, ax, radius=750, lcolor=lc[1], rcolor=rc[1], mpii=m)
                ax.set_title('Ground-truth 3D')

                ax = fig.add_subplot(133, projection='3d')
                show3Dpose(pred, ax, radius=750, lcolor=lc[1], rcolor=rc[1], mpii=m)
                ax.set_title('Prediction 3D ' + ' %s' % np.array(e_jt).mean())

                # plt.show()
                # plt.savefig('tmp.png')
                canvas.draw()
                vis = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
                vis = vis.reshape(canvas.get_width_height()[::-1] + (3,))
                # cv2.imwrite('tmp.png', vis)
                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_image('result_vis', vis, global_steps, dataformats='HWC')
                writer_dict['valid_global_steps'] = global_steps + 1

            dist.append(np.array(e_jt).mean())
            # dist_align.append(np.array(e_jt_align).mean())
            # dist_norm.append(np.array(e_jt_norm).mean())
            dist_14.append(np.array(e_jt_14).mean())
            # dist_14_align.append(np.array(e_jt_14_align).mean())
            # dist_14_norm.append(np.array(e_jt_14_norm).mean())
            dist_x.append(np.array(e_jt_x).mean())
            dist_y.append(np.array(e_jt_y).mean())
            dist_z.append(np.array(e_jt_z).mean())
            dist_per_joint.append(np.array(e_jt))

            subj = gt['image'].split('/')[1]
            split = 0 if subj == 'S9' else 1
            if has_gt:
                pred_to_save.append(np.hstack((n_sample, pre_3d_kpt.flatten())))
                gt_to_save.append(np.hstack((n_sample, gt_3d_kpt.flatten(), split)))
            else:
                pred_to_save.append(np.hstack((n_sample, pre_3d_kpt.flatten())))

        if has_gt:
            per_joint_error = np.array(dist_per_joint).mean(axis=0).tolist()
            joint_names = H36M_NAMES

            for idx in range(len(joint_names)):
                print(joint_names[idx], per_joint_error[idx])
            print("Average Joint Error", np.mean(per_joint_error))

            name_value = [
                ('hm36_17j      :', np.asarray(dist).mean()),
                ('hm36_17j_14   :', np.asarray(dist_14).mean()),
                ('hm36_17j_x    :', np.array(dist_x).mean()),
                ('hm36_17j_y    :', np.array(dist_y).mean()),
                ('hm36_17j_z    :', np.array(dist_z).mean()),
            ]
        else:
            name_value = []

        if save_path is not None:
            pred_save_path = os.path.join(save_path, '{}_pred_{}.csv'.format(self.image_set, time_str))
            pred_to_save = np.stack(pred_to_save, axis=0)
            np.savetxt(pred_save_path, pred_to_save, delimiter=',', header=header_pred, fmt=fmt_pred, comments='')
            if has_gt:
                gt_save_path = os.path.join(save_path, '{}_gt_{}.csv'.format(self.image_set, time_str))
                gt_to_save = np.stack(gt_to_save, axis=0)
                np.savetxt(gt_save_path, gt_to_save, delimiter=',', header=header_gt, fmt=fmt_gt, comments='')

        return name_value, np.asarray(dist).mean()

    def plot_2d_images_with_coords(self, img, pre_2d_kpt, gt_2d_kpt):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import matplotlib.pyplot as plt
        from lib.utils.vis import drawskeleton
        img = img.permute([1, 2, 0]).numpy()
        for i in range(0, 3):
            img[:, :, i] = (img[:, :, i] * self.std[i]) + self.mean[i]
        img = cv2.UMat(np.int8(img))

        pre_2d_kpt = pre_2d_kpt[:, 0:2]
        gt_2d_kpt = gt_2d_kpt[:, 0:2]

        m = 2  # 1 for MPII joints, 2 for H36M joints

        fig = plt.figure(figsize=(19.2, 10.8))
        canvas = FigureCanvas(fig)
        lc = (255, 0, 0), '#ff0000'
        rc = (0, 0, 255), '#0000ff'
        ax = fig.add_subplot(111)
        drawskeleton(img, pre_2d_kpt, thickness=1, lcolor=lc[0], rcolor=rc[0], mpii=m)
        drawskeleton(img, gt_2d_kpt, thickness=1, lcolor=(0, 255, 0), rcolor=(0, 255, 0), mpii=m)
        ax.imshow(img.get()[:, :, ::-1])
        ax.set_title('Predictions in 2D')
        canvas.draw()
        plt.show()
