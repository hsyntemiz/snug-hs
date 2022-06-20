# import pickle
#
# import numpy as np
# import tensorflow as tf
# import os
import random
import tensorflow as tf
import numpy as np
# from Data.smpl import smpl_np
# from Data.smpl.smpl_np import SMPLModel
from scipy.spatial.transform import Rotation as R

import glob
def load_motion(path):
    print(path)
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
    num_joints = poses.shape[-1] //3

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

    return finite_diff(v, h, diff-1)


class AMASS():
    def __init__(self, dataset_path, **kwargs):
        super(AMASS, self).__init__( **kwargs)
        self.pose=[]
        self.trans = []
        self.trans_vel = []
        self.pose3 =[]
        self.G3 = []
        self.trans3 = []
        self.trans_vel3= []
        self.train_list = ['07_02', '91_62', '91_33', '91_36', '91_19', '91_12', '76_11', '73_09', '46_01', '111_14', '105_05', '104_17', '128_02', '128_03', '128_04', '128_05', '128_06', '128_07', '108_11', '108_12', '108_27', '104_04', '104_53', '104_54', '01_01', '91_61', '91_38', '49_04', '108_16', '108_17', '108_18', '108_20', '13_13', '02_04', '55_27', '54_24', '40_12', '144_26', '144_30', '135_09', '132_54', '111_20', '111_37', '02_05', '63_25', '144_28', '111_02', '108_22', '26_11', '135_07', '05_20', '55_12', '131_03', '111_05', '76_05', '104_11']
        #self.train_list = ['07_02','01_01', '07_02', '104_54', '55_27', '91_62', '91_33', '91_36', '91_19']# '91_12', '76_11', '73_09', '46_01', '111_14', '105_05', '104_17', '128_02', '128_03', '128_04', '128_05', '128_06', '128_07', '108_11', '108_12', '108_27', '104_04', '104_53', '104_54', '01_01', '91_61', '91_38', '49_04', '108_16', '108_17', '108_18', '108_20', '13_13', '02_04', '55_27', '54_24', '40_12', '144_26', '144_30', '135_09', '132_54', '111_20', '111_37', '02_05', '63_25', '144_28', '111_02', '108_22', '26_11', '135_07', '05_20', '55_12', '131_03', '111_05', '76_05', '104_11']
        # self.train_list = random.sample(self.train_list,5)
        self.test_list = ['01_01', '07_02', '104_54', '55_27']


        self.amass_file_paths = glob.glob('assets/CMU2/**/*poses*')
        print(self.amass_file_paths)
        for file_name in self.amass_file_paths:
            pose_name = file_name.split('/')[-1]
            if not pose_name[:-10] in self.train_list:
                print(pose_name[:-10])
                continue
            poses, trans, trans_vel = load_motion(file_name)
            print(file_name)
            self.pose.append(poses)
            self.trans.append(trans)
            self.trans_vel.append(trans_vel)

        print('oylesine')
        self.frame(3)
        print('oylesine2')
        print('oylesine3')
        self.G_topla()

        self.dataset = tf.data.Dataset.from_tensor_slices((self.pose3,self.trans3,self.trans_vel3,self.G3))

        print('allaaahhj')

    def frame(self,window=3):

        for i,pose in enumerate(self.pose):
            trans = self.trans[i]
            trans_vel = self.trans_vel[i]
            num_frames = pose.shape[0]
            num_frame_seq = num_frames//window
            for j in range(num_frame_seq):
                self.pose3.append(pose[3*j:3*j+3,:])
                self.trans3.append(trans[3*j:3*j+3,:])
                self.trans_vel3.append(trans_vel[3*j:3*j+3,:])



    def G_topla(self):
        smpl_path = ''
        gender = "f"
        self.G = []
        rest_pose = np.zeros((24, 3))
        rest_pose[0, 0] = np.pi / 2
        rest_pose[1, 2] = .15
        rest_pose[2, 2] = -.15

        # gender = 'f'
        # smpl_path += 'Data/smpl/model_[G].pkl'.replace('[G]', 'm' if gender else 'f')
        smpl_path = 'Data/smpl/model_f.pkl'
        print('smpl-path-dataloader:',smpl_path)
        self.SMPL = SMPLModel(smpl_path, rest_pose)
        beta = np.zeros(10)


        for pose in self.pose3:
            g_list = []
            for p in pose:
                G = self.SMPL.update(p, beta, with_body=False)
                g_list.append(G[0].astype(np.float32))
            self.G3.append(np.array(g_list))

        print('lololo')


if 0:
    mydata = AMASS('assets/CMU')
    print(mydata.dataset.batch(8))
    print(mydata)
    iterator = iter(mydata.dataset.batch(8))
    # for i in range(10000):
    #     print(i)
    #     #a=mydata.dataset.batch(8)
    next_true = True
    i=0
    while next_true:
        optional = iterator.get_next_as_optional()
        if not optional.has_value():
            next_true = False
            continue

        optional = iterator.get_next_as_optional()
        # optional = iterator.get_next()
        print(optional.has_value())
        # print(optional.get_value())
        poses, trans, trans_vel = optional.get_value()
        print(poses.shape,trans.shape,trans_vel.shape)
        print(i,'-------------------------')
        i=i+1

if 0:
    mydata = AMASS('assets/CMU2')
    #self.amass_file_paths = glob.glob('assets/CMU2/**/*poses*')
    print(mydata.amass_file_paths)
    test_poses = []
    train_poses = []
    for file_name in mydata.amass_file_paths:
        pose_name = file_name.split('/')[-1]
        if pose_name[:-10] in mydata.train_list:
            print(pose_name[:-10],file_name)
            poses, trans, trans_vel = load_motion(file_name)
            train_poses.append(poses)

        if pose_name[:-10] in mydata.test_list:
            print('test',pose_name[:-10], file_name)
            poses, trans, trans_vel = load_motion(file_name)
            test_poses.append(poses)

    print(train_poses)

    trainn = np.vstack((x for x in train_poses))
    testt = np.vstack((x for x in test_poses))
    np.save('train.py',trainn)
    np.save('test.py',testt)

    print('allah')
    #         continue
    #

# mydata = AMASS('assets/CMU')
# #self.amass_file_paths = glob.glob('assets/CMU2/**/*poses*')
# print(mydata.amass_file_paths
#
# )

# mydata = AMASS('assets/CMU2')
# epochs = 20
# batch_size = 16
# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))
#
#     iterator = iter(mydata.dataset.batch(batch_size).shuffle(seed=42,buffer_size=5000))
#     # Iterate over the batches of the dataset.
#
#     next_exist = True
#     step = 0
#     total = 0
#     while next_exist:
#         optional = iterator.get_next_as_optional()
#         if not optional.has_value():
#             next_exist = False
#             if 0: #epoch == 0:
#                 print('son batchimi aldim', step)
#                 frame = 0  # bu kisim tek bir tane frame icin input verecek.
#                 garment_path = os.path.join(args.savedir, f"__hus_{epoch}_{frame:04d}_{args.garment}.obj")
#                 utils.save_obj(garment_path, logits[0], f_garment)
#
#                 # Eval body
#                 v_body, tensor_dict = body(
#                     shape=tf.reshape(betas[0, 0, :], [-1, 10]),
#                     pose=tf.reshape(poses[0, 0, :], [-1, 72]),
#                     translation=tf.reshape(trans[0, 0, :], [-1, 3]),
#                 )
#
#                 body_path = os.path.join(args.savedir, f"__hus__{epoch}_{frame:04d}_body.obj")
#                 utils.save_obj(body_path, v_body, body.faces)
#             break
#             ################################################################
#         total=total+batch_size
#
#         poses, trans, trans_vel,g = optional.get_value()
#
#         print(total,tf.reduce_sum(poses),tf.reduce_sum(g))
