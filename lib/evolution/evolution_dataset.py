import torch
from torch.utils.data import Dataset
from numpy import genfromtxt


class EvolutionGenerator(Dataset):
    def __init__(self, normalized_path, target_path):
        self.input_2d_joints = genfromtxt(normalized_path, delimiter=',')
        self.input_2d_joints = self.input_2d_joints.reshape((len(self.input_2d_joints), 17, 3))
        self.input_2d_joints = self.input_2d_joints[:, :, :-1]
        self.joints_3d = genfromtxt(target_path, delimiter=',')
        self.joints_3d = self.joints_3d.reshape((len(self.joints_3d), 17, 3))

    def __getitem__(self, index):
        joints_2d = self.input_2d_joints[index]
        joints_2d = torch.from_numpy(joints_2d).float()

        joints_3d = self.joints_3d[index]
        joints_3d = torch.from_numpy(joints_3d).float()

        return joints_2d, joints_3d

    def __len__(self):
        return len(self.input_2d_joints)


if __name__ == '__main__':
    ds = EvolutionGenerator(normalized_path='evolved_normalized2.csv', target_path='evolved2.csv')
    loader = torch.utils.data.DataLoader(ds)

    for i, data in enumerate(loader):
        j2d, j3d = data

    print('Done')