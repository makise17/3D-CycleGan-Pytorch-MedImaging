import matplotlib.pyplot as plt
from utils.NiftiDataset import *
from torch.utils.data import DataLoader
import utils.NiftiDataset as NiftiDataset
import argparse

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=r'/data/Data_folder/test/')
# parser.add_argument("--resample", action='store_true', default=False, help='Decide or not to resample the images to a new resolution')
# parser.add_argument("--new_resolution", type=float, default=(0.5, 0.5, 0.5), help='New resolution')
parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 64, 32], help="Input dimension for the generator")
parser.add_argument("--batch_size", type=int, nargs=1, default=5, help="Batch size to feed the network (currently supports 1)")
parser.add_argument("--drop_ratio", type=float, nargs=1, default=0, help="Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1")
parser.add_argument("--min_pixel", type=int, nargs=1, default=0.4, help="Percentage of minimum non-zero pixels in the cropped label")

args = parser.parse_args()

min_pixel = int(args.min_pixel*((args.patch_size[0]*args.patch_size[1]*args.patch_size[2])/100))

# choices = [0, 1, 2, 4]
# choices = [4]
trainTransforms = [
    # 比較
    NiftiDataset.Resize((128, 128, 32), True),
    NiftiDataset.Crop((128, 64, 32), args.drop_ratio, min_pixel),

    # デフォ
    # NiftiDataset.Augmentation(),
    # NiftiDataset.Padding((128, 128, 64)),
    # NiftiDataset.RandomCrop((128, 128, 64), args.drop_ratio, min_pixel),
 
    # NiftiDataset.Resample(args.new_resolution, args.resample),
    # NiftiDataset.Registration(),
    # NiftiDataset.Align(),
    # NiftiDataset.Augmentation(),
    # NiftiDataset.Padding((300, 300, 300)),
    # NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]),
                            # args.drop_ratio, min_pixel)
]

train_gen = NifitDataSet(args.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True)
print('lenght train list:',len(train_gen))
train_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=False)


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap= 'gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def plot3d(image):
    original=image
    original = np.rot90(original, k=-1)
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, original)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


# batch1 = train_loader.dataset[random.randint(0, len(train_gen) - 1)]
index = 0
batch1 = train_loader.dataset[index]


vol = batch1[0].numpy()
mask = batch1[1].numpy()
print(vol.shape)

vol = np.squeeze(vol, axis=0)
mask = np.squeeze(mask, axis=0)

plot3d(vol)
plt.savefig(f"./aug_samples/{index}_vol.png")
plot3d(mask)
plt.savefig(f"./aug_samples/{index}_mask.png")
# #dataの一つでも表示してやろう
# from monai.visualize import matshow3d
# import matplotlib.pyplot as plt
# fig = plt.figure()
# vol = np.transpose(vol, (2, 1, 0))
# mask = np.transpose(mask, (2, 1, 0))

# matshow3d(vol, fig=fig, cmap='gray')
# plt.gca().axis('off')
# # plt.subplots_adjust(left=0.05, right=0.05, bottom=0.05, top=0.05)
# plt.savefig(f"aug_samples/{index}_vol.png")

# fig = plt.figure()
# matshow3d(mask, fig=fig, cmap='gray')
# plt.gca().axis('off')
# # plt.subplots_adjust(left=0.05, right=0.05, bottom=0.05, top=0.05)
# plt.savefig(f"aug_samples/{index}_mask.png")