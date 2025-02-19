import torch
import h5py
import argparse
from video_trainer import Trainer
# from trainer import Trainer
from torch.utils.data import DataLoader
from Dataset.data import get_img_set,H5Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--fig',   type=str, default='train', help="chose train or test")
parser.add_argument('--upscale_factor', '-uf',  type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--n_colors', type=int, default=1,help='number of color channels to use')
# parser.add_argument('--model', '-m', type=str, default='edsr', help='choose which model is going to use')
# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='C',help='parameters config of RDN. (Use in RDN)')



args = parser.parse_args()



def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    #
    if args.fig == 'train':
        print('===> Train')
        path = "/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Fucongrui/video_dataset/vimeo_septuplet/train_vimeo_4_2.h5"
        f = h5py.File(path, 'r')
        In_train = f['Input']
        Ta_train = f['Target']

        train_set = H5Dataset(In_train, Ta_train)
        training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
        testing_data_loader = None

    if args.fig == 'test':
        print('===> Test')
        training_data_loader = None
        test_set = get_img_set("/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Fucongrui/video_dataset/vimeo_septuplet/sequences_4_2_mat_test/")
        testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = Trainer(args, training_data_loader,testing_data_loader)

    model.run(args)


if __name__ == '__main__':
        main()
