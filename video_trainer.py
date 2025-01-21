import torch
import numpy as np
from math import log10,sqrt
import cv2 as cv
from traintestfunction import SR_Separate,SuperResolution_Model
from Enhance_inter import Enhance_Block
from video_upsample_42 import make_ARCNN_model_upsample,make_RDN_model_upsample
import torch.backends.cudnn as cudnn
import gc
import os
from scipy.io import loadmat,savemat
from sewar.full_ref import ssim,psnr
# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Trainer(object):
    def __init__(self, config, training_loader,testing_loader):
        super(Trainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model_1 = None
        self.model_2 = None
        self.model_3 = None
        self.model_4 = None
        self.model_5 = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None      #lr 下降方式的控制
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.fig = config.fig

    def build_model(self,args):
        self.model_1 = make_RDN_model_upsample(args,'SAI').to(self.device)
        self.model_2 = make_RDN_model_upsample(args,'EPI_1').to(self.device)
        self.model_3 = make_RDN_model_upsample(args,'EPI_2').to(self.device)
        self.model_4 = make_ARCNN_model_upsample().to(self.device)
        self.model_5 = Enhance_Block(args).to(self.device)

        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam([
            {'params': self.model_1.parameters()},
            {'params': self.model_2.parameters()},
            {'params': self.model_3.parameters()},
            # {'params': self.model_4.parameters()},
            # {'params': self.model_5.parameters()}
        ],
            # filter(lambda p: p.requires_grad, self.model_5.parameters()),
            lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 65, 80, 90], gamma=0.5) # lr decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.5)


    def save(self,epoch):
        # model_out_path = "Super_video_vimeoAll_4_2.pth"
        # model_out_path1 = "Enhanced_video_vimeoAll_4_2.pth"
        model_out_path2 = "Model123_video_vimeoAll_4_2.pth"
        state = {
                 'model_1': self.model_1.state_dict(),
                 'model_2': self.model_2.state_dict(),
                 'model_3': self.model_3.state_dict(),
                 # 'model_4': self.model_4.state_dict(),
                 # 'model_5': self.model_5.state_dict(),
                 'epoch' : epoch
                 }
        torch.save(state, model_out_path2)
        # torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path2))

    def train(self,epoch):

        # model_out_path = "Super_video_vimeoAll_4_2.pth"
        #
        # checkpoint = torch.load(model_out_path)
        #
        # self.model_1.load_state_dict(checkpoint['model_1'])
        # self.model_2.load_state_dict(checkpoint['model_2'])
        # self.model_3.load_state_dict(checkpoint['model_3'])
        # self.model_4.load_state_dict(checkpoint['model_4'])

        # self.model_5.train()
        self.model_1.train()
        self.model_2.train()
        self.model_3.train()

        Maxbatchnum = 3000

        f = open("Loss_Model123_video_vimeoAll_4_2.txt", "a")

        train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader):
            if batch_num >= Maxbatchnum:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()


            try:
                loss_1,loss_2,loss_3,loss_4 = SuperResolution_Model(self,data,target)
                # loss_5 = SuperResolution_Model(self, data, target)
                loss = loss_1+loss_2+loss_3+loss_4
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception


            # loss_1,loss_2,loss_3,loss_4 = SuperResolution_Model(self,data,target)
            # loss_1, loss_2, loss_3, loss_4, loss_5 = SuperResolution_Model(self, data, target)

            # loss = loss_1+loss_2+loss_3+loss_4
            # train_loss += loss.item()
            # loss.backward()
            # self.optimizer.step()

            if batch_num % 50 == 0:
                # print(" Epoch:{:d}\tbatch_num:{:d}\tLoss: {:.4f}\t".format(epoch, batch_num, loss.item()))
                print(" Epoch:{:d}\tbatch_num:{:d}\tLoss: {:.4f}\t".format(epoch, batch_num, train_loss / (batch_num + 1)))

        # print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        print("=========Epoch:{:d}\t Average Loss: {:.4f}".format(epoch,train_loss/ Maxbatchnum))
        f.write("Epoch:{:d},  Average Loss: {:.4f}\n".format(epoch,train_loss / Maxbatchnum))
        f.close()

    # def test_img(self):
    #     f = open("PSNR_SSIM_video_EPI_SR_detail.txt", "a")
    #     avg_psnr = 0
    #     avg_ssim = 0
    #     a = 8
    #     with torch.no_grad():
    #         for batch_num, (data, target) in enumerate(self.testing_loader):
    #             print("####################{:d}############".format(batch_num))
    #             data, target = data.to(self.device), target.to(self.device)
    #
    #             savepath = "./Save_video_EPI_SR/"+str(batch_num)+"/"
    #             isExists = os.path.exists(savepath)
    #             if not isExists:
    #                os.makedirs(savepath)
    #             #
    #             # img = data[0, :, :, 10]*255
    #             # cv.imwrite("data.png", img.cpu().numpy())
    #
    #             prediction = SR_Separate(self,data)
    #
    #             psnr_score = np.zeros(a)
    #             curSSIM = np.zeros(a)
    #
    #
    #             for i in range(a):
    #
    #                 # path = savepath + str(i)+".png"
    #                 # img = prediction[0, :, :, i]*255
    #                 # cv.imwrite(path, img.cpu().numpy())   # save to .png
    #
    #
    #                 img1 = prediction[0, :, :, i]
    #                 img2 = target[0, :, :, i]
    #                 # psnr_score[i] = Cp_PSNR(prediction[0, :, :, i], target[0, :, :, i])
    #
    #                 psnr_score[i] = psnr(img2.cpu().numpy(),img1.cpu().numpy(),MAX=1)
    #                 curSSIM[i] = ssim(img2.cpu().numpy(),img1.cpu().numpy(),MAX=1)[0]
    #
    #                 # scio.savemat('./1.mat', {'out': prediction[0, :, :, i]})
    #                 print("psnr:{:.4f}".format(psnr_score[i]))
    #                 print("ssim:{:.4f}".format(curSSIM[i]))
    #             PSNR = np.mean(psnr_score)
    #             print(" each one_Img PSNR: {:.4f}".format(PSNR))
    #             f.write("PSNR: {:.4f}\n".format(PSNR))
    #             avg_psnr += PSNR
    #
    #             SSIM = np.mean(curSSIM)
    #             print(" each one_Img SSIM: {:.4f}".format(SSIM))
    #             f.write("SSIM: {:.4f}\n".format(SSIM))
    #             avg_ssim += SSIM
    #
    #             del data, target,prediction
    #             gc.collect()
    #
    #         print("=========Average Psnr: {:.4f}".format(avg_psnr / len(self.testing_loader)))
    #         f.write("=========Average Psnr: {:.4f}\n".format(avg_psnr / len(self.testing_loader)))
    #
    #         print("=========Average SSIM: {:.4f}".format(avg_ssim / len(self.testing_loader)))
    #         f.write("=========Average SSIM: {:.4f}\n".format(avg_ssim / len(self.testing_loader)))
    #     f.close()

    def video_test(self):
            f = open("Large_S_delAR_PSNR_SSIM_video_vimeo_detail_3to5.txt", "a")
            f1 = open("Large_S_delAR_PSNR_SSIM_vimeo_SR_3to5.txt", "a")
            f2 = open("Large_S_delAR_PSNR_SSIM_vimeo_VFI_3to5.txt", "a")

            avg_psnr = 0
            avg_ssim = 0
            avg_SR_psnr = 0
            avg_SR_ssim = 0
            avg_VFI_psnr = 0
            avg_VFI_ssim = 0
            # a = 6
            a = 5
            with torch.no_grad():
                for batch_num, (data, target,Imgname) in enumerate(self.testing_loader):
                    print("####################{:d}############".format(batch_num))
                    data, target = data.to(self.device), target.to(self.device)

                    file = Imgname[0][-14:-4] + "\t"
                    print(file)
                    savepath = "./Large_S_delAR_Save_video_vimeo_3to5/" + file + "/"
                    isExists = os.path.exists(savepath)
                    if not isExists:
                        os.makedirs(savepath)

                    prediction = SR_Separate(self, data)

                    # 保存测试结果为.mat
                    # savepathmat = "./Save_video_vimeo_mat/"
                    # isExists = os.path.exists(savepathmat)
                    # if not isExists:
                    #     os.makedirs(savepathmat)
                    # filemat = Imgname[0][-14:-4]
                    # dataNew = savepathmat + filemat + ".mat"
                    # savemat(dataNew, {'Input': prediction.cpu().numpy(),'Target': target.cpu().numpy()})

                    psnr_score = np.zeros(a)
                    curSSIM = np.zeros(a)
                    SR_psnr = 0
                    SR_ssim = 0
                    VFI_psnr = 0
                    VFI_ssim = 0
                    number = 3

                    for i in range(a):

                        # 保存测试结果为图片
                        path = savepath + str(i)+".png"
                        img = prediction[0, :, :, i]*255
                        cv.imwrite(path, img.cpu().numpy())   # save to .png

                        img1 = prediction[0, :, :, i]
                        img2 = target[0, :, :, i]
                        # psnr_score[i] = Cp_PSNR(prediction[0, :, :, i], target[0, :, :, i])

                        psnr_score[i] = psnr(img2.cpu().numpy(), img1.cpu().numpy(), MAX=1)
                        curSSIM[i] = ssim(img2.cpu().numpy(), img1.cpu().numpy(), MAX=1)[0]

                        print("psnr:{:.4f}".format(psnr_score[i]))
                        print("ssim:{:.4f}".format(curSSIM[i]))

                        if i%2==0:
                            SR_psnr = SR_psnr +  psnr_score[i]
                            SR_ssim = SR_ssim +  curSSIM[i]
                        else:
                            VFI_psnr = VFI_psnr +psnr_score[i]
                            VFI_ssim = VFI_ssim + curSSIM[i]

                    PSNR_SR = SR_psnr/3
                    PSNR_VFI = VFI_psnr / 2
                    # PSNR_VFI = VFI_psnr/3
                    PSNR = np.mean(psnr_score)
                    print(" each one_Img PSNR: {:.4f}".format(PSNR))
                    print(" SR PSNR: {:.4f}".format(PSNR_SR))
                    print(" VFI PSNR: {:.4f}".format(PSNR_VFI))
                    f.write(file)
                    f.write("PSNR: {:.4f}\t".format(PSNR))
                    f1.write(file)
                    f1.write("SR PSNR: {:.4f}\t".format(PSNR_SR))
                    f2.write(file)
                    f2.write("VFI PSNR: {:.4f}\t".format(PSNR_VFI))
                    avg_psnr += PSNR
                    avg_SR_psnr +=PSNR_SR
                    avg_VFI_psnr +=PSNR_VFI

                    SSIM = np.mean(curSSIM)
                    SSIM_SR = SR_ssim / 3
                    SSIM_VFI = VFI_ssim / 2
                    # SSIM_VFI = VFI_ssim / 3
                    print(" each one_Img SSIM: {:.4f}".format(SSIM))
                    print(" SR SSIM: {:.4f}".format(SSIM_SR))
                    print(" VFI SSIM: {:.4f}".format(SSIM_VFI))
                    # f.write(file)
                    f.write("SSIM: {:.4f}\n".format(SSIM))
                    # f1.write(file)
                    f1.write("SR SSIM: {:.4f}\n".format(SSIM_SR))
                    # f2.write(file)
                    f2.write("VFI SSIM: {:.4f}\n".format(SSIM_VFI))
                    avg_ssim += SSIM
                    avg_SR_ssim +=SSIM_SR
                    avg_VFI_ssim +=SSIM_VFI

                    del data, target, prediction
                    gc.collect()

                print("=========Average Psnr: {:.4f}".format(avg_psnr / len(self.testing_loader)))
                print("=========Average SR_Psnr: {:.4f}".format(avg_SR_psnr / len(self.testing_loader)))
                print("=========Average VFI_Psnr: {:.4f}".format(avg_VFI_psnr / len(self.testing_loader)))
                f.write("=========Average Psnr: {:.4f}\n".format(avg_psnr / len(self.testing_loader)))
                f1.write("=========Average SR_Psnr: {:.4f}\n".format(avg_SR_psnr / len(self.testing_loader)))
                f2.write("=========Average VFI_Psnr: {:.4f}\n".format(avg_VFI_psnr / len(self.testing_loader)))

                print("=========Average SSIM: {:.4f}".format(avg_ssim / len(self.testing_loader)))
                print("=========Average SR_SSIM: {:.4f}".format(avg_SR_ssim / len(self.testing_loader)))
                print("=========Average VFI_SSIM: {:.4f}".format(avg_VFI_ssim / len(self.testing_loader)))
                f.write("=========Average SSIM: {:.4f}\n".format(avg_ssim / len(self.testing_loader)))
                f1.write("=========Average SR_SSIM: {:.4f}\n".format(avg_SR_ssim / len(self.testing_loader)))
                f2.write("=========Average VFI_SSIM: {:.4f}\n".format(avg_VFI_ssim / len(self.testing_loader)))
            f.close()
            f1.close()
            f2.close()

    def run(self,args):

        if self.fig =='train':
            self.build_model(args)

            # model_out_path = "video_vimeo_4_2.pth"
            # checkpoint = torch.load(model_out_path)
            # self.model_1.load_state_dict(checkpoint['model_1'])
            # self.model_2.load_state_dict(checkpoint['model_2'])
            # self.model_3.load_state_dict(checkpoint['model_3'])
            # self.model_4.load_state_dict(checkpoint['model_4'])
            # start_epoch = checkpoint['epoch'] + 1
            start_epoch =1
            for epoch in range(start_epoch, self.nEpochs + 1):
                print("\n===> Epoch {} starts:".format(epoch))

                self.train(epoch)
                self.scheduler.step(epoch)
                if epoch % 1 == 0:
                    self.save(epoch)
        if self.fig == 'test':
            self.build_model(args)

            self.video_test()

# def Cp_PSNR(img1,img2):
#
#
#     diff = img1 - img2
#
#     # plt.figure(3)
#     # plt.imshow(diff.cpu(), cmap='gray')
#
#     diff = diff.flatten()
#     rmse = sqrt(torch.mean(diff ** 2.))
#     return 20 * log10(1.0 / rmse)
#
#     # mse = np.mean((img1  - img2 ) ** 2)
#     # if mse < 1.0e-10:
#     #     return 100
#     # PIXEL_MAX = 1
#     # return 20 * log10(PIXEL_MAX / sqrt(mse))
