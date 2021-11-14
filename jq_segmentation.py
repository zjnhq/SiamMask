from collections import namedtuple

from net import *
from net.losses import StdLoss, YIQGNGCLoss, GradientLoss, ExtendedL1Loss, GrayLoss
from net.noise import get_noise, NoiseNet
from utils.image_io import *
from net.downsampler import *
from skimage.measure import compare_psnr
from cv2.ximgproc import guidedFilter



import numpy as np
from pdb import set_trace
import torch
import logging
from pdb import set_trace
from utils.benchmark_helper import load_dataset, dataset_zoo

SegmentationResult = namedtuple("SegmentationResult", ['mask', 'learned_mask', 'left', 'right', 'psnr'])
# data_type = torch.cuda.FloatTensor
data_type = torch.FloatTensor

class Segmentation(object):
    def __init__(self, image_name, image, plot_during_training=True,
                 first_step_iter_num=2000,
                 second_step_iter_num=4000,
                 bg_hint=None, fg_hint=None,
                 show_every=500,
                 downsampling_factor=0.1, downsampling_number=0,dcvx = False):

        self.dcvx_ = dcvx
        self.dcvx_loss  =0.0
        self.dcvx_coeff = 0.00005
        self.image_name = image_name
        if dcvx == True:
            self.image_name  = "dcvx_" + self.image_name+"_"+str(self.dcvx_coeff)
        self.image = image
        
        logging.basicConfig(filename=self.image_name+".log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
        self.logger  = logging.getLogger()
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.DEBUG) 


        self.image = image
        if bg_hint is None or fg_hint is None: 
            raise Exception("Hints must be provided")
        # self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.downsampling_factor = downsampling_factor
        self.downsampling_number = downsampling_number
        self.mask_net = None
        self.show_every = show_every
        self.bg_hint = bg_hint
        self.fg_hint = fg_hint
        self.left_net = None
        self.right_net = None
        self.images = None
        self.images_torch = None
        self.left_net_inputs = None
        self.right_net_inputs = None
        self.mask_net_inputs = None
        self.left_net_outputs = None
        self.right_net_outputs = None
        self.second_step_done = False
        self.mask_net_outputs = None
        self.parameters = None
        self.gngc_loss = None
        self.fixed_masks = None
        self.blur_function = None
        self.first_step_iter_num = first_step_iter_num
        self.second_step_iter_num = second_step_iter_num
        self.input_depth = 2
        self.multiscale_loss = None
        self.total_loss = None
        self.gngc = None
        self.blur = None
        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        left_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.left_net = left_net.type(data_type)

        right_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.right_net = right_net.type(data_type)

        mask_net = skip_mask(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            filter_size_down=3,
            filter_size_up=3,
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(data_type)

    def _init_images(self):
        self.images = get_imresize_downsampled(self.image, downsampling_factor=self.downsampling_factor,
                                               downsampling_number=self.downsampling_number)
        self.images_torch = [np_to_torch(image).type(data_type) for image in self.images]
        if self.bg_hint is not None:
            assert self.bg_hint.shape[1:] == self.image.shape[1:], (self.bg_hint.shape[1:], self.image.shape[1:])
            self.bg_hints = get_imresize_downsampled(self.bg_hint, downsampling_factor=self.downsampling_factor,
                                                     downsampling_number=self.downsampling_number)
            self.bg_hints_torch = [np_to_torch(bg_hint).type(data_type) for bg_hint in self.bg_hints]
        else:
            self.bg_hints = None
        if self.fg_hint is not None:
            assert self.fg_hint.shape[1:] == self.image.shape[1:]
            self.fg_hints = get_imresize_downsampled(self.fg_hint, downsampling_factor=self.downsampling_factor,
                                                     downsampling_number=self.downsampling_number)
            self.fg_hints_torch = [np_to_torch(fg_hint).type(data_type) for fg_hint in self.fg_hints]
        else:
            self.fg_hints = None

    def _init_noise(self):
        input_type = 'noise'
        # self.left_net_inputs = self.images_torch
        self.left_net_inputs = [get_noise(self.input_depth,
                                          input_type,
                                          (image.shape[2], image.shape[3])).type(data_type).detach()
                                for image in self.images_torch]
        self.right_net_inputs = self.left_net_inputs
        input_type = 'noise'
        self.mask_net_inputs = [get_noise(self.input_depth,
                                          input_type,
                                          (image.shape[2], image.shape[3])).type(data_type).detach()
                                for image in self.images_torch]

    def _init_parameters(self):
        self.parameters = [p for p in self.left_net.parameters()] + \
                          [p for p in self.right_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]

    def _init_dcvx_parameters(self):
        # parameters = [p for p in self.image_net.named_parameters()] + \
        #              [p for p in self.mask_net.named_parameters()]
        net_to_dcvx = [self.image_net,self.mask_net]
        if self._is_learning_ambient():
            net_to_dcvx.append(self.ambient_net)
        
        # set_trace()
        parameters = list()
        for net in net_to_dcvx:
            # for name, p  in net.named_parameters():
            #     print(name, end=' ')
            layer_id = 0
            # set_trace()
            for name, p  in net.named_parameters():
                layer_id +=1
                if "weight" in name and layer_id>7 and p.dim()>1:
                    parameters.append(p)
                    # print(p.dim())
        # for p in parameters:
        #     print(p.shape)
        # if self._is_learning_ambient():
        #     parameters += [p for p in self.ambient_net.named_parameters()]
        # set_trace()

        self.dcvx_parameters = parameters

    def dcvx(self):
        neg_weight_reg  = 0.0 
        parameters =self.dcvx_parameters
        for p in parameters:
            neg_weight_reg += torch.norm(p[p<0])
        return neg_weight_reg


    def _init_losses(self):
        # data_type = torch.cuda.FloatTensor
        self.gngc_loss = YIQGNGCLoss().type(data_type)
        self.l1_loss = nn.L1Loss().type(data_type)
        self.extended_l1_loss = ExtendedL1Loss().type(data_type)
        self.blur_function = StdLoss().type(data_type)
        self.gradient_loss = GradientLoss().type(data_type)
        self.gray_loss = GrayLoss().type(data_type)

    def _init_all(self):
        self._init_images()
        self._init_losses()
        self._init_nets()
        self._init_parameters()
        
        self._init_dcvx_parameters()

        self._init_noise()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # step 1
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.first_step_iter_num):
            optimizer.zero_grad()
            self._step1_optimization_closure(j)
            self._finalize_iteration()
            if self.plot_during_training:
                self._iteration_plot_closure(j)
            optimizer.step()
        self._update_result_closure()
        if self.plot_during_training:
            self._step_plot_closure(1)
        # self.finalize_first_step()
        # step 2
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.second_step_iter_num):
            optimizer.zero_grad()
            self._step2_optimization_closure(j)
            self._finalize_iteration()
            if self.second_step_done:
                break
            if self.plot_during_training:
                self._iteration_plot_closure(j)
            optimizer.step()
        self._update_result_closure()
        if self.plot_during_training:
            self._step_plot_closure(2)

        self.finalize()

    def finalize_first_step(self):
        left = torch_to_np(self.left_net_outputs[0])
        right = torch_to_np(self.right_net_outputs[0])
        save_image(self.image_name + "_1_left", left)
        save_image(self.image_name + "_1_right", right)
        save_image(self.image_name + "_hint1", self.bg_hint)
        save_image(self.image_name + "_hint2", self.fg_hint)
        save_image(self.image_name + "_hint1_masked", self.bg_hint * self.image)
        save_image(self.image_name + "_hint2_masked", self.fg_hint * self.image)

    def finalize(self):
        save_image(self.image_name + "_left", self.best_result.left)
        save_image(self.image_name + "_learned_mask", self.best_result.learned_mask)
        save_image(self.image_name + "_right", self.best_result.right)
        save_image(self.image_name + "_original", self.images[0])
        # save_image(self.image_name + "_fg_bg", ((self.fg_hint - self.bg_hint) + 1) / 2)
        save_image(self.image_name + "_mask", self.best_result.mask)

    def _update_result_closure(self):
        self._finalize_iteration()
        self._fix_mask()
        self.current_result = SegmentationResult(mask=self.fixed_masks[0],
                                                 left=torch_to_np(self.left_net_outputs[0]),
                                                 right=torch_to_np(self.right_net_outputs[0]),
                                                 learned_mask=torch_to_np(self.mask_net_outputs[0]),
                                                 psnr=self.current_psnr)
        if self.best_result is None or self.best_result.psnr <= self.current_result.psnr:
            self.best_result = self.current_result

    def _fix_mask(self):
        """
        fixing the masks using soft matting
        :return:
        """
        masks_np = [torch_to_np(mask) for mask in self.mask_net_outputs]
        new_mask_nps = [np.array([guidedFilter(image_np.transpose(1, 2, 0).astype(np.float32),
                                               mask_np[0].astype(np.float32), 50, 1e-4)])
                        for image_np, mask_np in zip(self.images, masks_np)]

        def to_bin(x):
            v = np.zeros_like(x)
            v[x > 0.5] = 1
            return v

        self.fixed_masks = [to_bin(m) for m in new_mask_nps]

    def _initialize_step1(self, iteration):
        self._initialize_any_step(iteration)

    def _initialize_step2(self, iteration):
        self._initialize_any_step(iteration)

    def _initialize_any_step(self, iteration):
        if iteration == self.second_step_iter_num - 1:
            reg_noise_std = 0
        elif iteration < 1000:
            reg_noise_std = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std = 1 / 1000.
        right_net_inputs = []
        left_net_inputs = []
        mask_net_inputs = []
        # creates left_net_inputs and right_net_inputs by adding small noise
        for left_net_original_input, right_net_original_input, mask_net_original_input \
                in zip(self.left_net_inputs, self.right_net_inputs, self.mask_net_inputs):
            left_net_inputs.append(
                left_net_original_input + (left_net_original_input.clone().normal_() * reg_noise_std))
            right_net_inputs.append(
                right_net_original_input + (right_net_original_input.clone().normal_() * reg_noise_std))
            mask_net_inputs.append(
                mask_net_original_input + (mask_net_original_input.clone().normal_() * reg_noise_std))
        # applies the nets
        self.left_net_outputs = [self.left_net(left_net_input) for left_net_input in left_net_inputs]
        self.right_net_outputs = [self.right_net(right_net_input) for right_net_input in right_net_inputs]
        self.mask_net_outputs = [self.mask_net(mask_net_input) for mask_net_input in mask_net_inputs]
        self.total_loss = 0
        self.gngc = 0
        self.blur = 0

    def _step1_optimization_closure(self, iteration):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        self._initialize_step1(iteration)
        if self.fg_hints is not None and self.bg_hints is not None:
            self._step1_optimize_with_hints(iteration)
        else:
            self._step1_optimize_without_hints(iteration)

    def _step2_optimization_closure(self, iteration):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        self._initialize_step2(iteration)
        if self.fg_hints is not None and self.bg_hints is not None:
            self._step2_optimize_with_hints(iteration)
        else:
            self._step2_optimize_without_hints(iteration)

        step= iteration
        if self.dcvx_ and step%5==0:self.dcvx_loss = self.dcvx() * self.dcvx_coeff# 0.00005

        if step%20:
            # print("blur_out:"+ str(self.blur_out.data.cpu()))
            loss_info = "total_loss:"+ str(self.total_loss.data.cpu())
            if self.dcvx_:loss_info+="dcvx_loss:"+ str(self.dcvx_loss.data.cpu())
            self.logger.debug(loss_info)
        
        # if step%5==0:
            # self.dcvx_loss = self.dcvx() * 0.0003
        if self.dcvx_: self.total_loss += self.dcvx_loss
        self.total_loss.backward(retain_graph=True)

    def _step1_optimize_without_hints(self, iteration):
        self.total_loss += sum(self.l1_loss(torch.ones_like(mask_net_output) / 2, mask_net_output) for
                               mask_net_output in self.mask_net_outputs)
        # self.total_loss.backward(retain_graph=True)

    def _step1_optimize_with_hints(self, iteration):
        """
        optimization, where hints are given
        :param iteration:
        :return:
        """
        self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                               left_net_output, fg_hint, image_torch
                               in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
        self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                               right_net_output, bg_hint, image_torch
                               in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))

        self.total_loss += sum(self.l1_loss(((fg_hint - bg_hint) + 1) / 2, mask_net_output) for
                               fg_hint, bg_hint, mask_net_output in
                               zip(self.fg_hints_torch, self.bg_hints_torch, self.mask_net_outputs))
        # self.total_loss.backward(retain_graph=True)

    def _step2_optimize_without_hints(self, iteration):
        for left_out, right_out, mask_out, original_image_torch in zip(self.left_net_outputs,
                                                                       self.right_net_outputs,
                                                                       self.mask_net_outputs,
                                                                       self.images_torch):
            self.total_loss += 0.5 * self.l1_loss(mask_out * left_out + (1 - mask_out) * right_out,
                                                  original_image_torch)
            self.current_gradient = self.gray_loss(mask_out)
            # self.current_gradient = self.gradient_loss(mask_out)
            self.total_loss += (0.01 * (iteration // 100)) * self.current_gradient
        self.total_loss.backward(retain_graph=True)

    def _step2_optimize_with_hints(self, iteration):
        if iteration <= 1000:
            self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                                   left_net_output, fg_hint, image_torch
                                   in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
            self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                                   right_net_output, bg_hint, image_torch
                                   in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))

        for left_out, right_out, mask_out, original_image_torch in zip(self.left_net_outputs,
                                                                       self.right_net_outputs,
                                                                       self.mask_net_outputs,
                                                                       self.images_torch):
            self.total_loss += 0.5 * self.l1_loss(mask_out * left_out + (1 - mask_out) * right_out,
                                                  original_image_torch)
            self.current_gradient = self.gray_loss(mask_out)
            # self.current_gradient = self.gradient_loss(mask_out)
            iteration = min(iteration, 1000)
            self.total_loss += (0.001 * (iteration // 100)) * self.current_gradient
        self.total_loss.backward(retain_graph=True)

    def _finalize_iteration(self):
        left_out_np = torch_to_np(self.left_net_outputs[0])
        right_out_np = torch_to_np(self.right_net_outputs[0])
        original_image = self.images[0]
        mask_out_np = torch_to_np(self.mask_net_outputs[0])
        self.current_psnr = compare_psnr(original_image, mask_out_np * left_out_np + (1 - mask_out_np) * right_out_np)
        # TODO: run only in the second step
        if self.current_psnr > 30:
            self.second_step_done = True

    def _iteration_plot_closure(self, iter_number):
        step =iter_number
        info = 'Iteration {:5d} total_loss {:5f} grad {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                                   self.current_gradient.item(),
                                                                                   self.current_psnr)
        if step%50==0:self.logger.info( info)
        if self.current_gradient is not None:
            print('Iteration {:5d} total_loss {:5f} grad {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                                   self.current_gradient.item(),
                                                                                   self.current_psnr),
                  '\r', end='')
        else:
            print('Iteration {:5d} total_loss {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                        self.current_psnr),
                  '\r', end='')
        if iter_number % self.show_every == self.show_every - 1:
            self._plot_with_name(iter_number)

    def _step_plot_closure(self, step_number):
        """
        runs at the end of each step
        :param step_number:
        :return:
        """
        self._plot_with_name("step_{}".format(step_number))

    def _plot_with_name(self, name):
        if self.fg_hint is not None and self.bg_hint is not None:
            plot_image_grid("left_right_hints_{}".format(name),
                            [np.clip(self.fg_hint, 0, 1),
                             np.clip(self.bg_hint, 0, 1)])
        for i, (left_out, right_out, mask_out, image) in enumerate(zip(self.left_net_outputs,
                                                                       self.right_net_outputs,
                                                                       self.mask_net_outputs, self.images)):
            plot_image_grid("left_right_{}_{}".format(name, i),
                            [np.clip(torch_to_np(left_out), 0, 1),
                             np.clip(torch_to_np(right_out), 0, 1)])
            mask_out_np = torch_to_np(mask_out)
            plot_image_grid("learned_mask_{}_{}".format(name, i),
                            [np.clip(mask_out_np, 0, 1), 1 - np.clip(mask_out_np, 0, 1)])

            plot_image_grid("learned_image_{}_{}".format(name, i),
                            [np.clip(mask_out_np * torch_to_np(left_out) + (1 - mask_out_np) * torch_to_np(right_out),
                                     0, 1), image])


# if __name__ == "__main__":
#     # the gt_ambient is taken from Bahat's code (https://github.com/YuvalBahat/Dehazing-Airlight-estimation)

#     import argparse

#     # dcvx_  =0
#     parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    
#     parser.add_argument('--dcvx_', type=int, default=0, metavar='dcvx_',
#                         help='whether to use dcvx')

#     args = parser.parse_args()
#     dcvx_ = args.dcvx_

#     # i = prepare_image("segimages/hongkong.png")
#     # dehaze("hongkong", i, use_deep_channel_prior=False, gt_ambient=np.array([0.5600084 , 0.64564645, 0.72515032]),dcvx_=dcvx_)
#     i = prepare_image("images/tower1.jpg")
#     seg = Segmentation("tower1", i, first_step_iter_num=500,second_step_iter_num=1000, dcvx=dcvx_)
#     # seg._init_all()
#     seg.optimize()



def track_vos(model, video, hp=None, mask_enable=False, refine_enable=False, mot_enable=False, device='cpu'):
    image_files = video['image_files']

    annos = [np.array(Image.open(x)) for x in video['anno_files']]
    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    if not mot_enable:
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)
    object_num = len(object_ids)
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))-1
    for obj_id, o_id in enumerate(object_ids):

        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            tic = cv2.getTickCount()
            set_trace()
            if f == start_frame:  # init
                mask = annos_init[obj_id] == o_id
                # fg_hint = im[mask]
                # fg_hint = im.copy()
                fg_hint = np.zeros(im.shape)
                fg_hint[mask] = im[mask]

                # dehaze("hongkong", i, use_deep_channel_prior=False, gt_ambient=np.array([0.5600084 , 0.64564645, 0.72515032]),dcvx_=dcvx_)
                # i = prepare_image("images/tower1.jpg")
                # filename = video['name'] + '_' + str(f + 1)
                seg = Segmentation(filename, im, first_step_iter_num=500,second_step_iter_num=1000, fg_hint=fg_hint, dcvx=dcvx_)
                # seg._init_all()
                seg.optimize()

            # elif end_frame >= f > start_frame:  # tracking
            #     state = siamese_track(state, im, mask_enable, refine_enable, device=device)  # track
            #     mask = state['mask']
            toc += cv2.getTickCount() - tic
            if end_frame >= f >= start_frame:
                pred_masks[obj_id, f, :, :] = mask
    toc /= cv2.getTickFrequency()

    if len(annos) == len(image_files):
        multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, annos,
                                            start=video['start_frame'] if 'start_frame' in video else None,
                                            end=video['end_frame'] if 'end_frame' in video else None)
        for i in range(object_num):
            for j, thr in enumerate(thrs):
                logger.info('Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1), thr,
                                                                           multi_mean_iou[i, j]))
    else:
        multi_mean_iou = []

    if args.save_mask:
        video_path = join('test_ddip', args.dataset, 'SiamMask', video['name'])
        if not isdir(video_path): makedirs(video_path)
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for i in range(pred_mask_final.shape[0]):
            cv2.imwrite(join(video_path, image_files[i].split('/')[-1].split('.')[0] + '.png'), pred_mask_final[i].astype(np.uint8))

    if args.visualization:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask = COLORS[pred_mask_final]
        for f, image_file in enumerate(image_files):
            output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f,:,:,:])).astype("uint8")
            cv2.imshow("mask", output)
            cv2.waitKey(1)

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f*len(object_ids) / toc))

    return multi_mean_iou, f*len(object_ids) / toc


def main():
    global args, logger, v_id
    args = parser.parse_args()
    # cfg = load_config(args)

    # set_trace()
    # init_log('global', logging.INFO)
    # if args.log != "":
    #     add_file_handler('global', args.log, logging.INFO)

    # logger = logging.getLogger('global')
    # logger.info(args)
    args.dataset = 'DAVIS2016'
    dataset = load_dataset(args.dataset)
    # VOS or VOT?
    if args.dataset in ['DAVIS2016', 'DAVIS2017', 'ytb_vos'] and args.mask:
        vos_enable = True  # enable Mask output
    else:
        vos_enable = False

    total_lost = 0  # VOT
    iou_lists = []  # VOS
    speed_list = []

    model= 0 
    set_trace()
    for v_id, video in enumerate(dataset.keys(), start=1):
        if args.video != '' and video != args.video:
            continue

        if vos_enable:
            iou_list, speed = track_vos(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                 args.mask, args.refine, args.dataset in ['DAVIS2017', 'ytb_vos'], device=device)
            iou_lists.append(iou_list)
        else:
            lost, speed = track_vot(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                             args.mask, args.refine, device=device)
            total_lost += lost
        speed_list.append(speed)

    # report final result
    if vos_enable:
        for thr, iou in zip(thrs, np.mean(np.concatenate(iou_lists), axis=0)):
            logger.info('Segmentation Threshold {:.2f} mIoU: {:.3f}'.format(thr, iou))
    else:
        logger.info('Total Lost: {:d}'.format(total_lost))

    logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


if __name__ == '__main__':
    main()
