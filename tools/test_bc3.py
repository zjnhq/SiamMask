# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile
from cv2.ximgproc import jointBilateralFilter
import sys
syspath_windows="E:/code/tracking/SiamMask/SiamMask" 
syspath_windows2="E:/code/tracking/SiamMask/SiamMask/experiments/siammask_sharp"
if syspath_windows not in sys.path:
    sys.path.append(syspath_windows)
if syspath_windows2 not in sys.path:
    sys.path.append(syspath_windows2)

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str

from torch.nn.functional import interpolate
from pdb import set_trace


import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.util import img_as_float
global_visualize = 0 
diffuse = 1
# from experiments.siammask_sharp.resnet import resnet18_allfeat, resnet18

def superpixel_seg(img, mask = 0,visualize_ =0):
    # set_trace()
    # exm_img= astronaut()
    img = img_as_float(img)

    # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    # segments_slic = slic(img, n_segments=5, compactness=10, sigma=1)
    superpix_slic = slic(img, n_segments=150, compactness=20, sigma=1)
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # gradient = sobel(rgb2gray(img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    # print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    # print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
    # print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
    # set_trace()
    visualize_ =0& global_visualize
    if visualize_:
        fig, ax = plt.subplots(2, 3, figsize=(20, 20), sharex=True, sharey=True)
        ax[0, 0].imshow(mark_boundaries(img, segments_fz))
        ax[0, 0].set_title("Felzenszwalbs's method")
        ax[0, 1].imshow(mark_boundaries(img, segments_slic))
        ax[0, 1].set_title('SLIC')
        ax[1, 0].imshow(mark_boundaries(img, segments_quick))
        ax[1, 0].set_title('Quickshift')
        ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
        ax[1, 1].set_title('Compact watershed')
        if mask !=0:
            ax[1, 2].imshow(mark_boundaries(mask, segments_slic))
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.show()
        # plt.savefig('first_frame.pdf')  
    return superpix_slic

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC

from skimage.segmentation import active_contour
def train_rf(pixels,labels):
    # clf = RandomForestRegressor(max_depth=6, n_estimators=100, random_state=0,n_jobs=-1)
    clf = GradientBoostingRegressor(max_depth=6, n_estimators=4, random_state=0,n_iter_no_change=1)
    # clf = (n_iter_no_change
    # set_trace()
    clf.fit(pixels, labels)
    # clf.fit(pixels[:10000], pixels[:10000])
    return clf



thrs = np.arange(0.3, 0.5, 0.05)

parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='DAVIS2017', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--jq', action='store_true', help='whether use semiparametric')

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float().cuda()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch

def patch_no_padding(im, pos, model_sz, original_sz, avg_chans):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = int(max(round(pos[0] - c),0))
    context_xmax = int(min(context_xmin + sz - 1,im_sz[1]-1))
    context_ymin = int(max(round(pos[1] - c),0))
    context_ymax = int(min(context_ymin + sz - 1,im_sz[0]-1))
    # left_pad = int(max(0., -context_xmin))
    # top_pad = int(max(0., -context_ymin))
    # right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    # bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    # context_xmin = context_xmin + left_pad
    # context_xmax = context_xmax + left_pad
    # context_ymin = context_ymin + top_pad
    # context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
   
    # im_patch_copy = im.copy()
    im_patch = im[int(context_ymin):int(context_ymax ), int(context_xmin):int(context_xmax )] 

    return im_patch



def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu', semip =False):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    # set_trace()

    p.renew()

    net = model
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = model.anchor_num
    p.anchor = generate_anchor(model.anchors, p.score_size)
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    template_img = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans,out_mode='img')
    # neg_img = get_negative_pixels(im, target_pos, p.exemplar_size, s_z, avg_chans)

    visualize = 0

    z = Variable(z_crop.unsqueeze(0))
    net.template(z.to(device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['semip'] = True
    state['semip_time'] = 0

    return state

def siamese_track(state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False, visualize_=0,save_name='exp'):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    semip = state['semip']
    # fig2, ax2 = state['fig2'],state['ax2']
    # fig, ax = state['fig'],state['ax']
    if semip:
        rf_model = state['rf_model']
        lr_model = state['lr_model']
        sv_model = state['sv_model']


    labp1 = 0
    labp2 = 0
    labp = 0

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        cv2.imshow('search area', im_debug)
        cv2.waitKey(0)


    # extract scaled crops for search region x at previous target position
    
    # original code: extract the original patch and put it as torch variable
    # x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    x_crop =get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0)
    temp_im =patch_no_padding(im, target_pos, p.instance_size, round(s_x), avg_chans)
    # set_trace()
    # superpixel_seg(temp_im)

    x_crop = Variable(x_crop)

    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.to(device))
    else:
        score, delta = net.track(x_crop.to(device))
    visualize= 1
    
    if semip and mask_enable:
        tic_semip = cv2.getTickCount()
        llfeat = net.features2(im_to_torch(im).unsqueeze(0))#.squeeze(0)
        nchannel = llfeat.shape[1]
        sz1,sz2 = im.shape[0],im.shape[1]
        if llfeat.shape[2]==sz1 and llfeat.shape[3]==sz2:
            llfeat_resz = llfeat
        else:
            llfeat_resz = interpolate(llfeat, size=[sz1,sz2],  mode='cubic', align_corners=None).detach()
        llfeat_resz = llfeat_resz.squeeze(0).cpu().numpy()
        pixels = llfeat_resz.reshape(nchannel,-1).T

        # labp = rf_model.predict(pixels)#>0.2
        # labp = rf_model.predict(pixels)+ lr_model.predict(pixels)*0.3 
        labp1 = rf_model.predict_proba(pixels)[:,1]
        # labp2 = lr_model.predict(pixels)#>0.2
        use_svm =0
        if use_svm:
            labp2 =lr_model.predict_proba(pixels)[:,1]
            labp_concat = np.concatenate((labp1[:,np.newaxis],labp2[:,np.newaxis]),axis=1)
            sv_model.fit(labp_concat,mask.reshape(-1))
            labp = np.maximum(sv_model.decision_function(labp_concat),-0.1)+0.1
        else:

            labp2 =lr_model.predict_proba(pixels)[:,1]
            labp = labp1*0.95 + labp2*0.05

            # labp = np.maximum(sv_model.decision_function(labp_concat),-0.1)+0.1
        labp = labp.reshape(sz1,sz2)

        toc_semip = cv2.getTickCount() -tic_semip
        visualize=0& global_visualize
        # visualize = 1
        if visualize:
            fig, ax = plt.subplots(2, 2, figsize=(20, 20), sharex=True, sharey=True)
            ax[0,0].imshow(im)
            ax[0,0].set_title("image")
            ax[0,1].imshow(labp1.reshape(sz1,sz2))
            ax[0,1].set_title('forest estimation')
            ax[1,0].imshow(labp2.reshape(sz1,sz2))
            ax[1,0].set_title('linear estimation')
            ax[1,1].imshow(labp)
            ax[1,1].set_title('semiparametric estimation')
            # save_name = video['name']+str(obj_id)+'_'+str(o_id)
            plt.savefig(save_name+"_semip2.jpg")
            # plt.savefig(save_name+"_diffuse_mask.jpg")
            # plt.show()
            # plt.waitforbuttonpress()

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_sz*scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    # for Mask Branch
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if refine_enable:
            mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))
        # set_trace()
        visualize = 1 & global_visualize
        # set_trace()
        state['mask_nn'] = mask_in_img.copy()
        if semip:
            dsize = (mask_in_img.shape[1],mask_in_img.shape[0])
            labp =cv2.resize(labp, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            target_ind = np.where(mask_in_img < (p.seg_thr-0.12))
            effective_labp = labp.copy()
            effective_labp[target_ind]= 0.0
            state['mask_semip'] = effective_labp
        if visualize and semip and 0:
            fig, ax = plt.subplots(2, 3, figsize=(20, 20))
            
            ax[0,0].imshow(effective_labp)
            ax[0,0].set_title("effective labp")
            ax[0,1].imshow(mask_in_img)
            ax[0,1].set_title('mask_in_img')
            # cv2.resize(labp,mask.shape)
            # dsize = (mask.shape[0],mask.shape[1])
            # labp_rsz = cv2.resize(labp, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            # ax[1,0].imshow(labp_rsz+ mask)
            # ax[1,0].set_title("labp_rsz+ mask")
            dsize = (mask_in_img.shape[1],mask_in_img.shape[0])
            labp_rsz = cv2.resize(effective_labp, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            ax[1,1].imshow(labp_rsz+ mask_in_img)
            ax[1,1].set_title("labp_rsz+ mask_in_img")
            # ax[1].imshow(mask_in_img)
            # ax[1].set_title('mask_in_img')
            ax[1,2].imshow(im)
            ax[1,2].set_title('im')
            # plt.show()
            # plt.waitforbuttonpress()
        if semip:
            dsize = (mask_in_img.shape[1],mask_in_img.shape[0])
            labp =cv2.resize(labp, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            target_ind = np.where(mask_in_img > (p.seg_thr-0.15))
            # print("pos mean:"+ str(mask_in_img[target_ind].mean()))
            # target_ind2 = np.where(mask_in_img < (p.seg_thr-0.15))
            # print("neg mean:"+ str(mask_in_img[target_ind2].mean()))
            mask_in_img[target_ind] = mask_in_img[target_ind]*0.5 + labp[target_ind]*0.5
        if visualize and semip and 0:
            ax[1,0].imshow(mask_in_img)
            ax[1,0].set_title('final mask')
            plt.show()

        # else:
        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        # diffuse = 1
        if diffuse:
            # set_trace()
            # temp_mask = patch_no_padding(target_mask)
            # temp_mask = patch_no_padding(mask_in_img, target_pos, p.instance_size, round(s_x), avg_chans)
            # temp_im = patch_no_padding(im, target_pos, p.instance_size, round(s_x), avg_chans)
            # segments = superpixel_seg(temp_im,visualize_=0)
            # boundary_map = find_boundaries(segments, connectivity=1, mode='subpix')
            # import pickle as pkl
            # pkl.dump((temp_mask,temp_im),open('mask_im.pkl','wb'))
            temp_mask= mask_in_img.copy()
            temp_im = im.copy()
            indx = np.argwhere(temp_mask>p.seg_thr )
            seg_thr = p.seg_thr
            while indx.shape[0]==0:
                seg_thr = seg_thr -0.05
                indx = np.argwhere(temp_mask> seg_thr )
            sz_y,sz_x = temp_mask.shape[0],temp_mask.shape[1]
            roi_ymin, roi_xmin = indx[:,0].min(),indx[:,1].min()
            roi_ymax, roi_xmax = indx[:,0].max(),indx[:,1].max()
            ext_sz = 20
            roi_ymax, roi_xmax = min(roi_ymax+ext_sz,sz_y-1), min(roi_xmax+ext_sz,sz_x-1)
            roi_ymin, roi_xmin = max(roi_ymin- ext_sz,0), max(roi_xmin- ext_sz,0)
            temp_mask = temp_mask[roi_ymin:roi_ymax,roi_xmin:roi_xmax]
            temp_im = temp_im[roi_ymin:roi_ymax,roi_xmin:roi_xmax]
            color_base= 150
            # mask = (temp_mask*color_base).astype(np.uint8)
            superpix = superpixel_seg(temp_im,visualize_=0)
            # from skimage.filters import gaussian
            # boundary_map = find_boundaries(superpix, connectivity=8, mode='thick', background=0)
            target_mask=(temp_mask>seg_thr)
            seg_bd = find_boundaries(target_mask,connectivity=1, mode='thick')
            # seg_bd = find_boundaries(target_mask,connectivity=1, mode='subpix', background=0)
            seg_ids = np.unique(superpix[seg_bd])
            mask_rf = temp_mask.copy()

            # mask_rf = jointBilateralFilter( boundary_map,mask_rf, d=5, sigmaColor=250, sigmaSpace=20)

            for i in range(seg_ids.shape[0]):
                seg_id = seg_ids[i]
                region = superpix==seg_id
                mask_rf[region] = temp_mask[region].mean()*0.4+mask_rf[region]*0.6
                # mask_rf[region] = temp_mask[region].mean()
            target_mask2 = mask_rf>p.seg_thr
            seg_bd2_rf = find_boundaries(target_mask2,connectivity=1, mode='subpix')
            mask_in_img[roi_ymin:roi_ymax,roi_xmin:roi_xmax]=mask_rf-0.05
            state['mask_semip'] = mask_in_img
            vis_diffuse=visualize_ & global_visualize
            # vis_diffuse=1
            if vis_diffuse:
                fig, ax = plt.subplots(1, 5, figsize=(20, 20))
                num_fig = 4
                # row_id2 = obj_id%num_fig
                ax[0].imshow(mark_boundaries(temp_im, superpix))
                ax[0].set_title("Superpixel Boundary")
                ax[1].imshow(mark_boundaries(temp_im, seg_bd))
                ax[1].set_title('Deep Net Prediction' )
                # ax[1].imshow(mark_boundaries(mask, superpix))
                # # ax[1].set_title('soft deep mask' )
                # ax[2].imshow(mark_boundaries((temp_mask*color_base).astype(np.uint8), superpix))
                # ax[2].set_title('Deep Net Confidence Map')
                # ax[1].imshow(mark_boundaries(mask_rf, superpix))
                # ax[1].set_title('rf soft mask')
                ax[3].imshow(mark_boundaries(temp_im, seg_bd2_rf))
                ax[3].set_title('Semiparametric Prediction')
                ax[4].imshow(mark_boundaries( (mask_in_img*color_base).astype(np.uint8), superpix))
                ax[4].set_title('Semiparametric Confidence Map')
                # ax[0, 2].imshow(mark_boundaries(im, state['mask_nn']>p.seg_thr))
                # ax[0, 2].set_title('nn mask')
                # ax[5].imshow(mark_boundaries(im, mask_in_img>p.seg_thr))
                # ax[5].set_title('diffuse mask')
                # if img_neg !=0:
                #     ax[1, 2].imshow(img_neg)
                # for a in ax.ravel():
                #     a.set_axis_off()
                plt.tight_layout()
                plt.savefig(save_name+"_diffuse_mask.jpg")
                # # fig_counter1 +=1
                # if obj_id%num_fig==0:
                #     plt.savefig(save_name+"_diffuse_mask.jpg")
                #     fig, ax = plt.subplots(num_fig, 6, figsize=(20, 20))
                #     state['fig'],state['ax'] = fig, ax
                plt.show()


    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))



    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = state['mask_nn'] if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []

    if semip: state['semip_time'] = state['semip_time'] + toc_semip
    return state

def semip_init(model,im,mask, name = 'default'):
    tic_1 = cv2.getTickCount()
    llfeat = model.features2(im_to_torch(im).unsqueeze(0))#.squeeze(0)
    toc_1 = cv2.getTickCount()
    print("feature forward:"+str((toc_1-tic_1)/cv2.getTickFrequency()))

    # rf_model = GradientBoostingRegressor(max_depth=6, n_estimators=4, random_state=0,n_iter_no_change=1)
    # lr_model = LogisticRegression(C=1.0, random_state=0, solver='lbfgs', max_iter=100, n_jobs=-1)
    # rf_model = GradientBoostingClassifier(max_depth=20, n_estimators=20, random_state=0,n_iter_no_change=1)
    # rf_model = RandomForestRegressor(max_depth=6, n_estimators=10,  n_jobs=-1)
    rf_model = RandomForestClassifier(max_depth=20, n_estimators=20,  n_jobs=-1)
    lr_model = LogisticRegression(C=1.0, random_state=0, solver='lbfgs', max_iter=40, n_jobs=-1)
    # sv_model = SVC(C=1.0, kernel='rbf')
    sv_model = LinearSVC(C=1.0, max_iter=20)
    # rf_model = RandomForestRegressor(max_depth=6, n_estimators=100, random_state=0,n_jobs=-1)
    def deeprf_train(llfeat,mask, rf_model,lr_model):
        nchannel = llfeat.shape[1]
        (sz1,sz2) = mask.shape
        sz1f,sz2f = llfeat.shape[2],llfeat.shape[3]
        if llfeat.shape[2]==sz1 and llfeat.shape[3]==sz2:
            llfeat_resz = llfeat
        else:
            llfeat_resz = interpolate(llfeat, size=[sz1,sz2],  mode='cubic', align_corners=None).detach()
        llfeat_resz = llfeat_resz.squeeze(0).cpu().numpy()
        pixels = llfeat_resz.reshape(nchannel,-1).T
        lab = mask.reshape(-1).astype(float)
        pos_indx= lab>0
        num_pos_pixels= pos_indx.sum()
        ave_indx = np.arange(0, lab.size)[::10]
        x=np.concatenate((pixels[pos_indx],pixels[ave_indx]),axis=0)
        lab = np.concatenate((lab[pos_indx],lab[ave_indx]),axis=0)
        tic_1 = cv2.getTickCount()
        rf_model.fit(x,lab)
        toc_1 = cv2.getTickCount()
        print("rf train:"+str((toc_1-tic_1)/cv2.getTickFrequency()))
        print("x shape:"+str(x.shape))
        # tic_1 = cv2.getTickCount()
        # lr_model.fit(x,lab)
        # toc_1 = cv2.getTickCount()
        # print("lr train:"+str((toc_1-tic_1)/cv2.getTickFrequency()))
        # labp1 = rf_model.predict_proba(pixels)[:,1]
        # # labp2 = lr_model.predict(pixels)*0.3 #>0.2
        # use_svm =0
        # if use_svm:
        #     labp2 =lr_model.predict_proba(pixels)
        #     labp_concat = np.concatenate((labp1[:,np.newaxis],labp2[:,np.newaxis]),axis=1)
        #     sv_model.fit(labp_concat,mask.reshape(-1))
        #     labp = np.maximum(sv_model.decision_function(labp_concat),-0.1)+0.1
        #     labp = labp.reshape(sz1,sz2)
        # else:

        #     labp2 =lr_model.predict_proba(pixels)[:,1]
        #     labp = labp1*0.8 + labp2*0.2
        #     # labp = np.maximum(sv_model.decision_function(labp_concat),-0.1)+0.1
        #     labp = labp.reshape(sz1,sz2)
        labp = rf_model.predict(pixels).reshape(sz1,sz2)
        set_trace()
        visualize =1& global_visualize
        visualize =1
        if visualize:
            fig, ax2 = plt.subplots(2, 2, figsize=(40, 20), sharex=True, sharey=True)
            # row_id2 = obj_id%num_fig
            ax2[0,0].imshow(im)
            ax2[0,0].set_title("Image")
            # ax2[0,1].imshow((labp1.reshape(sz1,sz2)*100).astype(np.uint8))
            # ax2[0,1].set_title('Random Forest Prediction')
            # ax[1,0].imshow( (labp2.reshape(sz1,sz2)*100).astype(np.uint8))
            # ax[1,0].set_title('Linear Model Prediction')
            ax2[1,0].imshow(labp)
            ax2[1,0].set_title('Semiparametric Prediction')
            ax2[1,1].imshow(mask)
            ax2[1,1].set_title('Mask Ground Truth')
            plt.tight_layout()
            # save_name = video['name']+str(obj_id)+'_'+str(o_id)
            # plt.savefig(save_name+"_semip.jpg")
            # fig_counter2+=1
            # if obj_id %num_fig ==0:
            #     # fig_counter2 =0
            #     save_name = video['name']+str(obj_id)+'_'+str(o_id)
            #     plt.savefig(save_name+"_semip.jpg")
            #     fig2, ax2 = plt.subplots(num_fig, 4, figsize=(20, 20))
            #     state['fig2'],state['ax2'] = fig2, ax2 
            plt.show()
    deeprf_train(llfeat,mask,rf_model, lr_model)
    return rf_model, lr_model


def track_vot(model, video, hp=None, mask_enable=False, refine_enable=False, device='cpu'):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    set_trace()
    image_files, gt = video['image_files'], video['gt']

    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, model, hp, device)  # init tracker

            if semip:
                mask= np.zeros(im.shape,dtype= np.uint8)

                rf_model, lr_model= semip_init(model,im,mask)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            state = siamese_track(state, im, mask_enable, refine_enable, device, args.debug)  # track
            if mask_enable:
                location = state['ploygon'].flatten()
                mask = state['mask']

            else:
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                mask = []

            if 'VOT' in args.dataset:
                gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                              (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
                if mask_enable:
                    pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                    (location[4], location[5]), (location[6], location[7]))
                else:
                    pred_polygon = ((location[0], location[1]),
                                    (location[0] + location[2], location[1]),
                                    (location[0] + location[2], location[1] + location[3]),
                                    (location[0], location[1] + location[3]))
                b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
            else:
                b_overlap = 1

            if b_overlap:
                regions.append(location)
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                if mask_enable:
                    mask = mask > state['p'].seg_thr
                    im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im_show, str(state['score']) if 'score' in state else '', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()


    # save result
    name = args.arch.split('.')[0] + '_' + ('mask_' if mask_enable else '') + ('refine_' if refine_enable else '') +\
           args.resume.split('/')[-1].split('.')[0]

    if 'VOT' in args.dataset:
        video_path = join('test', args.dataset, name,
                          'baseline', video['name'])
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                        fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
    else:  # OTB
        video_path = join('test', args.dataset, name)
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:5.3f}fps Lost: {:d}'.format(
        v_id, video['name'], toc, f / toc, lost_times))

    return lost_times, f / toc


def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)

    num_frame = targets.shape[0]
    if start is None:
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        object_ids = [int(id) for id in start]

    num_object = len(object_ids)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)

    output_max_id = np.argmax(outputs, axis=0).astype('uint8')+1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = targets == object_ids[j]

            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(object_ids[j])] - 1
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j+1)
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            res[j, k] = np.mean(iou)
    return res


def track_vos(model, video, hp=None, mask_enable=False, refine_enable=False, mot_enable=False, device='cpu',semip=1):
    image_files = video['image_files']

    annos = [np.array(Image.open(x)) for x in video['anno_files']]
    # set_trace()
    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    if not mot_enable:
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]

        # object_ids = object_ids[:2]
    else:
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]

        # object_ids = object_ids[:2]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)
    object_num = len(object_ids)
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))-1

    masks_nn=0
    if semip or diffuse: masks_nn, masks_semip = pred_masks.copy(),pred_masks.copy()
    visualize = 1
    rf_model = 0
    sv_model = 0
    lr_model = 0
    global fig2,ax2,fig,ax
    # fig2, ax2 = plt.subplots(num_fig, 4, figsize=(20, 20), sharex=True, sharey=True)
    # fig, ax = plt.subplots(num_fig, 6, figsize=(20, 20), sharex=True, sharey=True)
    
    im = cv2.imread(image_files[0])
    mask = annos_init[0]
    if semip: rf_model, lr_model= semip_init(model,im,mask)
    for obj_id, o_id in enumerate(object_ids):

        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)
        set_trace()
        print("number of images:"+str(len(image_files)))
        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            # tic = cv2.getTickCount()
            if f == start_frame:  # init
                # set_trace()
                # if obj_id%num_fig==0:
                #     fig2, ax2 = plt.subplots(num_fig, 4, figsize=(20, 20), sharex=True, sharey=True)
                #     fig, ax = plt.subplots(num_fig, 6, figsize=(20, 20), sharex=True, sharey=True)

                mask = annos_init[obj_id] == o_id
                if semip: rf_model, lr_model= semip_init(model,im,mask)
                tic = cv2.getTickCount()
                x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                cx, cy = x + w/2, y + h/2
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                state = siamese_init(im, target_pos, target_sz, model, hp, device=device)  # init tracker

                if semip:
                    state['rf_model'] = rf_model
                    state['lr_model'] = lr_model
                    state['sv_model'] = sv_model
                if semip or diffuse:
                    state['mask_nn'] = mask
                    state['mask_semip'] = mask

                # if obj_id%num_fig==0:
                #     state['fig'] = fig
                #     state['ax'] = ax 
                #     state['fig2'] = fig2
                #     state['ax2'] = ax2
                #     state['semip'] = semip
            elif end_frame >= f > start_frame:  # tracking
                # set_trace()
                if (f- start_frame)%20==0:
                    visualize_ =1
                else:
                    visualize_=0
                save_name = video['name']+str(obj_id)+'_'+str(o_id)+'_'+str(f)
                state = siamese_track(state, im, mask_enable, refine_enable, device=device,visualize_ =visualize_,save_name = save_name)  # track
                mask = state['mask']
            toc += cv2.getTickCount() - tic
            if end_frame >= f >= start_frame:
                pred_masks[obj_id, f, :, :] = mask
                if semip or diffuse:
                    masks_nn[obj_id, f, :, :] = state['mask_nn']
                    masks_semip[obj_id, f, :, :] = state['mask_semip']

            # break

    
    print("overall time:"+ str(int(toc/cv2.getTickFrequency())))

    if semip:
        toc_semip = state['semip_time']
        print("semip time:"+ str(int(toc_semip/cv2.getTickFrequency())))# if semip:
        # toc = toc*2
        # toc_semip = state['semip_time']
    toc /= cv2.getTickFrequency()
    # print("getTickFrequency:"+str(cv2.getTickFrequency()))

    # pick the optimal threshold

    
    search_optim_semip = 1
    if search_optim_semip:
        optim_iou = 0.0
        optim_seg_bias = 0.0
        optim_ratio = 0.5
        optim_seg_thr=0
        for param_it in range(2,8):

            ratio = param_it*0.1
            guess = ratio * masks_nn + (1.0- ratio)* masks_semip
            for param_it2 in range(-2,3):
                seg_bias =  0.04* param_it2
                # guess_mask = (guess > seg_thr).astype(np.uint8)
                guess_mask = guess - seg_bias
                if len(annos) == len(image_files):
                    multi_mean_iou = MultiBatchIouMeter(thrs, guess_mask, annos,
                                                        start=video['start_frame'] if 'start_frame' in video else None,
                                                        end=video['end_frame'] if 'end_frame' in video else None)
                    # iou_list_semip.append()
                    current_iou = multi_mean_iou[:object_num].max()
                    if current_iou> optim_iou:
                        optim_iou = current_iou
                        optim_seg_bias = seg_bias
                        optim_ratio = ratio
                        # optim_seg_thr = multi_mean_iou[:object_num].argmax()
        print('optim_ratio:'+str(optim_ratio)+ ' optimal_iou:'+str(optim_iou)+ ' optim_seg_bias'+str(optim_seg_bias))
                    # ratio_list.append(ratio)
                    # set_trace()
                    # for i in range(object_num):
                    #     # for j, thr in enumerate(thrs):
                    #     logger.info('Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1),
                    #                                                                    multi_mean_iou[i].mean()))                    
        guess =optim_ratio * masks_nn + (1.0- optim_ratio)* masks_semip
        pred_masks= guess - optim_seg_bias
        # - optim_seg_thr
      
    if len(annos) == len(image_files):
        multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, annos,
                                            start=video['start_frame'] if 'start_frame' in video else None,
                                            end=video['end_frame'] if 'end_frame' in video else None)
        for i in range(object_num):
            for j, thr in enumerate(thrs):
                logger.info('Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1), thr,
                                                                           multi_mean_iou[i, j]))
    else:
        multi_mean_iou = []

    print("args.save_mask:"+str( args.save_mask))
    args.save_mask = True
    if args.save_mask:

        # video_path = join('test', args.dataset, 'SiamMask', video['name'])
        # if diffuse:
        video_path = join('test_diffuse', args.dataset, 'SiamMask', video['name'])
        if not isdir(video_path): makedirs(video_path)
        pred_mask_final = np.array(pred_masks)
        # if search_optim_semip:
        #     seg_thr = optim_seg_thr
        # else:
        #     seg_thr =  state['p'].seg_thr
        seg_thr =  state['p'].seg_thr
        
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > seg_thr).astype('uint8')
        for iii in range(pred_mask_final.shape[0]):
            pred_mask_final[iii,:,:]=cv2.morphologyEx(pred_mask_final[iii,:,:], cv2.MORPH_CLOSE, (5,5))
        mask = COLORS[pred_mask_final]

        if semip:
            postfix = '_difmask.png'
        else:
            postfix='_difmask.png'
        for i in range(pred_mask_final.shape[0]):
            # output = ((0.4 * cv2.imread(image_files[i])) + (0.6 * mask[i,:,:,:])).astype("uint8")
            # output = np.clip(output*1.3,0,255).astype("uint8")
            output2 = pred_mask_final[i].astype(np.uint8)
            # set_trace()
            filename =join(video_path, image_files[i].split('\\')[-1].split('.')[0] + postfix)
            # set_trace()
            cv2.imwrite(filename, output)
            # cv2.imshow("mask_diffuse", output)
            # cv2.waitKey(1)

        pred_mask_final = np.array(masks_nn)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > seg_thr).astype('uint8')
        # pred_mask_final = cv2.morphologyEx(pred_mask_final, cv2.MORPH_CLOSE, (5,5))
        for iii in range(pred_mask_final.shape[0]):
            pred_mask_final[iii,:,:]=cv2.morphologyEx(pred_mask_final[iii,:,:], cv2.MORPH_CLOSE, (5,5))
        mask = COLORS[pred_mask_final]
        if semip:
            postfix = '_base.png'
        else:
            postfix='_base.png'
        for i in range(pred_mask_final.shape[0]):
            # output = ((0.4 * cv2.imread(image_files[i])) + (0.6 * mask[i,:,:,:])).astype("uint8")
            # output = np.clip(output*1.3,0,255).astype("uint8")
            output2 = pred_mask_final[i].astype(np.uint8)
            # set_trace()
            filename =join(video_path, image_files[i].split('\\')[-1].split('.')[0] + postfix)
            cv2.imwrite(filename, output)
                
    if args.visualization:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask = COLORS[pred_mask_final]
        video_path = join('test', args.dataset, 'SiamMask', video['name'])
        if not isdir(video_path): makedirs(video_path)
        #     pred_mask_final = np.array(pred_masks)
        for f, image_file in enumerate(image_files):
            # output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f,:,:,:])).astype("uint8")
            output = ((0.7 * cv2.imread(image_file)) + (0.3 * mask[f,:,:,:])).astype("uint8")
            cv2.imshow("mask", output)
            cv2.waitKey(1)

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f*len(object_ids) / toc))

    return multi_mean_iou, f*len(object_ids) / toc


def main():
    global args, logger, v_id
    args = parser.parse_args()
    cfg = load_config(args)

    global num_fig,fig2,ax2,fig,ax
    num_fig =4
    # fig2, ax2 = plt.subplots(num_fig, 4, figsize=(20, 20), sharex=True, sharey=True)
    global fig_counter2, fig_counter1
    
    # fig, ax = plt.subplots(num_fig, 6, figsize=(20, 20), sharex=True, sharey=True)

    # set_trace()
    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)

    # setup model
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(anchors=cfg['anchors'])
    elif args.arch == 'Custom_semip':
        from custom_semip import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        parser.error('invalid architecture: {}'.format(args.arch))

    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)
    model.eval()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    model = model.to(device)
    # setup dataset
    dataset = load_dataset(args.dataset)

    # model_shallow = resnet18(pretrained=True)
    # model_shallow.eval()
    # # device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    # model_shallow = model_shallow.to(device)



    # VOS or VOT?
    if args.dataset in ['DAVIS2016', 'DAVIS2017', 'ytb_vos'] and args.mask:
        vos_enable = True  # enable Mask output
    else:
        vos_enable = False

    total_lost = 0  # VOT
    iou_lists = []  # VOS
    speed_list = []
    if args.jq:
        semip=1
    else:
        semip = 0
    print('args.jq'+str(args.jq))

    for v_id, video in enumerate(dataset.keys(), start=1):
        # for v_id, video in list(enumerate(dataset.keys(), start=1))[3:]:
        if args.video != '' and video != args.video:
            continue

        if vos_enable:
            iou_list, speed = track_vos(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                 args.mask, args.refine, args.dataset in ['DAVIS2017', 'ytb_vos'], 
                                 device=device,semip = semip)
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
