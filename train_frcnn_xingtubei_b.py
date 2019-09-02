"""
this code will train on kitti data set
这个函数是原始faster-rcnn在星图杯数据集上的训练
"""
from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import pickle
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses_fn
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
import os
from keras_frcnn import resnet as nn
from keras_frcnn.simple_parser import get_data


def train_kitti():
    # config for data argument
    cfg = config.Config()
    cfg.balanced_classes = True
    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.num_rois = 50# 对于星图杯的光学遥感飞机检测，应该改为50+
    cfg.anchor_box_scales = [41,70,120,20,90]
    cfg.anchor_box_ratios = [[1,1.4],[1,0.84],[1,1.17],[1,0.64],[1,1]]

    cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path())
    

    # TODO: the only file should to be change for other data to train
    cfg.model_path = './model/kitti_frcnn_last.hdf5'
    cfg.simple_label_file = 'E:/Xingtubei/official_datas/OpticalAircraft/laptop_Chreoc_OpticalAircraft_bboxes.txt' # '/media/liuhuaqing/Elements/Xingtubei/official_datas/OpticalAircraft/Chreoc_OpticalAircraft_bboxes.txt'#'F:/Xingtubei/official_datas/OpticalAircraft/Chreoc_OpticalAircraft_bboxes.txt' # 'kitti_simple_label.txt'

    all_images, classes_count, class_mapping = get_data(cfg.simple_label_file)#读取数据集，cv2.imread()要求数据里不能有中文路径

    if 'bg' not in classes_count:#'bg'应该是代表背景
        classes_count['bg'] = 0 # =0表示训练数据中没有“背景”这一类别
        class_mapping['bg'] = len(class_mapping)

    cfg.class_mapping = class_mapping
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cfg.config_save_file))

    inv_map = {v: k for k, v in class_mapping.items()}#class_mapping的逆向map

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    random.shuffle(all_images)
    num_imgs = len(all_images)
    train_imgs = [s for s in all_images if s['imageset'] == 'trainval'] #训练集，列表形式，列表中的元素是字典
    val_imgs = [s for s in all_images if s['imageset'] == 'test'] #验证集，列表形式，列表中的元素是字典

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,
                                                   K.image_dim_ordering(), mode='train')#数据扩增，然后生成frcnn所需的训练数据（如：图片、rpn的梯度等等）
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,
                                                 K.image_dim_ordering(), mode='val')#数据扩增，然后生成frcnn所需的验证数据（如：图片、rpn的梯度等等）

    # 根据keras实际用的后端，定义相应的输入数据维度，因为两类后端的维度顺序不一样
    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)#当后端是thaneo
    else:
        input_shape_img = (None, None, 3)#当后端是tensorflow

    img_input = Input(shape=input_shape_img) # 输入图片
    roi_input = Input(shape=(None, 4)) # 输入人工标注的roi坐标，4表示x1,y1,x2,y2

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)# shared_layers是frcnn网络底部那些共享的层,在这里是ResNet。由nn定义好

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(classes_count), trainable=True)

    model_rpn = Model(img_input, rpn[:2])#rpn网络由keras_frcnn/resnet定义好。rpn[:2]的前两个元素分别表示rpn网络的分类输出和回归输出
    model_classifier = Model([img_input, roi_input], classifier)#Keras的函数式模型为Model，即广义的拥有输入和输出的模型

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)#rpn[:2]+classifier的含义是？？？？？？

    try:
        # 尝试载入与训练网络权值
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.model_path, by_name=True)
        model_classifier.load_weights(cfg.model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
              'https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = Adam(lr=1e-5)# 定义一个Adam求解器，学习率lr
    optimizer_classifier = Adam(lr=1e-5)# 定义一个Adam求解器，学习率lr
    # num_anchors等于9
    model_rpn.compile(optimizer=optimizer,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 100 # 每迭代epoch_length次就检查一次是否要保存网络权值，然后重置iter_num = 0
    num_epochs = int(cfg.num_epochs)
    iter_num = 0 # 迭代次数的初值

    losses = np.zeros((epoch_length, 5)) # 初始化loss数组，记录每个周期的loss
    rpn_accuracy_rpn_monitor = []  # 初始化一个数组，记录rpn的训练过程中的精度变化
    rpn_accuracy_for_epoch = [] # 初始化一个数组，记录rpn的每个训练周期的的精度变化 
    start_time = time.time() # 开始训练的时间

    best_loss = np.Inf # 改变量纪律训练以来最小的loss

    class_mapping_inv = {v: k for k, v in class_mapping.items()} # class_mapping_inv是一个字典，key是目标类别编号，value是类别名称
    print('Starting training')

    vis = True

    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length) # 生成一个进度条对象
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs)) # 输出当前训练周期数/总周期数

        while True:# 什么时候才结束这个循环？答：第247行的break(每迭代epoch_length次)
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose: # 每epoch_length次训练周期就在窗口显示一次RPN平均精度
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap'
                              ' the ground truth boxes. Check RPN settings or keep training.')
                
                #X应该是图像，如kitti尺寸是(1,600,1987,3)。Y是label，img_data是字典，包含文件名、尺寸、人工标记的roi和类别等
                X, Y, img_data = next(data_gen_train)
                Y_1=Y[0]
                Y_1=Y_1[0,:,:,:]

                loss_rpn = model_rpn.train_on_batch(X, Y)#为什么Y的尺寸与P_rpn的尺寸不同？为什么loss_rpn的尺寸是3，含义是什么，在哪里定义的？

                P_rpn = model_rpn.predict_on_batch(X)#P_rpn的尺寸是(1, 124, 38, 9) (1, 124, 38, 36)

                result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=300) #result的尺寸是300*4
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # X2的尺寸是100*4，Y1的尺寸是1*100*8（8=训练集中目标类别总数），IouS尺寸是100
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)#Y2的尺寸是1*1*56,56=28*2，（28=4*7）前28是coords，后28是labels(是该类别则标1)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)#Y1的尺寸是1*1*8表示分类预测结果，最后一个元素为1表示是背景
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if cfg.num_rois > 1:
                    if len(pos_samples) < cfg.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])#用rpn输出的roi输入给classifier

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:# 每迭代epoch_length次就检查一次是否要保存网络权值，然后重置iter_num = 0
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if cfg.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if cfg.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg.model_path)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                # save model
                model_all.save_weights(cfg.model_path)
                continue
    print('Training complete, exiting.')


if __name__ == '__main__':
    train_kitti()
