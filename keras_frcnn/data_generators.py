# 被训练脚本调用，
# 输入：训练集列表（列表的每个元素是字典）
# 输出：数据扩增，然后生成frcnn所需的训练/验证数据（如：图片、rpn的梯度等等）

from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment  # '.'表示上一级目录
import threading
import itertools


def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)

# 按比例缩放原图的size,取整
def get_new_img_size(width, height, img_min_side=600):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]#找出box样本数非零的那些类别编号
		self.class_cycle = itertools.cycle(self.classes)# 循环器是对象的容器，包含有多个对象。通过调用循环器的cycle方法则代表重复序列的无限循环，举例子：cycle('abc') 则重复序列的元素，即a, b, c, a, b, c ...
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):

	downscale = float(C.rpn_stride) #cfg.rpn_stride=16  这个参数是resnet50版本的basenet输出的sharelayer相对原图的缩小比例
	anchor_sizes = C.anchor_box_scales #cfg.anchor_box_scales=[128, 256, 512]
	anchor_ratios = C.anchor_box_ratios #cfg.anchor_box_ratios=[[1, 1], [1, 2], [2, 1]]
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	 #num_anchors=9

	# calculate the output map size based on the network architecture
   # 函数img_length_calc_function是nn.get_img_output_length，通过参数传递的方式传递进来，计算的是share_layer的长宽尺寸
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height)#resized_width=600#output_width=38，按原图等比例缩放output_height

	n_anchratios = len(anchor_ratios) #n_anchratios=3
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

	num_bboxes = len(img_data['bboxes'])#num_bboxes:一个训练样本中的目标框个数，对于每个样本，取值不一样

   # 预定义几个数组，用于记录最优anchor的相关参数
	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))#人工标注的目标框坐标（resize之后的）
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# rpn ground truth
   # 第一个for循环：遍历固定框的尺寸anchor_sizes=[128, 256, 512]，
   # 第二个循环：遍历固定框的长宽比anchor_ratios=[[1, 1], [1, 2], [2, 1]]，
   # 第三个循环：遍历share_layer宽度方向每一个像素
   # 第四个循环：遍历share_layer长度方向每一个像素
   # 第五个循环：遍历每个真实的目标框（num_bboxes:一个训练样本中的目标框个数，对于每个样本，取值不一样）
	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
			
			for ix in range(output_width):					
				# x-coordinates of the current anchor box	
             # 计算出当前anchor对应的原图坐标（上顶点x1_anc,下顶点x2_anc）
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > resized_width:
					continue
					
				for jy in range(output_height):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# bbox_type indicates whether an anchor should be a target 
					bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0
                 
                # 循环num_bboxes:一个训练样本中的目标框个数，对于每个样本，取值不一样
					for bbox_num in range(num_bboxes):
						
						# get IOU of the current GT box and the current anchor box
                    # gta是人工标注的目标框在share_layer上投影的坐标
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
                    # cfg.rpn_max_overlap=0.7是预设的一个阈值 （大于该阈值则是正样本）
                    
                    # 注意：这部分的定义与吴佳祥硕士学位论文公式（4-2）同，可见本程序的rpn是采用偏移坐标
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0 #人工标注的原图（resize之后的原图）目标框的几何中心坐标
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0 #当前anchor对应的原图的几何中心坐标
							cya = (y1_anc + y2_anc)/2.0
                       
                       #以下梯度就是rpn要预测的值，当一个anchor与任意一个人工标注的框完全重合的时候，该值就等于[1 1 1 1] 
							tx = (cx - cxa) / (x2_anc - x1_anc) #一个梯度，表征当前anchar与当前人工标注目标框的几何中心x方向的（归一化）距离
							ty = (cy - cya) / (y2_anc - y1_anc) #一个梯度,~y方向~
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc)) #也是一个梯度
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc)) #还是一个梯度
						
						if img_data['bboxes'][bbox_num]['class'] != 'bg':

							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# turn on or off outputs depending on IOUs
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1 #y_is_box_valid就是rpn的cls分支输出的那个(shared_layer)w*(shared_layer)h*（类别数（8）+1（bg））的label值
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr # 只有正样的y_rpn_regr才不为0

	# we ensure that every bbox has at least one positive RPN region
   # 对于任意一个人工标注目标框，如果不存在一个anchor与它的iou大于预先设定的阈值（0.7），则令其中iou最大的那一个anchor作为
	for idx in range(num_anchors_for_bbox.shape[0]):#num_anchors_for_bbox用来记录当前训练样本的各个人工标记目标框的正样anchor数量
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))#之前尺寸是(h,w,classnum(包括背景))，transpose之后的尺寸是(classnum(包括背景),h,w)
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)#尺寸是(1,classnum(包括背景),h,w)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))#tranpoe之后的尺寸是(4*classnum(包括背景),h,w)
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)#尺寸是(1,4*classnum(包括背景),h,w)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))#正样标记
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))#负样标记

	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256 #以上求出的anchor负样数量远远多于正样，所以在此减少正样+负样数量到256个

	if len(pos_locs[0]) > num_regions/2: #这种情况下，随机删减正样数量到128个
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:#这种情况下，随机删减负样，使得正样+负样总数为256
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1) # 前一个元素为1（0）则表示是（不是）正或负样本，后一个为1（0）则表示是（不是）正样本
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)#前四个元素表示是不是正样，后四个元素才是bbox#为什么来个repeat赋值给前面一半

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

# get_anchor_gt函数返回一个生成器
def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)
    sample_selector = SampleSelector(class_count)
    while True:
        if mode == 'train':
            random.shuffle(all_img_data)
            
        for img_data in all_img_data:
            try:
                #如果用户指定对类别的样本数做均衡处理，且当前图片中存在当前被均衡的类别，则下一个循环
                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue
                # read in image, and optionally add augmentation
                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)# img_data_aug, x_img分别是扩增后的数据信息和图像本身
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)
                    
                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape
                
                assert cols == width  #assert的作用是如果它的条件返回错误,则终止程序执行
                assert rows == height  #assert的作用是如果它的条件返回错误,则终止程序执行
                
                resized_height = height if height%16==0 else ( height+(16-height%16) if height%16>4 else height-height%16 )
                resized_width = width if width%16==0 else ( width+(16-width%16) if width%16>8 else width-width%16 ) 
                # 在这里对原图做缩放,我认为是可能是为了减少对内存需求
                # 我建议取消这种缩放，会导致原图中像数90以下的目标在bsenet输出的share_layer中缩小为一个像素不到
                # get image dimensions for resizing
                #(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)#C.im_size=600
                # resize the image so that smalles side is length = 600px
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                
                
                
                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                except:
                    continue
                
                
                # Zero-center by mean pixel, and preprocess image
                x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB   # 转换前后x_img的size都是600*600*3
                x_img = x_img.astype(np.float32)  # 试验表明，数据类型转换之前就是float32
                x_img[:, :, 0] -= C.img_channel_mean[0] # 减去该通道的像素值均值，这个均值由用户设定
                x_img[:, :, 1] -= C.img_channel_mean[1] # 减去该通道的像素值均值，这个均值由用户设定
                x_img[:, :, 2] -= C.img_channel_mean[2] # 减去该通道的像素值均值，这个均值由用户设定
                x_img /= C.img_scaling_factor  # 源代码中，C.img_scaling_factor == 1,可以考虑改为255
                
                x_img = np.transpose(x_img, (2, 0, 1)) # transpose之后的3个维度分别是（channel,h,w）
                x_img = np.expand_dims(x_img, axis=0) # expand_dims之后，size是(1,channel,h,w)
                
                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling
                
                if backend == 'tf':
                    x_img = np.transpose(x_img, (0, 2, 3, 1))#transpose之后的给维度分别是（1,h,w,channel）
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))#transpose之后的尺寸是，eg:(1,share_layer_h,share_layer_w,2*scale数*ratio数)
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))#transpose之后的尺寸是，eg:(1,share_layer_h,share_layer_w,8*scale数*ratio数)
                    
                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
                
            except Exception as e:
                print(e)
                continue
