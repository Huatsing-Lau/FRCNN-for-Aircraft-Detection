import cv2
import numpy as np

# 输入一个label(*.txt)文件的全路径，该函数将读取训练数据
# 输出all_data, classes_count, class_mapping
# 其中：

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}#字典：统计数据集中各类目标的个数

    class_mapping = {}#字典：给数据集中各类目标编号0,1,2,...

    visualise = True

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            # label(*.txt)文件的格式，例子如下：
            # ./media/jintian/Netac/Datasets/Kitti/object/training/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            # 生成class_mapping，读取txt文件过程中一旦发现新的类别名称，就在class_mapping中增加一个词条
            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)# class_mapping是字典，表示类别名称与类别编号的对应关系


            if filename not in all_imgs:
                all_imgs[filename] = {}# all_imgs也是字典，key是文件名图片，value是一个子字典（包含bboxes（标注信息）,filepath,imageset,height,width等词条）

                img = cv2.imread(filename)# 读取图片数据，只是为了读取图片的size而已
                (rows, cols) = img.shape[:2]# 图片的size,应该注意第三维是通道数（3）
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 10) > 0:  #0 1 2 3 4 5 等概率取随机值，按5比1分割数据集为训练/验证集和测试集
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:#all_imgs的key就是一个个图片文件名
            all_data.append(all_imgs[key])#(all_imgs[key]就是all_imgs的value值，是字典，包含bboxes（标注信息）,filepath,imageset,height,width等词条

        # make sure the bg class is last in the list
        # 确保让bg的编号是最后一个
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping  #返回的all_data是一个字典，里面没有RGB数据本身，有的是图片数据的信息（路径/文件名、height、width、boxes,imageset）

