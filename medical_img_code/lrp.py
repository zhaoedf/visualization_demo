import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.nn.functional as F
import pickle
import numpy as np

import cv2


def model_flattening(module_tree):
    module_list = []
    children_list = list(module_tree.children())
    if len(children_list) == 0 or isinstance(module_tree, BasicBlock) or \
        isinstance(module_tree, Bottleneck):
        return [module_tree]
    else:
        for i in range(len(children_list)):
            module = model_flattening(children_list[i])
            module = [j for j in module]
            module_list.extend(module)
        return module_list


class ActivationStoringNet(nn.Module):
    def __init__(self, module_list):
        super(ActivationStoringNet, self).__init__()
        self.module_list = module_list

    def basic_block_forward(self, basic_block, activation):
        identity = activation

        basic_block.conv1.activation = activation
        activation = basic_block.conv1(activation)
        activation = basic_block.relu(basic_block.bn1(activation))
        basic_block.conv2.activation = activation
        activation = basic_block.conv2(activation)
        activation = basic_block.bn2(activation)
        if basic_block.downsample is not None:
            for i in range(len(basic_block.downsample)):
                basic_block.downsample[i].activation = identity
                identity = basic_block.downsample[i](identity)
            basic_block.identity = identity
        basic_block.activation = activation
        output = activation + identity
        output = basic_block.relu(output)

        return basic_block, output

    def bottleneck_forward(self, bottleneck, activation):
        identity = activation

        bottleneck.conv1.activation = activation
        activation = bottleneck.conv1(activation)
        activation = bottleneck.relu(bottleneck.bn1(activation))
        bottleneck.conv2.activation = activation
        activation = bottleneck.conv2(activation)
        activation = bottleneck.relu(bottleneck.bn2(activation))
        bottleneck.conv3.activation = activation
        activation = bottleneck.conv3(activation)
        activation = bottleneck.bn3(activation)
        if bottleneck.downsample is not None:
            for i in range(len(bottleneck.downsample)):
                bottleneck.downsample[i].activation = identity
                identity = bottleneck.downsample[i](identity)
            bottleneck.identity = identity
        bottleneck.activation = activation
        output = activation + identity
        output = bottleneck.relu(output)

        return bottleneck, output

    def forward(self, x):
        module_stack = []
        activation = x

        for i in range(len(self.module_list)):
            module = self.module_list[i]
            if isinstance(module, BasicBlock):
                module, activation = self.basic_block_forward(module, activation)
                module_stack.append(module)
            elif isinstance(module, Bottleneck):
                module, activation = self.bottleneck_forward(module, activation)
                module_stack.append(module)
            else:
                module.activation = activation
                module_stack.append(module)
                activation = module(activation)
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    activation = activation.view(activation.size(0), -1)

        output = activation

        return module_stack, output


class DTD(nn.Module):
    def __init__(self, lowest=0., highest=1.):
        super(DTD, self).__init__()
        self.lowest = lowest
        self.highest = highest

    def forward(self, module_stack, y, class_num, model_archi):
        '''
            store_model = ActivationStoringNet(model_flattening(model))
            stack, output = store_model(inp)

            saliency_map = dtd(stack, output, n_classes, 'resnet')
        '''
        R = torch.eye(class_num)[torch.max(y, 1)[1]] 
        '''
            # 这里的torch.max(y,1)表示默认以predict logits最大的那个类为作为预测类别，来进行可视化。
            # 因为类似grad-CAM等基于梯度的可视化方法里面会一般会默认从一个logit开始求导，因为叶子节点 && scalar才能求导，易懂。
            # 所以你可以使用GT对应的logit，或者使用max的logit，根据实际情况(i.e. 有无GT)来确定就好了。
        '''

        for i in range(len(module_stack)):
            module = module_stack.pop()
            if len(module_stack) == 0:
                if isinstance(module, nn.Linear):
                    activation = module.activation
                    R = self.backprop_dense_input(activation, module, R)
                elif isinstance(module, nn.Conv2d):
                    activation = module.activation
                    R = self.backprop_conv_input(activation, module, R)
                else:
                    raise RuntimeError(f'{type(module)} layer is invalid initial layer type')
            elif isinstance(module, BasicBlock):
                R = self.basic_block_R_calculate(module, R)
            elif isinstance(module, Bottleneck):
                R = self.bottleneck_R_calculate(module, R)
            else:
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    if model_archi == 'vgg':
                        R = R.view(R.size(0), -1, 7, 7)
                        continue
                    elif model_archi == 'resnet':
                        R = R.view(R.size(0), R.size(1), 1, 1)
                activation = module.activation
                R = self.R_calculate(activation, module, R)

        return R

    def basic_block_R_calculate(self, basic_block, R):
        if basic_block.downsample is not None:
            identity = basic_block.identity
        else:
            identity = basic_block.conv1.activation
        activation = basic_block.activation
        R0, R1 = self.backprop_skip_connect(activation, identity, R)
        R0 = self.backprop_conv(basic_block.conv2.activation, basic_block.conv2, R0)
        R0 = self.backprop_conv(basic_block.conv1.activation, basic_block.conv1, R0)
        if basic_block.downsample is not None:
            for i in range(len(basic_block.downsample)-1, -1, -1):
                R1 = self.R_calculate(basic_block.downsample[i].activation,
                                      basic_block.downsample[i], R1)
        else:
            pass
        R = self.backprop_divide(R0, R1)
        return R

    def bottleneck_R_calculate(self, bottleneck, R):
        if bottleneck.downsample is not None:
            identity = bottleneck.identity
        else:
            identity = bottleneck.conv1.activation
        activation = bottleneck.activation
        R0, R1 = self.backprop_skip_connect(activation, identity, R)
        R0 = self.backprop_conv(bottleneck.conv3.activation, bottleneck.conv3, R0)
        R0 = self.backprop_conv(bottleneck.conv2.activation, bottleneck.conv2, R0)
        R0 = self.backprop_conv(bottleneck.conv1.activation, bottleneck.conv1, R0)
        if bottleneck.downsample is not None:
            for i in range(len(bottleneck.downsample)-1, -1, -1):
                R1 = self.R_calculate(bottleneck.downsample[i].activation,
                                      bottleneck.downsample[i], R1)
        else:
            pass
        R = self.backprop_divide(R0, R1)

        return R

    def R_calculate(self, activation, module, R):
        if isinstance(module, nn.Linear):
            R = self.backprop_dense(activation, module, R)
            return R
        elif isinstance(module, nn.Conv2d):
            R = self.backprop_conv(activation, module, R)
            return R
        elif isinstance(module, nn.BatchNorm2d):
            R = self.backprop_bn(R)
            return R
        elif isinstance(module, nn.ReLU):
            R = self.backprop_relu(activation, R)
            return R
        elif isinstance(module, nn.MaxPool2d):
            R = self.backprop_max_pool(activation, module, R)
            return R
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            R = self.backprop_adap_avg_pool(activation, R)
            return R
        elif isinstance(module, nn.Dropout):
            R = self.backprop_dropout(R)
            return R
        else:
            raise RuntimeError(f"{type(module)} can not handled currently")

    def backprop_dense(self, activation, module, R):
        W = torch.clamp(module.weight, min=0)
        Z = torch.mm(activation, torch.transpose(W, 0, 1)) + 1e-9
        S = R.cuda() / Z
        C = torch.mm(S, W)
        R = activation * C

        return R

    def backprop_dense_input(self, activation, module, R):
        W_L = torch.clamp(module.weight, min=0)
        W_H = torch.clamp(module.weight, max=0)

        L = torch.ones_like(activation, dtype=activation.dtype) * self.lowest
        H = torch.ones_like(activation, dtype=activation.dtype) * self.highest

        Z_O = torch.mm(activation, torch.transpose(module.weight, 0, 1))
        Z_L = torch.mm(activation, torch.transpose(W_L, 0, 1))
        Z_H = torch.mm(activation, torch.transpose(W_H, 0, 1))

        Z = Z_O - Z_L - Z_H + 1e-9
        S = R / Z

        C_O = torch.mm(S, module.weight)
        C_L = torch.mm(S, W_L)
        C_H = torch.mm(S, W_H)

        R = activation * C_O - L * C_L - H * C_H

        return R

    def backprop_conv(self, activation, module, R):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        output_padding = activation.size(2) - ((R.size(2) - 1) * stride[0] \
                                        - 2 * padding[0] + kernel[0])
        W = torch.clamp(module.weight, min=0)
        Z = F.conv2d(activation, W, stride=stride, padding=padding) + 1e-9
        S = R / Z
        C = F.conv_transpose2d(S, W, stride=stride, padding=padding, output_padding=output_padding)
        R = activation * C

        return R

    def backprop_conv_input(self, activation, module, R):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        output_padding = activation.size(2) - ((R.size(2) - 1) * stride[0] \
                                                - 2 * padding[0] + kernel[0])

        W_L = torch.clamp(module.weight, min=0)
        W_H = torch.clamp(module.weight, max=0)

        L = torch.ones_like(activation, dtype=activation.dtype) * self.lowest
        H = torch.ones_like(activation, dtype=activation.dtype) * self.highest

        Z_O = F.conv2d(activation, module.weight, stride=stride, padding=padding)
        Z_L = F.conv2d(L, W_L, stride=stride, padding=padding)
        Z_H = F.conv2d(H, W_H, stride=stride, padding=padding)

        Z = Z_O - Z_L - Z_H + 1e-9
        S = R / Z

        C_O = F.conv_transpose2d(S, module.weight, stride=stride, padding=padding, output_padding=output_padding)
        C_L = F.conv_transpose2d(S, W_L, stride=stride, padding=padding, output_padding=output_padding)
        C_H = F.conv_transpose2d(S, W_H, stride=stride, padding=padding, output_padding=output_padding)

        R = activation * C_O - L * C_L - H * C_H

        return R

    def backprop_bn(self, R):
        return R

    def backprop_dropout(self, R):
        return R

    def backprop_relu(self, activation, R):
        return R

    def backprop_adap_avg_pool(self, activation, R):
        kernel_size = activation.shape[-2:]
        Z = F.avg_pool2d(activation, kernel_size=kernel_size) * kernel_size[0] ** 2 + 1e-9
        S = R / Z
        R = activation * S

        return R

    def backprop_max_pool(sef, activation, module, R):
        kernel_size, stride, padding = module.kernel_size, module.stride, module.padding
        Z, indices = F.max_pool2d(activation, kernel_size=kernel_size, stride=stride, \
                                  padding=padding, return_indices=True)
        Z = Z + 1e-9
        S = R / Z
        C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride, \
                            padding=padding, output_size=activation.shape)
        R = activation * C

        return R

    def backprop_divide(self, R0, R1):
        return R0 + R1

    def backprop_skip_connect(self, activation0, activation1, R):
        Z = activation0 + activation1 + 1e-9
        S = R / Z
        R0 = activation0 * S
        R1 = activation1 * S

        return (R0, R1)


def clone_freeze(model: nn.Module):
    model = pickle.loads(pickle.dumps(model))
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    return model

@torch.no_grad()
def dtd_for_resnet(model: nn.Module, inp, n_classes=1000, scale=5000):
    model = clone_freeze(model)
    dtd = DTD()
    store_model = ActivationStoringNet(model_flattening(model))
    stack, output = store_model(inp)
    # def forward(self, module_stack, y, class_num, model_archi):
    saliency_map = dtd(stack, output, n_classes, 'resnet')
    saliency_map = saliency_map.detach().permute((0, 2, 3, 1)).cpu().numpy() # (1,224,224,15)

    # sum over channel dim.
    # saliency_map = np.sum(saliency_map, axis=3) # (1,224,224)

    saliency_map = np.maximum(0, saliency_map) * 255 * scale
    saliency_map = np.minimum(255, saliency_map)
    saliency_map = np.uint8(saliency_map)
    
    return saliency_map


def get_img_ndarray(path: str):
    sub_images = os.listdir(path)
    sub_images = sorted(sub_images, key=lambda x: int(x[:-4]) )
    
    array_list = []  # 存放15张子图

    for image_file in sub_images[:15]:
        image_array = np.load(os.path.join(path, image_file))
        # print(image_array.shape)
        image_array = Image.fromarray(image_array, mode='RGB')
        image_array = image_array.copy().convert('L')  # 转换成灰度图

        array_list.append(tf.ToTensor()(image_array))
    
    
    return torch.cat(tuple(array_list))


def heatmap_overlap_display_cv2(saliency_map, oimg, save_path, color_map=cv2.COLORMAP_JET):    
    # assume both saliency_map and oimg are in the range of [0,255], but their dtype can be float.
    if len(oimg.shape) == 2:
        oimg = oimg[:,:,np.newaxis]
        oimg = oimg.repeat(3, axis=2)
    oimg = np.uint8(oimg)
    
    H, W = saliency_map.shape
    save_img = np.zeros((H, W*2, 3), dtype=np.uint8)
    
    heatmap = cv2.applyColorMap(saliency_map, color_map)
    heatmap = cv2.addWeighted(src1=oimg, alpha=1, src2=heatmap, beta=0.5, gamma=0)
    
    save_img[:H, :W] = np.uint8(oimg)
    save_img[:H, W:] = heatmap
    
    cv2.imwrite(save_path, save_img)

def heatmap_overlap_display_plt(saliency_map, oimg, save_path, color_map='jet'):
    if len(oimg.shape) == 2:
        oimg = oimg[:,:,np.newaxis]
        oimg = oimg.repeat(3, axis=2)
    oimg = np.uint8(oimg)
    
    H, W = saliency_map.shape
    save_img = np.zeros((H, W*2, 3), dtype=np.uint8)
    
    colormap = plt.get_cmap(color_map)
    heatmap = (colormap(saliency_map) * 255).astype(np.uint8)[:,:,:3] # https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
    heatmap_overlapped = cv2.addWeighted(src1=oimg, alpha=1.0, src2=heatmap, beta=0.5, gamma=0)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR) no need for this, cos this is whole matplotlib RGB stack.
    
    # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    fig, (ax1, ax2) = plt.subplots(figsize=(32,32), nrows=1, ncols=2)
    ax1.imshow(oimg, vmin=0, vmax=255)
    ax1.axis('off')
    im = ax2.imshow(heatmap_overlapped, cmap=color_map, vmin=0, vmax=255)
    ax2.axis('off')

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.3) # https://matplotlib.org/stable/gallery/color/colorbar_basics.html
    fig.savefig(save_path,  bbox_inches='tight') # https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen



if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from PIL import Image
    import copy
    import torchvision.transforms as tf
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    import torchvision.models as models

    image_path = './test_dataset/20211129/msmi_image/label=0/204307_LuoHezhong' # 20211129/msmi_image/label=0/211726_XiaoZuyun
    image = get_img_ndarray(image_path)
    ori_image = copy.deepcopy(cv2.resize(image.numpy(), (224, 224)))
    
    trans = tf.Compose([
        tf.Resize((224, 224)),
        # tf.ToTensor(),
        # tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    image = trans(image).unsqueeze(0) # (1, 15, 224, 224)
    # model = models.resnet18(pretrained=True)
    model = torch.load('./model.pth')

    import matplotlib.pyplot as plt

    saliency_map = dtd_for_resnet(model.cuda(), image.cuda(), n_classes=2, scale=50000)[0] # (224, 224, 15/1)
    # plt.imshow(saliency_map, cmap='gray')
    # plt.axis('off')
    # plt.savefig('./res_lrp.png')
    
    SAVE_PATH = './res_lrp'
    if os.path.exists(SAVE_PATH):
        import shutil
        shutil.rmtree(SAVE_PATH)
    os.mkdir(SAVE_PATH)
    
    image = image.detach().permute((0, 2, 3, 1)).cpu().numpy() # (1, 224, 224, 15)
    image = image[0] # (224, 224, 15)
    for c in range(saliency_map.shape[2]): # 3 ~ channel dim
        lrp_img = saliency_map[:, :, c]
        oimg = image[:, :, c]*255

        save_path = os.path.join(SAVE_PATH, f'{str(c)}.png')
        # heatmap_overlap_display_cv2(lrp_img, oimg, save_path)
        heatmap_overlap_display_plt(lrp_img, oimg, save_path)

    print(saliency_map[0,0], saliency_map[30, 30])
        
