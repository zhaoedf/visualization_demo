import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
import numpy as np
from PIL import Image
import copy
import matplotlib.cm as mpl_color_map


def clone_freeze(model: nn.Module):
    model = pickle.loads(pickle.dumps(model))
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    return model


def gradcam_for_resnet(model: nn.Module, inp, target):
    model = clone_freeze(model)
    features = nn.Sequential(*list(model.children())[:-2])
    avgpool = model.avgpool
    fc = model.fc

    feature_maps = features(inp)
    feature_maps = Variable(feature_maps.detach(), requires_grad=True)
    out = fc(avgpool(feature_maps).view(inp.size(0), -1))
    one_hot = torch.zeros(target.size(0), out.size(1)).to(inp.device)
    one_hot[range(target.size(0)), target] = 1
    out.backward(gradient=one_hot)
    weight = feature_maps.grad.detach()
    weight = torch.mean(weight, dim=(2, 3))
    n_channel = feature_maps.size(1)

    cam = torch.zeros((inp.size(0), feature_maps.size(2), feature_maps.size(3))).to(inp.device)
    for i in range(n_channel):
        cam += weight[:, i][:, None, None] * feature_maps[:, i]
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-9)
    cam = np.uint8(cam * 255)

    ret = np.uint8(np.zeros((inp.size(0), inp.size(2), inp.size(3))))
    for i in range(inp.size(0)):
        ret[i] = np.uint8(Image.fromarray(cam[i]).resize((inp.size(2), inp.size(3)), Image.ANTIALIAS))
    return ret


def apply_colormap_on_image(org_im, activation, colormap_name):
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import torchvision.models as models
    import torchvision.transforms as tf
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    import matplotlib.pyplot as plt


    image = Image.open(r'E:\Projects\GradCAM\testImage\ILSVRC2012_val_00043300.JPEG')
    ori_image = copy.deepcopy(image.resize((224, 224)))
    trans = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor(),
        tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    image = trans(image).unsqueeze(0)
    target = torch.LongTensor([15])

    model = models.resnet18(pretrained=True)
    cam = gradcam_for_resnet(model.cuda(), image.cuda(), target.cuda())

    ori, cam = apply_colormap_on_image(ori_image, cam[0], 'jet')
    plt.imshow(cam)
    plt.axis('off')
    plt.show()