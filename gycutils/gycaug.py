from PIL import Image,ImageEnhance
import numpy as np
import random
from skimage import color,exposure
#import cv2

class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,label):
        for t in self.transforms:
            img,label = t(img,label)
        return img,label

# 用于图像增强。具体来说，它对输入的图像进行直方图均衡化和伽马校正，并返回一个4维图像。
class Retina_enhance(object):#图像增强
    def __init__(self):
        pass
    def __call__(self, img):
        '''
        :param img:should be pil image
        :return:4-dimension image (l,a,b,g-enhance)
        '''
        npimg=np.array(img)#转为numpy格式
        g_enhance = exposure.equalize_hist(npimg[:,:,1])#直方图均衡化
        g_enhance = exposure.adjust_gamma(g_enhance, 0.1)
        return np.dstack((g_enhance,g_enhance,g_enhance))

class Random_vertical_flip(object):
    def _vertical_flip(self,img,label):
        return img.transpose(Image.FLIP_TOP_BOTTOM),label.transpose(Image.FLIP_TOP_BOTTOM)
    def __init__(self,prob):
        '''

        :param prob: should be (0,1)
        '''
        assert prob>=0 and prob<=1,"prob should be [0,1]"
        self.prob=prob
    def __call__(self, img,label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image),"should be PIL image"
        assert isinstance(label, Image.Image),"should be PIL image"
        if random.random()<self.prob:
            return self._vertical_flip(img,label)
        return img,label


class Random_horizontal_flip(object):
    def _horizontal_flip(self,img,label):
        #dsa
        return img.transpose(Image.FLIP_LEFT_RIGHT),label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self,prob):
        '''

        :param prob: should be (0,1)
        '''
        assert prob>=0 and prob<=1,"prob should be [0,1]"
        self.prob=prob

    def __call__(self, img,label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image),"should be PIL image"
        assert isinstance(label, Image.Image),"should be PIL image"
        if random.random()<self.prob:
            return self._horizontal_flip(img,label)
        else:
            return img,label
class ColorAug(object):
    def _randomColor(self,image):
        """
        :param image: PIL的图像image
        :return:
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

    def __call__(self, img):
        '''

        :param img:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image),"should be PIL image"
        return self._randomColor(img)

class Add_Gaussion_noise(object):
    def _gaussian_noise(self,npimg,mean=0,sigma=0):
        noise=np.random.normal(mean,sigma,npimg.shape)
        newimg=npimg+noise
        newimg[newimg>255]=255
        newimg[newimg<0]=0
        return Image.fromarray(newimg.astype(np.uint8))

    def __init__(self,prob):
        self.prob=prob

    def __call__(self,img):
        return self._gaussian_noise(np.array(img),0,random.randint(0,15))

class Random_rotation(object):
    def _randomRotation(self,image,label, mode=Image.NEAREST):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(0, 360)
        return image.rotate(random_angle, mode)

    def __init__(self,prob):
        self.prob=prob

    def __call__(self, img,label):
        return self._randomRotation(img,label)

class Random_crop(object):
    def _randomCrop(self,img,label):
        width, height = img.size
        x, y = random.randint(0, width - 512), random.randint(0, height - 512)
        region = [x, y, x + self.width, y + self.height]
        return img.crop(region),label.crop(region)

    def __init__(self,height,width):
        self.height=height
        self.width=width

    def __call__(self,img,label):
        assert img.size==label.size,"img should have the same shape as label"
        width,height=img.size
        assert height>=self.height and width>=self.width,"Cropimg should larger than origin"
        return self._randomCrop(img,label)

class Random_img_crop(object):
    def _random_Crop(self,img):
        #width = img.shape[1]
        #height = img.shape[2]
        width, height = img.size
        x, y = random.randint(0, width - 512), random.randint(0, height - 512)
        region = [x, y, x + self.width, y + self.height]
        return img.crop(region)

    def __init__(self,height,width):
        self.height=height
        self.width=width

    def __call__(self,img):

        #width = img.shape[1]
        #height = img.shape[2]
        width, height = img.size
        assert height>=self.height and width>=self.width,"Cropimg should larger than origin"
        return self._random_Crop(img)

class Gamma_adjust(object):
    def __int__(self, img):
        self.img = img

    def gamma_adjust(self, img):
        im = img.astype(np.float32)
        im = 255. * 1 * np.power(im / 255., 0.8)
        im = im.clip(min=0., max=255.)
        return im.astype(img.dtype)
    def __call__(self,img):
        """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """

        return self.gamma_adjust(img)





