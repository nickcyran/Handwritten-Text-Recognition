from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

class ResizeAndPad:
    def __init__(self, height=64, target_width=1024):
        self.height = height
        self.target_width = target_width

    def __call__(self, img: Image.Image):
        w, h = img.size
        new_w = min(int(self.height * (w / h)), self.target_width)
        resized = img.resize((new_w, self.height), Image.BILINEAR)
        padded = Image.new('L', (self.target_width, self.height), color=255)
        padded.paste(resized, ((self.target_width - new_w) // 2, 0))
        return padded

class MedianFilter:
    def __init__(self, size=3):
        self.size = size

    def __call__(self, img: Image.Image):
        return img.filter(ImageFilter.MedianFilter(self.size))

class EnhanceBrightnessContrast:
    def __init__(self, brightness=1.2, contrast=2.0):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image):
        img = ImageEnhance.Brightness(img).enhance(self.brightness)
        img = ImageEnhance.Contrast(img).enhance(self.contrast)
        return img
    
def dataset_transform():
    return transforms.Compose([
            ResizeAndPad(height=64, target_width=1024),
            transforms.RandomApply([transforms.RandomAffine(degrees=3, shear=10, translate=(0.02, 0.02))], p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
def evaluation_transform():
    return transforms.Compose([
            EnhanceBrightnessContrast(brightness=1.2, contrast=2.0),
            MedianFilter(3),
            ResizeAndPad(height=64, target_width=1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])