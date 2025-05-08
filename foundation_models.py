import logging
import torch
from PIL import Image, ImageFile
from abc import ABC, abstractmethod
import clip
import os
import cv2

try:
    import Llip.llip.open_clip as llip
except ImportError:
    print('Llip not installed')

try:
    import open_clip
except ImportError:
    print('Open CLIP not available')


# handling big images
Image.MAX_IMAGE_PIXELS = 999999999
ImageFile.LOAD_TRUNCATED_IMAGES = True

# logging
logger = logging.getLogger('captioning')


class Model(ABC):
    def __init__(self, device):
        assert 'MODEL_CACHE' in os.environ, 'MODEL_CACHE environment variable is not defined'
        self.download_root = os.environ['MODEL_CACHE']
        self.backbone = None
        self.vision_preprocess = None
        self.language_preprocess = None
        self.device = device
        self.dim = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def language_embedding(self, text):
        pass

    def patch_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        image = Image.fromarray(image).convert('RGB')
        w, h = image.size
        logger.debug('original dim {}x{}'.format(image.size[0], image.size[1]))
        if w != h:
            image = image.crop((0, 0, min([h, w]), min([h, w])))
            logger.debug('patch dim {}x{}'.format(image.size[0], image.size[1]))

        w, h = image.size
        crops = []
        resize_dim = min(w//2, self.dim)
        crops.append(image.crop((0, 0, w//2, h//2)).resize((resize_dim, resize_dim)))
        crops.append(image.crop((w//2, 0, w, h//2)).resize((resize_dim, resize_dim)))
        crops.append(image.crop((0, h//2, w//2, h)).resize((resize_dim, resize_dim)))
        crops.append(image.crop((w//2, h//2, w, h)).resize((resize_dim, resize_dim)))
        logger.debug('patch resized dim {}x{}'.format(crops[0].size[0], crops[0].size[1]))

        return crops

    def similarity(self, text_features, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).max()

    def patch_embedding(self, image_path):
        patches_embeddings = []
        patches = self.patch_image(image_path)
        with torch.no_grad():
            for image in patches:
                image = self.vision_preprocess(image).unsqueeze(0).to(self.device)
                patches_embeddings.append(self.backbone.encode_image(image))

        return torch.stack(patches_embeddings)

    def visual_embedding(self, image, resize=False, crop=False):
        if type(image) is str:
            image = self.prepare_image(image, resize, crop)
            return self.backbone.encode_image(image)
        else:
            return self.backbone.encode_image(image)

    # TODO: adicionar crop central para imagens que nao sao quadradas
    def prepare_image(self, image_path, resize=False, crop=False):
        # opencv works better when reading big images
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        image = Image.fromarray(image).convert('RGB')
        if resize and image.size[0] > self.dim:
            logger.debug('resizing image, original size: {}x{}'.format(image.size[0], image.size[1]))
            image = image.resize((self.dim, self.dim))
            logger.debug('resize image: {}x{}'.format(image.size[0], image.size[1]))

        return self.vision_preprocess(image).unsqueeze(0).to(self.device)


class CLIP(Model):
    def load_model(self):
        self.backbone, self.vision_preprocess = clip.load('ViT-L/14',
                                                          device=self.device,
                                                          download_root=self.download_root)
        self.dim = 224

    def language_embedding(self, text):
        with torch.no_grad():
            text = clip.tokenize(text, context_length=77, truncate=True).to(self.device)
            return self.backbone.encode_text(text)


class OpenCoCa(Model):
    def language_embedding(self, text):
        text = self.language_preprocess(text)
        # print(text.shape, text)
        text = text[:, :76].to(self.device)

        return self.backbone.encode_text(text)

    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="laion2B-s13B-b90k",
            device=self.device,
            cache_dir=self.download_root
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')
        self.dim = 224


class SigLIP(Model):
    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-SO400M-14-SigLIP-384",
            pretrained="webli",
            device=self.device,
            cache_dir=self.download_root
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')
        self.dim = 384

    def language_embedding(self, text):
        text = self.language_preprocess(text)
        return self.backbone.encode_text(text.to(self.device))


class OpenCLIP(Model):
    def language_embedding(self, text):
        text = self.language_preprocess(text)
        return self.backbone.encode_text(text.to(self.device))

    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            device=self.device,
            cache_dir=self.download_root
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')
        self.dim = 224


class Llip(Model):
    def load_model(self):
        pass

    def language_embedding(self, text):
        pass


model_dict = {'coca': OpenCoCa,
              'clip': CLIP,
              'openclip': OpenCLIP,
              'sig-lip': SigLIP}


if __name__ == "__main__":
    # model = CLIP('cuda:0')
    # model.load_model()
    # crops = model.patch_image('./plots/cars result.png')
    # embeds = model.patch_embedding(crops)
    # logger = logging.getLogger('captioning')
    # logging.basicConfig(level=logging.DEBUG)

    models = llip.list_models()
    print(models)

