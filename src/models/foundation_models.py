import logging
import torch
from PIL import Image, ImageFile
from abc import ABC, abstractmethod
import clip
import os, sys
import cv2
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from models.longclip import longclip
from transformers import AutoImageProcessor, AutoModel
from util import learnable_parameters


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


class FoundationModel(ABC):
    def __init__(self, device):
        assert 'MODEL_CACHE' in os.environ, 'MODEL_CACHE environment variable is not defined'
        self.download_root = os.environ['MODEL_CACHE']
        self.backbone = None
        self.vision_preprocess = None
        self.tokenizer = None
        self.device = device
        self.dim = None

    @abstractmethod
    def load_model(self):
        pass

    def language_embedding(self, text):
        with torch.no_grad():
            text = self.tokenizer(text).to(self.device)
            return self.backbone.encode_text(text)

    def visual_embedding(self, image, resize=False, ):
        with torch.no_grad():
            if type(image) is str:
                image = prepare_image(image, resize, dim=self.dim)
                image = self.vision_preprocess(image).unsqueeze(0)

            elif type(image) is list:
                if len(image) > 1:
                    image = [self.vision_preprocess(im) for im in image]
                    image = torch.stack(image).to(self.device)
                    logging.debug('Image embeddings shape: {}'.format(image.shape))

                elif len(image) == 1:
                    image = self.vision_preprocess(image[0]).unsqueeze(0)

                else:
                    raise IndexError('Image list is empty')

            else:
                image = self.vision_preprocess(image).unsqueeze(0)

            return self.backbone.encode_image(image.to(self.device))

    def similarity(self, text_features, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).max()

    def grid_features(self, image_path):
        pass


def prepare_image(image_path, resize=False, dim=224):
    # opencv works better when reading big images
    # print(image_path)
    # crop image to 1x1 ratio and then resize
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    image = Image.fromarray(image).convert('RGB')
    h, w = image.size

    if h != w:
        min_dim = min(h, w)
        center = (w // 2, h // 2)
        image = image.crop((center[0] - min_dim // 2, center[1] - min_dim // 2,
                            center[0] + min_dim // 2, center[1] + min_dim // 2))

    if resize and image.size[0] > dim:
        logger.debug('resizing image, original size: {}x{}'.format(image.size[0], image.size[1]))
        image = image.resize((dim, dim))
        logger.debug('resize image: {}x{}'.format(image.size[0], image.size[1]))

    return image


class LongCLIP(FoundationModel):
    def load_model(self):
        self.backbone, self.vision_preprocess = longclip.load("./checkpoints/longclip-B.pt", device=self.device)
        self.dim = 224

    def language_embedding(self, text):
        with torch.no_grad():
            text = longclip.tokenize(text).to(self.device)
            return self.backbone.encode_text(text)


class CLIP(FoundationModel):
    def load_model(self):
        self.backbone, self.vision_preprocess = clip.load('ViT-L/14',
                                                          device=self.device,
                                                          download_root=self.download_root)
        self.dim = 224

    def language_embedding(self, text):
        with torch.no_grad():
            text = clip.tokenize(text, context_length=77, truncate=True).to(self.device)
            return self.backbone.encode_text(text)


class OpenCoCa(FoundationModel):
    def language_embedding(self, text):
        text = self.tokenizer(text)
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
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.dim = 224


class SigLIP_384(FoundationModel):
    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-SO400M-14-SigLIP-384",
            pretrained="webli",
            device=self.device,
            cache_dir=self.download_root
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.dim = 384


class SigLIP_512(FoundationModel):
    def load_model(self):
        self.backbone, self.vision_preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512',
                                                                   device=self.device,
                                                                   cache_dir=self.download_root)
        self.tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-512')
        self.dim = 512


class OpenCLIP(FoundationModel):
    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            device=self.device,
            cache_dir=self.download_root
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.dim = 224


class DinoV3(FoundationModel):
    def load_model(self):
        pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.vision_preprocess = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name,
        )
        self.backbone.to(self.device)

    def visual_embedding(self, image, resize=False, ):
        if type(image) is str:
            inputs = self.vision_preprocess(images=Image.open(image), return_tensors="pt")

        elif type(image) is list:
            if len(image) > 1:
                inputs = self.vision_preprocess(images=image, return_tensors="pt")

            elif len(image) == 1:
                inputs = self.vision_preprocess(images=image[0], return_tensors="pt")

            else:
                raise IndexError('Image list is empty')

        else:
            inputs = self.vision_preprocess(images=image, return_tensors="pt")

        output = self.backbone(**inputs.to(self.device))
        return output.pooler_output

    def language_embedding(self, text):
        if type(text) == str:
            return torch.ones((1, 768))

        if type(text) == list:
            return torch.ones((len(text), 768))


class DinoV3_L(DinoV3):
    def load_model(self):
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.vision_preprocess = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name,
        )
        self.backbone.to(self.device)


class DinoV2(FoundationModel):
    def load_model(self):
        model_name = 'facebook/dinov2-base'
        self.vision_preprocess = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name,)
        self.backbone.to(self.device)

    def visual_embedding(self, image, resize=False, ):
        if type(image) is str:
            inputs = self.vision_preprocess(images=Image.open(image), return_tensors="pt")

        elif type(image) is list:
            if len(image) > 1:
                inputs = self.vision_preprocess(images=image, return_tensors="pt")

            elif len(image) == 1:
                inputs = self.vision_preprocess(images=image[0], return_tensors="pt")

            else:
                raise IndexError('Image list is empty')

        else:
            inputs = self.vision_preprocess(images=image, return_tensors="pt")

        inputs.to(self.device)
        output = self.backbone(**inputs)
        return output.pooler_output

    def language_embedding(self, text):
        if type(text) == str:
            return torch.ones((1, 768))

        if type(text) == list:
            return torch.ones((len(text), 768))


model_dict = {'coca': OpenCoCa,
              'clip': CLIP,
              'openclip': OpenCLIP,
              'longclip': LongCLIP,
              'siglip-384': SigLIP_384,
              'siglip-512': SigLIP_512,
              'dinov2': DinoV2,
              'dinov3': DinoV3,
              'dinov3-L': DinoV3_L}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = 'texto de teste'

    for key in ['dinov2', 'dinov3', 'dinov3-L']:
        model = model_dict[key](device)
        model.load_model()
        if 'dino' in key:
            print(key, learnable_parameters(model.backbone))

        else:
            # print(model.backbone)
            print(key, learnable_parameters(model.backbone.visual))


    # embeddings = model.visual_embedding('C:/Users/Usuario/PycharmProjects/capincho/src/models/img.png')
    # embeddings = model.language_embedding(text)
    # print(embeddings.shape)



