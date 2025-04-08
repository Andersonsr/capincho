import torch
from PIL import Image, ImageFile
from abc import ABC, abstractmethod
import clip
import os

try:
    import open_clip
except ImportError:
    print('Open CLIP not available')

# handling big images
Image.MAX_IMAGE_PIXELS = 999999999
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Model(ABC):
    def __init__(self, device):
        assert 'MODEL_CACHE' in os.environ, 'MODEL_CACHE environment variable is not defined'
        self.download_root = os.environ['MODEL_CACHE']
        self.backbone = None
        self.vision_preprocess = None
        self.language_preprocess = None
        self.device = device

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def visual_embedding(self, image_path):
        pass

    @abstractmethod
    def language_embedding(self, text):
        pass

    def similarity(self, text_features, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).max()


class CLIP(Model):
    def load_model(self, encoder='ViT-L/14'):
        self.backbone, self.vision_preprocess = clip.load(encoder, device=self.device, download_root=self.download_root)

    def visual_embedding(self, image_path):
        with torch.no_grad():
            image = self.vision_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            return self.backbone.encode_image(image)

    def language_embedding(self, text):
        with torch.no_grad():
            text = clip.tokenize(text, context_length=77, truncate=True).to(self.device)
            return self.backbone.encode_text(text)


class OpenCoCa(Model):
    def visual_embedding(self, image_path, ):
        image = Image.open(image_path).convert("RGB")
        image = self.vision_preprocess(image).unsqueeze(0).to(self.device)
        return self.backbone.encode_image(image)

    def language_embedding(self, text):
        text = self.language_preprocess(text)
        # print(text.shape, text)
        text = text[:, :76].to(self.device)

        return self.backbone.encode_text(text)

    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="laion2B-s13B-b90k",
            # pretrained="mscoco_finetuned_laion2B-s13B-b90k",
            device=self.device,
            cache_dir=self.download_root
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')


class OpenCLIP(Model):
    def visual_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.vision_preprocess(image).unsqueeze(0).to(self.device)
        return self.backbone.encode_image(image)

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


if __name__ == "__main__":
    for i in open_clip.list_pretrained():
        if 'L' in i[0]:
            print(i)
