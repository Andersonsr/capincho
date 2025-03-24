import torch
from PIL import Image, ImageFile
from abc import ABC, abstractmethod
import clip

try:
    import open_clip
except ImportError:
    print('Open CLIP not available')

try:
    from capivara.src.models.open_CLIP import OpenCLIP
    from capivara.src.models.open_CLIP_adapter import OpenCLIPAdapter
    from capivara.src.models.open_clip_wrapper import OpenCLIPWrapper
    from capivara.src.utils.capivara_utils import download_pretrained_from_hf
except ImportError:
    print('Capivara not available')

# handling big images
Image.MAX_IMAGE_PIXELS = 999999999
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Model(ABC):
    def __init__(self, device):
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
    def load_model(self, encoder='ViT-L/14', download_root='~/.cache/clip'):
        self.backbone, self.vision_preprocess = clip.load(encoder, device=self.device, download_root=download_root)
        self.language_preprocess = clip.tokenize

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

    def load_model(self, download_root='~/.cache/clip'):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="laion2B-s13B-b90k",
            # pretrained="mscoco_finetuned_laion2B-s13B-b90k",
            device=self.device
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')


class Capivara(Model):
    def visual_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img = self.backbone.image_preprocessor(image).unsqueeze(0).to(self.device)
        return self.backbone.encode_visual(img)

    def language_embedding(self, text):
        tokens = self.backbone.text_tokenizer(text)
        return self.backbone.encode_text(tokens.to(self.device))

    def load_model(self):
        model_path = download_pretrained_from_hf(model_id="hiaac-nlp/CAPIVARA")
        self.backbone = OpenCLIPWrapper.load_from_checkpoint(model_path, strict=False).model
        self.backbone = self.backbone.to(self.device)


class OpenCLIP(Model):
    def visual_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.vision_preprocess(image).unsqueeze(0).to(self.device)
        return self.backbone.encode_image(image)

    def language_embedding(self, text):
        text = self.language_preprocess(text)
        return self.backbone.encode_text(text.to(self.device))

    def load_model(self,  download_root='~/.cache/clip'):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            device=self.device
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')


if __name__ == "__main__":
    from captioningDataset import CaptioningDataset
    # model = Capivara(device="cuda:0")
    # model.load_model()
    #
    # emb = model.language_embedding('texto de texto de texto de texto de')
    # print(emb.shape)
    # emb = model.visual_embedding('./coco retrieval one.png')
    # print(emb.shape)
    for i in open_clip.list_pretrained():
        if 'L' in i[0]:
            print(i)
