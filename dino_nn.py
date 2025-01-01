import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class Dino:
    def __init__(self, name_model="dino_vits16"):
        self.dino_model = torch.hub.load('facebookresearch/dino:main', name_model)
        self.dino_model.eval()
        self.n_feats = 384
        self.patch_size = self.dino_model.patch_embed.patch_size

    def extract_dino_feat_from_img(self, image):
        desired_size_img_h = 480
        desired_size_img_w = 640
        desired_size_img_h_cropped = (desired_size_img_h//self.patch_size) * self.patch_size
        desired_size_img_w_cropped = (desired_size_img_w//self.patch_size) * self.patch_size

        transform_img = transforms.Compose([
                        transforms.Resize((desired_size_img_h, desired_size_img_w)),
                        transforms.CenterCrop((desired_size_img_h_cropped, desired_size_img_w_cropped)), # should be multiple of model self.patch_size
                        transforms.ToTensor(), # transforms from 0 - 255 to 0 - 1
                            ])

        image_tensor = transform_img(image).unsqueeze(0)
        with torch.no_grad():
            assert image_tensor.shape[2] % self.patch_size == 0
            assert image_tensor.shape[3] % self.patch_size == 0

            feats = self.dino_model.get_intermediate_layers(image_tensor)[0].clone() # 1 x (img.shape[2] // self.patch_size * img.shape[3] // self.patch_size) + 1) x feats_dim
            feats_h = image_tensor.shape[2] // self.patch_size
            feats_w = image_tensor.shape[3] // self.patch_size

            b, h, w, d = feats.shape[0], feats_h, feats_w, feats.shape[-1]
            feats = feats[:, 1:, :].reshape(b, h, w, d) # 0 is the class token
            image_feat_raw = feats.permute(0, 3, 1, 2)

        return image_feat_raw.squeeze().numpy()

    def process_img_and_get_dino_feat(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_features_dino = self.extract_dino_feat_from_img(img) # 384 x 30 x 40
        img_lower_res = img.resize((img_features_dino.shape[2], img_features_dino.shape[1]), Image.LANCZOS) # same size as the Dino image
        img_lower_res = np.array(img_lower_res)
        return img_features_dino, img_lower_res
