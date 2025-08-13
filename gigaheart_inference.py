import os
import timm
import torch
import argparse
import numpy as np

from PIL import Image
from torchvision import transforms
from torch import nn
from timm.layers import Mlp
from utils import process_ct


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)
    
transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        MaybeToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def get_finetune_params():

    parser = argparse.ArgumentParser(description='Calculate GigaPath Volumeric Embeddings')
    parser.add_argument('--input_ct',       type=str, default='example/CTseg_1_raw.tif', help='Input CT file path')
    parser.add_argument('--save_dir',      type=str, default='output', help='Directory to save the output embeddings')

    return parser.parse_args()


def main():
    args = get_finetune_params()

    # data pre-processing
    images, _, _ = process_ct(args.input_ct)
    images_torch = transform(images)

    # visualize the input
    # image_slice = images[64].cpu().permute(1, 2, 0).numpy() # Convert to HWC format
    # image = Image.fromarray((image_slice * 255).astype(np.uint8))
    # image.save('released_model/example/processed_input_image.png')
    gigaheart = timm.create_model("hf_hub:gigaheart/gigaheart-test", act_layer=nn.GELU, mlp_layer=Mlp, pretrained=True)
    gigaheart = gigaheart.cuda()
    # run inference
    gigaheart.eval()
    with torch.no_grad():
        embedding_output = gigaheart(images_torch.cuda())
        print(f'Embedding output shape: {embedding_output.shape}')

    # save the output embeddings
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, 'gigaheart_embeddings.pt')
    torch.save(embedding_output.cpu(), output_path)


if __name__ == '__main__':
    main()