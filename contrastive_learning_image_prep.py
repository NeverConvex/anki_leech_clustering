# NOTE WORK-IN-PROGRESS
"""
This is a simple script for pre-generating contrastive transformations of preexisting png's depicting all possible NotoSansCJK-representable kanji/kana.
"""

import glob, os, pathlib

from torchvision.transforms import GaussianBlur
from torchvision.transforms import RandomRotation
from torchvision.transforms import RandomResizedCrop
import numpy as np

from PIL import Image

import numpy as np

#noise = np.random.normal(0, 10, img.shape)
#noisy = img + noise

def generateImages(test=-1, overwrite=False, verbosity_rate=1000):
    """
    Used to populate these subdirectories, from the base images in notosans_cjk_chars_as_imgs:
        notosans_cjk_chars_as_imgs/GaussianBlur/   notosans_cjk_chars_as_imgs/ResizeCrop/  notosans_cjk_chars_as_imgs/Sobel/
        notosans_cjk_chars_as_imgs/GaussianNoise/  notosans_cjk_chars_as_imgs/Rotation/
    """
    augmentation2method =   {
                                "GaussianBlur"  : lambda img: GaussianBlur(kernel_size=33, sigma=(150.0, 500.0))(img),
                                "Rotation"      : lambda img: RandomRotation(70)(img),
                                "GaussianNoise" : None,
                                "ResizeCrop"    : lambda img: RandomResizedCrop(list(reversed(img.size)))(img),
                                "Sobel"         : None,
                            }

    source_img_paths = glob.glob("notosans_cjk_chars_as_imgs/*.png")
    test = len(source_img_paths) if test<=0 else test
    for img_index, src_img_path in enumerate(source_img_paths[:test]):
        for augmentation_name, augmentation_fxn in augmentation2method.items():
            if augmentation_fxn is not None:
                new_img_path = f"notosans_cjk_chars_as_imgs/{augmentation_name}/" + pathlib.Path(src_img_path).stem + ".png"
                if not os.path.exists(new_img_path) or overwrite:
                    src_img = Image.open(src_img_path)
                    new_img = augmentation_fxn(src_img)
                    new_img.save(new_img_path)
                    if img_index % verbosity_rate == 0:
                        print(f"src_img size (width, height): {src_img.size} new_img: {new_img.size} ({src_img_path} -> {new_img_path})")
                else:
                    print(f"{new_img_path} already exists and overwrite={False}. Leaving as is.")

def convertRWtoBW(test=-1, verbosity_rate=1000):
    """
        The originally-generated images were written in red/white --- which was a bit silly, and kind of surprising, since our call in util.py was:

            font = ImageFont.truetype( ) ...
            img = Image.new("RGBA", (width, height), "white")). 
            draw = ImageDraw.Draw(img) ...
            draw.text((-left, -top), text, font=font, fill="black")

        which I would have expected to yield white-on-black or black-on-white, but certainly not red-on-white. 

        Maybe this is a bit of a happy accident, though, as it gives us slightly 'dirtier' images after this fxn converts the r-on-w imgs to b-on-w images via
        simpler threshold comparison, a bit like an accidental extra augmentation. (Separately, we'll also generate completely clean images to serve as ideal 
        ground truth.)
    """
    for augmentation_name in ("GaussianBlur", "Rotation", "ResizeCrop"):
        imgs_root = f"notosans_cjk_chars_as_imgs/{augmentation_name}/downsized"
        source_img_paths = glob.glob(f"{imgs_root}/*.png")
        if len(source_img_paths)>0:
            os.makedirs(os.path.dirname(f"{imgs_root}/bw/"), exist_ok=True)
        num2process = len(source_img_paths) if test<=0 else test
        for img_index, src_img_path in enumerate(source_img_paths[:num2process]):
            img = Image.open(src_img_path).convert("RGBA")
            arr = np.array(img)
            bw = (arr[:, :, 1] > 128).astype(np.uint8)
            Image.fromarray(bw * 255, mode="L").save(f"{imgs_root}/bw/" + pathlib.Path(src_img_path).stem + f".pgm")
            if img_index % verbosity_rate == 0:
                print(f"For {augmentation_name}, converted img # {img_index} from {src_img_path} to bw image in ../bw/ subdir")

if __name__ == "__main__":
    generateImages()
