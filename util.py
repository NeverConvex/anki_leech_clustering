import os
from PIL import Image, ImageDraw, ImageFont

def text2img(text, write_folder, fname_extra="", ftype="png", overwrite=False, font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", font_size=40):
    write_path = f"{write_folder}{text}{fname_extra}.{ftype}"
    if not os.path.exists(write_path) or overwrite:
        font = ImageFont.truetype(
            font_path,
            size=font_size
        )      
        # Determine drawn img size
        dummy_img = Image.new("RGB", (1, 1)) 
        draw = ImageDraw.Draw(dummy_img)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        width = right - left
        height = bottom - top

        # Generate + save img at appropriate size
        img = Image.new("RGBA", (width, height), "white")
        draw = ImageDraw.Draw(img)
        # Offset by (-left, -top) because the bounding box isn't necessarily (0,0)
        draw.text((-left, -top), text, font=font, fill="black")
        img.save(write_path)
    else:
        print(f"{write_path} already exists and overwrite={overwrite}. Returning write_path...")
    return write_path

def main():
    raise NotImplementedError(f"Only meant to be used to call individual utility fxns.")

if __name__ == "__main__":
    main()
