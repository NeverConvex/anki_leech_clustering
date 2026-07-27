import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

def levenshteinDistance(s1, s2, verbose=False, sub_cost_fxn=None, dtype=np.int32):
    """
    Computes the Levenshtein Distance between strings s1, s2. dist[i, j] in the computed matrix contains the Levenshtein distance
    between the len-i prefix of s1 and the len-j prefix of s2.

    See: https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_full_matrix
    """
    def defaultSubCostFxn(char1, char2):
        if char1 != char2:
            return 1
        return 0
    substituteCostFxn = defaultSubCostFxn if sub_cost_fxn is None else sub_cost_fxn

    dist = np.zeros(shape=(len(s1)+1, len(s2)+1), dtype=dtype)
    dist[:, 0] = np.arange(0, len(s1)+1, 1) # len-k s1 prefix can be turned into the empty string by k deletions
    dist[0, :] = np.arange(0, len(s2)+1, 1) # empty string can be turned into len-k s2 prefix by k insertions
    for j in range(1, len(s2)+1):
        for i in range(1, len(s1)+1):
            #substitute_cost = 1
            #if s1[i-1] == s2[j-1]:
            #    substitute_cost = 0
            substitute_cost = substituteCostFxn(s1[i-1], s2[j-1])
            if verbose:
                print(f"sub cost for {s1[i-1]}, {s2[j-1]}: {substitute_cost}")
            dist[i, j] = min(   
                                dist[i-1, j] + 1,                  # Deletion: matched 1st j w/ 1st i-1, delete unneeded ith
                                dist[i, j-1] + 1,                  # Insertion: matched 1st j-1 w/ 1st i, insert jth
                                dist[i-1, j-1] + substitute_cost   # Substitution: from a match to 1st j-1 w/ 1st i-1, sub ith char to jth char
                            )                                      
    if verbose:
        print(f"On s1={s1}, s2={s2}, Levenshtein Matrix is:")
        print(f"\t{dist}")
    return dist[len(s1), len(s2)]

def main():
    raise NotImplementedError(f"Only meant to be used to call individual utility fxns.")

if __name__ == "__main__":
    main()
