# Standard libraries
import glob, pathlib, json, warnings, sys, os
import numpy as np
from collections import defaultdict
from copy import deepcopy

# Non-standard libraries
import scipy, sklearn, manga_ocr, fire

# Project-specific imports
import util

def mangaOcrDirectCluster(    read_glob_expr, img_write_folder, acts_write_folder, test_amount=-1, overwrite=False, comment_char='#', sep='\t',
                            num_clusters=8):
    """
        This functions reads words/expressions (presumably Japanese tokens accumulated as Anki flashcard leeches), feeds these to the manga-ocr
        (https://github.com/kha-white/manga-ocr) pre-trained transformers-based vision encoder/decoder neural network, extracts the activations
        observed in the network for each word/expression, and returns clusters of the target words/expressions on the basis of similarities in
        these activations.

        The general idea is that the activations serve as a convenient numerical representation of notable visual features in
        the targeted words/expressions, and that visually similarity is often an important driver of a flashcard being difficult to memorize
        (since it may be confused with visually similar competitors) and so becoming a leech.

        [Presumably semantic similarity and phonetic similarity are also drivers of flashcards becoming leeches. I intend to examine these, and
        methods of clustering based on the simutlaneous consideration of all three kinds of similarity, in the future.]
    """

    # Setup
    pathlib.Path(img_write_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(acts_write_folder).mkdir(parents=True, exist_ok=True)
    leech_deck_paths    = glob.glob(read_glob_expr)
    mocr                = manga_ocr.MangaOcr()   
    leeches             = set()

    # Get desired leech tokens
    for leech_deck in leech_deck_paths:
        with open(leech_deck, 'r', encoding='utf8') as rf:
            lines           = rf.readlines() 
            lines           = [l for l in lines if l[0] != comment_char]
            leech_tokens    = [l.split(sep)[0] for l in lines]
            leeches         = leeches.union( set(leech_tokens) )
    leeches = sorted(list(leeches))

    # For each leech token, generate an image of it, feed that to manga-ocr, and extract the observed encoder activations
    test_amount = len(leeches) if test_amount < 0 else test_amount
    leech2acts  = {}
    for leech in leeches[:test_amount]:
        image_path = util.text2img(  leech, img_write_folder, fname_extra="", ftype="png", overwrite=overwrite,
                                font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", font_size=40)
        
        image_fname_wo_ext = pathlib.Path(image_path).stem 
        print(f"Checking for existence of files matching glob expr: {acts_write_folder}{image_fname_wo_ext}*.npy") # TODO 9/16/2025 replace w/ more conservative check
        existing_act_fpaths = glob.glob(f"{acts_write_folder}{image_fname_wo_ext}*.npy")
        if len(existing_act_fpaths) > 0 and not overwrite:
            print(f"Skipping input image at: {image_path} Output file: {acts_write_folder}{image_fname_wo_ext}.npy already exists (set overwrite=True to ignore)")
            if len(existing_act_fpaths) > 1:
                warnings.warn(f"WARNING: For {leech}, found multiple existing activations files: {existing_act_fpaths}. Only using: {existing_act_fpaths[0]}")
            acts = np.load(existing_act_fpaths[0])
            leech2acts[leech] = acts.flatten()
        else:
            activations_encoder_output = []
            handles = []
            def encoder_output_activations_hook(module, inputs, outputs):
                activations_encoder_output.append(outputs)#.detach())
            handles.append(mocr.model.encoder.register_forward_hook(encoder_output_activations_hook))

            text = mocr(image_path)
            print(f"mocr thinks the text in {image_path} is: {text}")
            for h in handles:
                h.remove()

            for i, act in enumerate(activations_encoder_output):
                pooler_output = act.pooler_output

                np_act = pooler_output.float().numpy()
                num_scalars     = np.prod(np_act.shape)
                print(f"mangaocr encoder act # {i}:\n\t{type(np_act)} {np_act} {num_scalars}")
                print(f"\n\tnp_act shape: {np_act.shape}")
                with open(f".npy", 'wb') as wf:
                    np.save(f"{acts_write_folder}{image_fname_wo_ext}_{i}.npy", np_act)
            if len(activations_encoder_output) > 1:
                warnings.warn(f"WARNING: For {leech}, found multiple manga-ocr encoder pooler activations. Only using the first...")
            leech2acts[leech] = np_act.flatten()

    # Cluster based on the observed manga-ocr encoder activations
    leeches = list(leech2acts.keys())
    activations = list(leech2acts.values()) # NOTE keys/values order will always match: https://stackoverflow.com/a/835430/4286018
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters).fit(activations)
    for cluster_num in range(num_clusters):
        cluster_members = [leeches[int(i)] for i in np.where(kmeans.labels_==cluster_num)[0]]
        print(f"Cluster # {cluster_num}\n\t{cluster_members}")

if __name__ == "__main__":
    # Example cmd-line run command:
    # python manga_ocr_direct_cluster.py mangaOcrDirectCluster --read_glob_expr="example_leeches.txt" --img_write_folder="leechesDirectTestImgs/" --acts_write_folder="leechesDirectTestActs/"
    fire.Fire()
