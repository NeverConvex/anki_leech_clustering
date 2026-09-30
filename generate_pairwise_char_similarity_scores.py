# NOTE WORK-IN-PROGRESS
"""
This script is not actively in use, but has some example code for storing activations from a certain public, small-enough-to-run-locally VLM stored on HuggingFaces,
and some auxiliary code for computing and storing the list of components ('parts') Jisho considers a Japanese kanji to possess. These might be useful for
clustering work in the future, so we're keeping this script around for reference.
"""

### Must be executed from an environment with dependencies suitable for running the SmolVLM-256M-Instruct model TODO 9/4/2025 insert Hugging-Faces link
import torch, pathlib, glob, time, os, scipy, json, urllib, requests, warnings#, fire # TODO install fire in SmolVLM conda env
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText#AutoModelForVision2Seq
from transformers.image_utils import load_image
from collections import defaultdict

import manga_ocr

def getKanjiPartsFromJisho(kanji):
    """
        This fxn based loosely on jisho-api's straightforward use of BS4; see especially lines
            https://github.com/pedroallenrevez/jisho-api/blob/b1d9ec09e999bb35b9903a1023cf8bc9c74107c1/jisho_api/kanji/request.py#L289
            https://github.com/pedroallenrevez/jisho-api/blob/b1d9ec09e999bb35b9903a1023cf8bc9c74107c1/jisho_api/kanji/request.py#L102
        Could have probably just used jisho-api directly, but we have a much narrower use-case in mind
    """
    from bs4 import BeautifulSoup
    kana = "あアかカさサたタなナはハまマやヤらラわワいイきキしシちチにニひヒみミりリうウくクすスつツぬヌふフむムゆユるルをヲえエけケせセてテねネへヘめメれレおオこコそソとトのノほホもモよヨろロんンがガざザだダばバぱパぎギじジびビぴピぐグずズづヅぶブぷプげゲぜゼでデべベぺペごゴぞゾどドぼボぽポゃャゅュょョ"
    parts = [kanji]
    if kanji not in kana:
        try: # TODO 9/19/2025 check what happens if result isn't found, add various robustness checks, make except more specific
            url = "https://jisho.org/search/" + urllib.parse.quote(f"{kanji}" + " #kanji")
            r = requests.get(url).content
            soup = BeautifulSoup(r, "html.parser")
            res = soup.find_all("div", {"class": "radicals"})
            parts_res = res[1].find_all("a")
            parts = [p.text for p in parts_res]
        except Exception as e:
            warnings.warn(f"Failed to scrape {kanji} from Jisho. Error: {e}")
    else:
        warnings.warn(f"{kanji} is not a kanji, but rather kana. Returning it as its own 'parts'....")
    return parts

def getCharactersFromCsv(   csv_read_path, write_path, comment_character="#", sep="\t", word_column_index=0):
    """
        Reads target csv from <csv_read_path>; splits each line based on <sep>, except for lines beginning with <comment_character>, which are ignored. Takes the
        entry in the split line at <word_column_index> to be a word of interest, and further splits it into individual characters. The union of all such characters
        is returned as the characters of interest in the csv.
    """
    lines = open(csv_read_path, 'r', encoding='utf-8').readlines()
    lines = [l for l in lines if l[0] != comment_character]
    lines = [l.split(sep)[0] for l in lines]
    chars = ''.join(list(set([c for word in lines for c in word if c not in (' ', ')', '(')])))
    with open(write_path, 'w', encoding='utf-8') as wf:
        wf.write(chars)
    print(f"Wrote {len(chars)} to {write_path}. Returning them to calling fxn as well...")
    return chars

def getKanjiParts(      txt_read_path, json_write_path):
    """
        txt_read_path is assumed to be a single-line file containing kanji of interest. This fxn retrieves the parts (something like Jisho's alternative to
        traditional Japanese character components/radicals/部首) for each kanji and returns them in a kanji->[parts] dictionary. This is later used to implement
        a very simple, heuristic, non-neural-network-based measure of (mostly visual) similarity b/w kanji.
    """
    kanjis = open(txt_read_path, 'r', encoding='utf-8').readline().strip()
    kanji2parts = defaultdict(list)    
    print(F"Kanjis: {kanjis}")
    for i, kanji in enumerate(kanjis):
        print(f"Gathering parts for kanji # {i}: {kanji}....")
        time.sleep(0.5)
        kanji_parts = getKanjiPartsFromJisho(kanji)
        kanji2parts[kanji] = kanji_parts
        print(f"Retrieved Kanji parts: {kanji_parts}")
    with open(json_write_path, 'w', encoding='utf-8') as wf:
        json.dump(kanji2parts, wf)
    print(f"Write {len(kanji2parts)} Kanjis' parts to {json_write_path}. Returning them to calling fxn as dict as well...")
    return kanji2parts

def getMangaOcrVisualActivations(   read_glob_expr, write_folder, test_amount=-1, max_new_tokens=500,
                                    activation_layer_types=["visual_encoder_layers", "visual_embedding_patch", "visual_embedding_position"],
                                    overwrite=False, normalized=True, rounded=True, demoded=True, write_raw=False):
    """
        Finds imgs w/ glob.glob via read_glob_expr, feeds these to the manga-ocr (https://github.com/kha-white/manga-ocr) transformer-based vision encoder-decoder
        neural network, extracts various activations from them, and stores these to be used as numerical representation of notable visual features in the
        targeted images.
    """
    image_paths         = glob.glob(read_glob_expr)
    image_paths         = [p for p in image_paths if "base_image" not in p]
    mocr = manga_ocr.MangaOcr()   
    pathlib.Path(write_folder).mkdir(parents=True, exist_ok=True)
    for image_path in image_paths[:test_amount]:
        image_fname_wo_ext = pathlib.Path(image_path).stem 
        print(f"Checking for existence of files matching glob expr: {write_folder}{image_fname_wo_ext}*.npy") # TODO 9/16/2025 replace w/ more conservative check
        if len(glob.glob(f"{write_folder}{image_fname_wo_ext}*.npy")) > 0 and overwrite==False:
            print(f"Skipping input image at: {image_path} Output file: {write_folder}{image_fname_wo_ext}.npy already exists (set overwrite=True to force overwrite anyway)")
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
                last_hidden_state = act.last_hidden_state
                pooler_output = act.pooler_output
                hidden_states = act.hidden_states   # if requested
                attentions = act.attentions      

                np_act = pooler_output.float().numpy()
                num_scalars     = np.prod(np_act.shape)
                #sparse_act2save = scipy.sparse.csr_matrix(np_act.flatten())
                print(f"mangaocr encoder act # {i}:\n\t{type(np_act)} {np_act} {num_scalars}")
                print(f"\n\tnp_act shape: {np_act.shape}")
                with open(f".npy", 'wb') as wf:
                    np.save(f"{write_folder}{image_fname_wo_ext}_{i}.npy", np_act)

def getAllImageActivations(     read_glob_expr, write_folder, test_amount=-1, max_new_tokens=500,
                                activation_layer_types=["visual_encoder_layers", "visual_embedding_patch", "visual_embedding_position"],
                                overwrite=False, normalized=True, rounded=True, demoded=True, write_raw=False):
    """
        Fxn for computing and storing to file activations associated with (selected) CNN layers from SmolVLM-256M-Instruct, when given input from each image
        corresponding to an image_path in image_paths. Activations are written to write_folder, with a single write_folder file
        for each input image_path. Args:
            - read_glob_expr    :       expr interpreted by glob to find input image paths
            - write_folder      :       folder into which activations will be saved in numpy .npy format
            - test_amount       :       # of image_paths processed will be <= test_amount, unless it is -1 (interpreted as no limit)
            - write_raw         :       True to save raw activations w/o renormalizing to [0, 1] and then demoding; False to save w/ that transofmration
    """
    # NOTE TODO 9/30/2026 SmolVLM takes ~30 secs per inference + activations read & write, which makes this very slow to use during testing. Also do not
    # really have good reasons to expect SmolVLM's visual encoding layers to be more meaningful than MangaOCR (where inference/activation-retrieval is near-instant),
    # a small custom-trained FCN, etc. So, this is largely superseded by MangaOCR use and FCN development for now, but intend to revisit SmolVLM once I'm happier with 
    # basic clustering performance, and maybe also for trying to use semantic similarity for clustering, rather than just visual & phonetic
    image_paths         = glob.glob(read_glob_expr)
    image_paths         = [p for p in image_paths if "base_image" not in p]
    # Before saving, we normalize to [0, 1], then round, then subtract modal value from activation vectors,
    activation_modes    = {} # for sparse forrmat; saving these allows reconstructing original vector
    activation_maxes    = {} # (up to rounding and float loss)
    activation_mins     = {}
    print(f"Found {len(image_paths)} total input image paths")
    pathlib.Path(write_folder).mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    #model = AutoModelForVision2Seq.from_pretrained(
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    # NOTE presumably only the encoder layer is really needed, but while playing originally explored hooks for storing a variety of visual layers
    str2layer                                   = {}
    str2layer["visual_encoder_layers"]          = model.base_model.vision_model.encoder
    str2layer["visual_embedding_patch"]         = model.base_model.vision_model.embeddings.patch_embedding
    str2layer["visual_embedding_position"]      = model.base_model.vision_model.embeddings.position_embedding

    test_amount = len(image_paths) if test_amount == -1 else test_amount 
    print(f"Testing # of image_paths: {test_amount}")
    for image_path in image_paths[:test_amount]:
        image_fname_wo_ext = pathlib.Path(image_path).stem 
        print(f"Checking for existence of files matching glob expr: {write_folder}{image_fname_wo_ext}*.npz") # TODO 9/16/2025 replace w/ more conservative check
        if len(glob.glob(f"{write_folder}{image_fname_wo_ext}*.npz")) > 0 and overwrite==False:
            print(f"Skipping input image at: {image_path} Output file: {write_folder}{image_fname_wo_ext}.npy already exists (set overwrite=True to force overwrite anyway)")
        else:
            print(f"Loading image from: {image_path}")
            image = load_image(image_path)
            # Setup callback for gathering CNN activations, & initialize/empty out corresponding lists (for holding refs to callbacks, and activations observed)
            handles = defaultdict(list)
            activations_embedding_patch         = []
            activations_embedding_position      = []
            activations_encoder_layers          = []
            def embedding_patch_hook(module, inputs, outputs): # NOTE renamed input to inputs to avoid global name conflict (in case this causes unexpected errors)
                activations_embedding_patch.append(outputs)#.detach())
            def embedding_position_hook(module, inputs, outputs): # NOTE renamed input to inputs to avoid global name conflict (in case this causes unexpected errors)
                activations_embedding_position.append(outputs)#.detach())
            def encoder_layers_hook(module, inputs, outputs): # NOTE renamed input to inputs to avoid global name conflict (in case this causes unexpected errors)
                activations_encoder_layers.append(outputs)#.detach())
            # NOTE tried to avoid above duplication w/ a lambda fxn but seemed to only get position embedding, as if others got overwritten...?
            str2hook                                    = {}
            str2hook["visual_encoder_layers"]           = encoder_layers_hook
            str2hook["visual_embedding_patch"]          = embedding_patch_hook
            str2hook["visual_embedding_position"]       = embedding_position_hook
            for layer_type in activation_layer_types:
                print(f"Inserting {layer_type} handles...")
                nn_module = str2layer[layer_type]
                if hasattr(nn_module, "layers"):
                    for l in nn_module.layers:
                        print(f"Operating on l = {l}....")
                        handles[l].append( l.register_forward_hook(str2hook[layer_type]) )
                else:
                    l = nn_module
                    print(f"Operating on l = {l}....")
                    handles[l].append( l.register_forward_hook(str2hook[layer_type]) )

            # NOTE we don't actually care about the question here, but SmolVLM needs a prompt in order to generate CNN activations as a biproduct; this example prompt
            prompt = "Can you describe this image?" # seems as good as anything (question will matter again for language layers; currently only using visual layers)
            messages = [            
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                },
            ]

            # Prepare inputs
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(DEVICE)

            # Generate outputs
            print(f"Invoking model.generate with max_new_tokens: {max_new_tokens}")
            t0 = time.time()
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            print(f"Model generation took: {time.time() - t0} secs")
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            for k in handles.keys():
                for h in handles[k]:
                    h.remove()
            # NOTE we don't really care what the VLM reply was, but printing it out of curiosity (might sometimes serve as a check for e.g. unexpected behavior)
            print(f"For {image_path}, response to \"{prompt}\" was: {generated_texts[0]}")
            print(f"handles keys: {handles.keys()}")

            # Saving activations to various sparse npz's
            for layer_type, activations in zip(activation_layer_types, [activations_encoder_layers, activations_embedding_patch, activations_embedding_position]):
                #np_act = None
                for i, act in enumerate(activations):
                    print(f"For {image_fname_wo_ext} layer_type {layer_type}, working on activations # {i}...")
                    np_act = act[0].float().numpy()
                    num_scalars     = np.prod(np_act.shape)
                    sparse_act2save = scipy.sparse.csr_matrix(np_act.flatten())
                    if not write_raw:
                        np_act_tmp = np_act
                        if normalized:
                            np_act_tmp              = (np_act - np_act.min()) / (np_act.max() - np_act.min()) # Normalize activations to [0, 1]
                        if rounded:
                            np_act_tmp              = np.round(np_act_tmp, decimals=3) # Expect large concentration of activations near original modal activation
                        values, counts              = np.unique(np_act_tmp, return_counts=True)
                        mode_index                  = np.argmax(counts)
                        print(f"Activations for {image_fname_wo_ext}, layer_type {layer_type}, layer # {i}:")
                        print(f"Subtracting modal value {values[mode_index]} at index {mode_index} element-wise, converting to sparse CSR format, then saving...")
                        activation_modes[image_fname_wo_ext]        = str(values[mode_index])
                        activation_maxes[image_fname_wo_ext]        = str(float(np_act.max()))
                        activation_mins[image_fname_wo_ext]         = str(float(np_act.min()))
                        zero_mode_act                               = np_act_tmp - values[mode_index]
                        flat_zero_mode_act                          = zero_mode_act.flatten()
                        sparse_zero_mode_act                        = scipy.sparse.csr_matrix(flat_zero_mode_act)
                        sparse_act2save                             = sparse_zero_mode_act
                        print(f"Saving sparse activtaions to: {write_folder}{image_fname_wo_ext}_layerNo{i}_sparse.npz")
                        scipy.sparse.save_npz(f"{write_folder}{image_fname_wo_ext}_{layer_type}_layerNo{i}_sparse.npz", sparse_act2save, compressed=True)
                        for label, json_data_obj in zip(["activation_modes", "activation_maxes", "activation_mins"], [activation_modes, activation_maxes, activation_mins]):
                            json_outpath = f"{write_folder}{layer_type}_{label}_layerNo{i}.json"
                            if overwrite or not os.path.isfile(json_outpath):
                                with open(json_outpath, 'w', encoding='utf-8') as wf:
                                    json.dump(json_data_obj, wf, ensure_ascii=False)
                            else:
                                with open(json_outpath, 'r', encoding="utf-8") as rf:
                                    data = json.load(rf)
                                data.update(json_data_obj)
                                with open(json_outpath, 'w', encoding='utf-8') as wf:
                                    json.dump(data, wf, ensure_ascii=False)

if __name__ == "__main__":
    """
    Example run cmds:
    """
    #fire.Fire() # TODO 9/4/2025 install fire in SmolVLM conda env
    #getAllImageActivations(     
    #                            read_glob_expr="japanese_chars_as_images/*.png",
    #                            write_folder="japanese_chars_as_images/activations/",
    #                            test_amount=-1,
    #                            overwrite=False
    #                        )
