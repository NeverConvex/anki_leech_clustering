# NOTE WORK-IN-PROGRESS
"""
This script uses the pre-trained manga-ocr transformer-based visual encoder-decoder network to extract numerical representations of notable visual features in
Japanese words and expressions, then uses this information to cluster them. Currently two clustering approaches are considered:
1. basic k-means clustering directly applied to the visual encoding-layer activations. This works acceptably but still requires user selection of k
2. converting the encoding-layer activations to 'dissimiliarities' by linearly transforming pairwise correlations between visual-encoding-layer activations, then
formulating a combinatorial optimization problem, and trying to solve that problem (currently, with Google ORTOOLS' somewhat unusual branch-and-bound/SAT-solver
hybrid algorithm, CPSAT) to automatically determine number of clusters and cluster membership. This currently struggles to find an optimal solution; considering ways
to warmstart and/or make the (admittedly quite inefficient) model formulation more efficient, or to find valid cuts, and so forth. May also try alternative solvers,
e.g. the various solvers considered in the Mittelman benchmarks (https://plato.asu.edu/bench.html). May also try to determine if a decomposition algorithm (Benders,
etc) could be useful
"""

# Standard libraries
import glob, pathlib, json, warnings, sys, os, random, time
import numpy as np
from collections import defaultdict
from copy import deepcopy

# Non-standard libraries
import scipy, sklearn, manga_ocr, fire
from ortools.sat.python import cp_model
from huggingface_hub import login

# Project-specific imports
import util

def vPrint(msg, verbose=False):
    if verbose:
        print(msg)  

def getHFToken():
    token = open("tokens/hf_read.token").readline().strip()
    login(token=token)

def checkObjVal(activations, new_cluster_cost, cluster_assignments):
    dissims = 1 - (np.corrcoef(activations)+1)/2
    obj_val = 0.0
    for ind0, cluster in enumerate(cluster_assignments):
        cluster_dissims = []
        for ind1, c1 in enumerate(cluster):
            for ind2, c2 in enumerate(cluster[ind1+1:]):
                cluster_dissims.append(float(dissims[c1, c2]))
        diam = max(cluster_dissims) if len(cluster)>1 else 0.0
        obj_val += diam
        print(f"Verification: cluster # {ind0} has diam: {diam}")
    obj_val += new_cluster_cost * len(cluster_assignments)
    print(f"Obj val value independently computed as: {obj_val}")

def getMaxDiamOrder(dissims, verbose=False):
    order = []
    remaining_inds = list(range(dissims.shape[0]))
    dissim_weights = [float(np.max(dissims[i, ::])) for i in range(dissims.shape[0])]
    num2sample = len(remaining_inds)
    while len(order)<num2sample:
        sampled_ind = random.choices(remaining_inds, dissim_weights)[0]
        vPrint(f"From {remaining_inds}, sampled index: {sampled_ind}", verbose=verbose)
        critical_ind = remaining_inds.index(sampled_ind)
        remaining_inds = remaining_inds[:critical_ind]+remaining_inds[critical_ind+1:]
        dissim_weights = dissim_weights[:critical_ind]+dissim_weights[critical_ind+1:]
        order.append(sampled_ind)
    return order

def getMinDiamOrder(dissims, verbose=False):
    order = []
    remaining_inds = list(range(dissims.shape[0]))
    dissim_weights = [1/float(np.max(dissims[i, ::])) for i in range(dissims.shape[0])]
    num2sample = len(remaining_inds)
    while len(order)<num2sample:
        sampled_ind = random.choices(remaining_inds, dissim_weights)[0]
        vPrint(f"From {remaining_inds}, sampled index: {sampled_ind}", verbose=verbose)
        critical_ind = remaining_inds.index(sampled_ind)
        remaining_inds = remaining_inds[:critical_ind]+remaining_inds[critical_ind+1:]
        dissim_weights = dissim_weights[:critical_ind]+dissim_weights[critical_ind+1:]
        order.append(sampled_ind)
    return order

def greedyDissimCluster(activations, leeches, new_cluster_cost=None, num_restarts=1000, verbose=False, order_alg="min_diam"):
    """
        Greedily optimizes
            sum_{c in C} DIAM_c + new_cluster_cost * num_clusters
        by adding leeches to new or existing clusters one-by-one.

        Also used to generate a warmstart feasible solution by cpsatCluster.

        order_alg:
            random          : uses random permutation of leech indices in each greedy restart
            max_diam        : also uses random permutation, but weighted proportional to max diam
    """
    t0 = time.time()
     # Setup
    dissims = 1 - (np.corrcoef(activations)+1)/2
    new_cluster_cost = np.percentile(dissims, 10) if new_cluster_cost is None else new_cluster_cost
    best_cluster_assignments, min_obj_val = None, np.inf
    for restart_index in range(num_restarts):
        obj_fxn_val = 0.0
        shuffle_index = 0
        if order_alg=="random":
            order = list(range(len(leeches)))
            random.shuffle(order)
        elif order_alg=="max_diam":
            order = getMaxDiamOrder(dissims)
        elif order_alg=="min_diam":
            order = getMinDiamOrder(dissims)
        else:
            raise ValueError(f"Greedy ordering algorithm {order_alg} not recognized")
        cur_leech_index = order[shuffle_index]
        clusters, assigned_leeches, active_cluster_diams = [[cur_leech_index]], set([cur_leech_index]), [0.0]
        obj_fxn_val += new_cluster_cost # Start with a single singleton cluster
        shuffle_index += 1
        while len(assigned_leeches) < len(leeches):
            cur_leech_index = order[shuffle_index]
            cand_cost_incrs = [np.inf] * len(clusters)
            cand_diams      = [None] * len(clusters)
            cluster_assigned = False
            for cand_cluster_ind in range(len(clusters)):
                dissim_vals = [dissims[cur_leech_index, other_leech_ind] for other_leech_ind in clusters[cand_cluster_ind]]
                if max(dissim_vals) <= active_cluster_diams[cand_cluster_ind]:
                    # Dissims diameter won't increase, so we just insert this leech here and move on
                    clusters[cand_cluster_ind].append(cur_leech_index)
                    #obj_fxn_val += cand_cost_incrs[cand_cluster_ind] # Obj fxn doesn't change, since leech's assigned cluster diameter didn't increase
                    cluster_assigned = True
                    msg = f"Assigned {cur_leech_index} to cluster # {cand_cluster_ind} w/ max(dissim_vals) (COST UNCHANGED): "
                    msg += f"\n{max([float(dv) for dv in dissim_vals])} vs active diam {active_cluster_diams[cand_cluster_ind]}"
                    vPrint(msg, verbose=verbose)
                    break
                # O.w., store candidate cost increment for comparison
                cand_cost_incrs[cand_cluster_ind]   = max(dissim_vals) - active_cluster_diams[cand_cluster_ind]
                cand_diams[cand_cluster_ind]        = max(dissim_vals)
            if not cluster_assigned:
                best_cand_cluster_ind   = np.argmin(cand_cost_incrs)
                best_cand_incr          = cand_cost_incrs[best_cand_cluster_ind]
                if best_cand_incr < new_cluster_cost:
                    vPrint(f"Assigning {leeches[cur_leech_index]} (# {len(assigned_leeches)+1}) to old cluster # {best_cand_cluster_ind} (COST INCR)", verbose=verbose)
                    clusters[best_cand_cluster_ind].append(cur_leech_index)
                    active_cluster_diams[best_cand_cluster_ind] = cand_diams[best_cand_cluster_ind]
                    obj_fxn_val += best_cand_incr
                else:
                    vPrint(f"Assigning {leeches[cur_leech_index]} (# {len(assigned_leeches)+1}) to new cluster # {len(clusters)+1} (COST INCR)", verbose=verbose)
                    clusters.append([cur_leech_index])
                    active_cluster_diams.append(0.0) # Singleton cluster has 0 dissim diameter trivially
                    obj_fxn_val += new_cluster_cost
            assigned_leeches.add(cur_leech_index)
            shuffle_index += 1
        if verbose:
            print(f"Greedy heuristic iterate # {restart_index} solution had obj val: {obj_fxn_val}")
        else:
            if len(restart_index) % 50 == 0:
                print(f"Greedy heuristic iterate # {restart_index} solution had obj val: {obj_fxn_val}")   
        if obj_fxn_val < min_obj_val:
            min_obj_val = obj_fxn_val
            best_cluster_assignments = clusters
            print(f"Greedy heuristic iterate # {restart_index} found solution with improved obj val: {obj_fxn_val}")

    elapsed_time = time.time() - t0
    print(f"\n# best greedy cluster assignments (leech indices): {best_cluster_assignments}")
    print(f"# best greedy cluster assignments (leeches):")
    for cluster in best_cluster_assignments:
        print(f"\t{[leeches[i] for i in cluster]}")
    print(f"greedyDissimCluster took {elapsed_time} secs ({elapsed_time/num_restarts} secs per greedy search)")
    print(f"cluster lens: {[len(cluster) for cluster in best_cluster_assignments]}")
    print(f"# clusters used: {len(best_cluster_assignments)}")
    print(f"new cluster cost was: {new_cluster_cost}")
    best_cluster_diams = []
    for k, cluster in enumerate(best_cluster_assignments):
        dissims_vals = []
        for i in range(len(cluster)):
            i2 = cluster[i]
            for j in range(i+1, len(cluster)):
                j2 = cluster[j]
                dissims_vals.append( float(dissims[i2, j2]) )
        print(f"cluster # {k} had {len(cluster)} elements")
        if len(cluster)>8:
            print(f"cluster # {k} dissim values histogram:")
            dissim_counts, dissim_bins = np.histogram(dissims_vals)
            for (dc, l, u) in zip(dissim_counts, dissim_bins[:-1], dissim_bins[1:]):
                print(f"\t{dc} dissims in [{l}, {u}]")
        elif len(cluster)>1:
            print(f"\tdissim vals: {dissims_vals}")
        dissims_vals = [0.0] if len(dissims_vals)==0 else dissims_vals 
        best_cluster_diams.append( max(dissims_vals) )
    print(f"best cluster diameters: {best_cluster_diams}")
    print(f"best cluster assignment obj val: {min_obj_val}")
    checkObjVal(activations, new_cluster_cost, best_cluster_assignments)
    return best_cluster_assignments, min_obj_val

def cpsatCluster(activations, leeches, cpsat_cluster_model_num, cpsat_cluster_cost, max_cpsat_clusters, dissim_resolution=100, time_limit=3600.0):
    """
        To determine clusters, this implements and solves a model (after some rounding to move from R to Z) equivalent to:

            sum_{c in C} DIAM_c + cpsat_cluster_cost * num_clusters

        resolution: cpsat doesn't directly support continuous variables, but mangaocr-based dissimilarities are continuous; we convert them to approximate equivalents,
        discretizing at 1/resolution increments
    """
    # Setup
    dissims = np.round(dissim_resolution * (1 - (np.corrcoef(activations)+1)/2))
    max_dissim = np.max(dissims)
    cpsat_cluster_cost = int(cpsat_cluster_cost*dissim_resolution)
    model = cp_model.CpModel()

    # Variables
    cluster_diameter_vars   = dict( (i, model.new_int_var(0, int(max_dissim), f"cl_diam_var{i}") ) for i in range(max_cpsat_clusters) )
    cluster_indicator_vars  = dict( (i, model.new_bool_var(f"cluster_active_var{i}")) for i in range(max_cpsat_clusters) )
    cluster_membership_vars = defaultdict(lambda: defaultdict(lambda: None))
    for lindex, leech in enumerate(leeches):
        for clind in range(max_cpsat_clusters):
            cluster_membership_vars[leech][clind] = model.new_bool_var(f"leech{lindex}_{leech}_cluster_member_var{clind}")

    # Constraints
    for lindex, leech in enumerate(leeches):
        model.add_exactly_one( list( cluster_membership_vars[leech].values() ) ) # A leech must be placed in exactly one clusterr
    for clind, civ in cluster_indicator_vars.items(): # If a cluster contains a leech, it is active
        model.add( max_cpsat_clusters * civ >= sum( cluster_membership_vars[leech][clind] for leech in cluster_membership_vars.keys() ) )
    for clind, cdv in cluster_diameter_vars.items():
        for lind, leech in enumerate(leeches):
            for lind2, leech2 in enumerate(leeches[lind+1:]):
                # Lower bound cluster # clind (cdv) diameter by dissim only if leeches are both in cluster # clind (cdv)
                model.add( cdv >= int(dissims[lind][lind2+lind+1]) ).only_enforce_if([cluster_membership_vars[leech][clind], cluster_membership_vars[leech2][clind]])
    
    # Sum of cluster diameters plus additive penalty for each new active cluster
    model.minimize( sum(cluster_diameter_vars.values()) + sum(cpsat_cluster_cost*civ for civ in cluster_indicator_vars.values()) )

    print("cpsat attempting to solve model...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.num_workers = 4
    solver.parameters.symmetry_level = 3
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        num_active_clusters = sum( solver.Value(civ) for civ in cluster_indicator_vars.values() )
        print(f"# of clusters active in solution: {num_active_clusters}")
        for lindex, leech in enumerate(leeches):
            for clind in range(max_cpsat_clusters):
                var_val = solver.Value(cluster_membership_vars[leech][clind])
                if var_val > 0:
                    print(f"Leech # {lindex}, {leech}, assigned to cluster {clind} (cluster_mem_var[{leech}][{clind}] = {var_val})")
        print(f"Histogram of input dissimilarities (min, max=({np.min(dissims)}, {np.max(dissims)}):")
        dissim_counts, dissim_bins = np.histogram(dissims)
        for (dc, l, u) in zip(dissim_counts, dissim_bins[:-1], dissim_bins[1:]):
            print(f"{dc} dissims in [{l}, {u}]")
        #print(f"Total cost = {solver.objective_value}\n")
        #selected = data.loc[solver.boolean_values(x).loc[lambda x: x].index]
        #for unused_index, row in selected.iterrows():
        #    print(f"{row.task} assigned to {row.worker} with a cost of {row.cost}")
    elif status == cp_model.INFEASIBLE:
        print("No solution found")
    else:
        print("Something is wrong, check the status and the log of the solve")

def kmeansCluster(activations, leeches, num_clusters):
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters).fit(activations)
    for cluster_num in range(num_clusters):
        cluster_members = [leeches[int(i)] for i in np.where(kmeans.labels_==cluster_num)[0]]
        print(f"Cluster # {cluster_num}\n\t{cluster_members}")

def mangaOcrDirectCluster(      read_glob_expr, img_write_folder, acts_write_folder, test_amount=-1, overwrite=False, comment_char='#', sep='\t',
                                cluster_method="kmeans", num_clusters=8, # kmeans params
                                cpsat_cluster_model_num=None, new_cluster_cost=0.05, # cpsat_logic1,2,... params
                                max_cpsat_clusters=50, # (if new_cluster_cost==0.0, creating a 1-elem cluster is free)
                                # >0 has a cost; e.g., at 0.3, it is better to make a new 1-elem cluster unless we can add to a cluster w/o increasing max
                                # pairwise dissimilarity (diameter) by more than 0.3
                                num_greedy_restarts=10,
                            ):
    """
        This functions reads words/expressions (presumably Japanese tokens accumulated as Anki flashcard leeches), feeds these to the manga-ocr
        (https://github.com/kha-white/manga-ocr) pre-trained transformers-based vision encoder/decoder neural network, extracts the activations
        observed in the network for each word/expression, and returns clusters of the target words/expressions on the basis of similarities in
        these activations.

        The general idea is that the activations serve as a convenient numerical representation of notable visual features in
        the targeted words/expressions, and that visual similarity is often an important driver of a flashcard being difficult to memorize
        (since it may be confused with visually similar competitors) and so becoming a leech.

        [Presumably semantic similarity and phonetic similarity are also drivers of flashcards becoming leeches. I intend to examine these, and
        methods of clustering based on the simutlaneous consideration of all three kinds of similarity, in the future.]

        cpsat_cluster_model_num: Python bigint
            - used in cpsatCluster to determine details of model used
                    - 1 uses cpsat logical constraints aggressively
                    - 2 models w/ MILP algebraic constraints
                    - 3 is a mix
        cluster_method: str
            - kmeans:        applies default https://sklearn.org/stable/modules/generated/sklearn.cluster.KMeans.html directly to mangaocr activations
            - cpsat_logic:   uses https://developers.google.com/optimization/cp/cp_solver to build and solve a MIQP equivalent to
                                sum_{c in C} DIAM_c + new_cluster_cost * num_clusters
                             where DIAM_c is the radius of dissimilarities for elements in cluster c; i.e., the largest pairwise dissimilarity in this cluster
    """
    # Setup
    getHFToken()
    assert cpsat_cluster_model_num in (None,1,2,3)
    if cpsat_cluster_model_num not in (None,1):
        raise NotImplementedError(f"cpsat_cluster_model_num vals currently only supports values None,1 (2,3 planned for later)")
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
    cluster_fxn_map =   {
                            "kmeans"        :   (kmeansCluster, {   "activations":activations, "leeches":leeches, "num_clusters":num_clusters}),
                            "cpsat_logic"   :   (cpsatCluster,  {   "activations":activations, "leeches":leeches,
                                                                    "cpsat_cluster_model_num":cpsat_cluster_model_num,
                                                                    "cpsat_cluster_cost":new_cluster_cost,
                                                                    "max_cpsat_clusters":max_cpsat_clusters}),
                            "greedy"        :   (greedyDissimCluster, {"activations":activations, "leeches":leeches,
                                                                            "new_cluster_cost":new_cluster_cost,
                                                                            "num_restarts":num_greedy_restarts,
                                                                        }),
                        }
    cluster_fxn_map[cluster_method][0](**cluster_fxn_map[cluster_method][1])
    
if __name__ == "__main__":
    # Example cmd-line run command:
    # python manga_ocr_direct_cluster.py mangaOcrDirectCluster --read_glob_expr="leeches_6_30_2026.txt" --img_write_folder="leechesDirectTestImgs/" --acts_write_folder="leechesDirectTestActs/"
    fire.Fire()
