from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from itertools import product, combinations_with_replacement
import subprocess
import numpy as np
import os
import re
import time
import shutil
import glob
from multiprocessing import Pool
from functools import partial

# --- Config ---
start_time = time.time()
seq_file = snakemake.input.seq_file
seq_format = snakemake.params.get("sequence_format", "phylip")
k = int(snakemake.params.get("k", 5))
shuffle = bool(snakemake.params.get("shuffle_alignment", False))
seed = snakemake.params.get("seed", None)
if seed is None:
    seed = np.random.randint(1, 1000000)
    print(f"Using random seed: {seed}")
else:
    seed = int(seed)
    print(f"Using specified seed: {seed}")
out_file = snakemake.output[0]
n_cores = int(snakemake.params.get("n_cores", 1))

iqtree_bin = snakemake.params["iqtree_bin"]
model_base = snakemake.params["model"]
freq_options = snakemake.params["freq"]
inv_options = snakemake.params["invariant"]
rate_het_options = snakemake.params["rate_heterogeneity"]
max_order = int(snakemake.params["max_order"])

if isinstance(model_base, str):
    base_models = [model_base]
else:
    base_models = model_base

print(f"Base substitution models: {base_models}")

min_components = int(snakemake.params.get("min_components", 1))
stopping_patience = int(snakemake.params.get("stopping_patience", 1))
save_folds = bool(snakemake.params.get("save_folds", False))

print(f"Stopping rules: min_components={min_components}, stopping_patience={stopping_patience}")
print(f"Save fold files: {save_folds}")


# --- Load alignment and create k folds ---
alignment = AlignIO.read(seq_file, seq_format)
L = alignment.get_alignment_length()
indices = np.arange(L)
if shuffle:
    np.random.seed(seed)
    np.random.shuffle(indices)

bins = np.array_split(indices, k)
fold_files = [f"fold_{i}.phy" for i in range(k)]
train_files = [f"train_fold_{i}.phy" for i in range(k)]

# Create fold files (validation sets)
for i, bin_idx in enumerate(bins):
    sub_alignment = MultipleSeqAlignment([
        SeqRecord(
            seq=Seq(''.join(record.seq[j] for j in bin_idx)),
            id=record.id,
            description=record.description
        ) for record in alignment
    ])
    AlignIO.write(sub_alignment, fold_files[i], seq_format)

# Create training files (complement of each fold)
for i in range(k):
    train_indices = [j for j in range(k) if j != i]
    train_bins = [bins[j] for j in train_indices]
    merged_indices = np.concatenate(train_bins)
    
    train_alignment = MultipleSeqAlignment([
        SeqRecord(
            seq=Seq(''.join(record.seq[j] for j in merged_indices)),
            id=record.id,
            description=record.description
        ) for record in alignment
    ])
    AlignIO.write(train_alignment, train_files[i], seq_format)

print(f"Alignment partitioned into {k} folds")
print(f"Created {k} validation files: {fold_files}")
print(f"Created {k} training files: {train_files}")
print(f"Using {n_cores} cores for parallelization")

# Calculate cores per fold
if n_cores < k:
    cores_per_fold_list = [1] * k
    parallel_folds = n_cores
else:
    parallel_folds = k
    cores_per_fold_list = []
    base_cores = n_cores // k
    extra_cores = n_cores % k
    for i in range(k):
        if i < extra_cores:
            cores_per_fold_list.append(base_cores + 1)
        else:
            cores_per_fold_list.append(base_cores)
    print(f"Cores per fold: {cores_per_fold_list}")

print(f"Running {parallel_folds} folds in parallel")

# --- Single matrix models ---
single_models = []
for base in base_models:
    for f, i, r in product(freq_options, inv_options, rate_het_options):
        if r.startswith("+R") and "+G4" in r:
            continue
        single_models.append(base + f + i + r)

# --- Helper: extract trained model ---
def extract_trained_model(iqtree_file, single=False):
    with open(iqtree_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if single:
        for line in lines:
            if "--alisim" in line:
                m = re.search(r'-m\s*"([^"]+)"', line)
                if m:
                    model_str = m.group(1)
                    model_str = model_str.replace('+FU{', '+F{')
                    return model_str
        raise ValueError("Line containing --alisim model not found")

    for i, line in enumerate(lines):
        if line.strip() == "SUBSTITUTION PROCESS":
            start_idx = i
            break
    else:
        raise ValueError("SUBSTITUTION PROCESS section not found")

    params = []
    table_started = False
    for line in lines[start_idx:]:
        line = line.strip()
        if line.startswith("No  Component"):
            table_started = True
            continue
        if table_started:
            if line == "" or line.startswith("Model of rate heterogeneity") or line.startswith("Proportion of invariable sites"):
                break
            columns = re.split(r'\s{2,}', line)
            if len(columns) >= 5:
                component_model = columns[4].strip()
                component_model = component_model.replace('+FU{', '+F{')
                params.append(component_model)

    model_str = "MIX{" + ",".join(params) + "}"

    for line in lines[start_idx:]:
        line = line.strip()
        if line.startswith("Proportion of invariable sites"):
            m = re.search(r"Proportion of invariable sites:\s*([\d\.Ee+-]+)", line)
            if m:
                model_str += f"+I{{{m.group(1)}}}"
            break

    for line in lines[start_idx:]:
        line = line.strip()
        if line.startswith("Site proportion and rates:"):
            numbers = re.findall(r'\(([^)]+)\)', line)
            all_nums = ",".join(n.replace(" ", ",") for n in numbers)
            ncat = len(numbers)
            model_str += f"+R{ncat}{{{all_nums}}}"
            break
        elif line.startswith("Gamma shape alpha:"):
            m = re.search(r"Gamma shape alpha:\s*([\d\.Ee+-]+)", line)
            if m:
                model_str += f"+G4{{{m.group(1)}}}"
            break

    return model_str

# --- Helper: run k-fold CV ---
def iqtree_cv(train_file, test_file, iqtree_bin, model, single=False, n_threads=1, fold_id=0, fixed_tree=None):
    import glob
    import time as time_module
    
    pid = os.getpid()
    tmp_prefix = f"cv_tmp_fold{fold_id}_pid{pid}"
    tmp_val_prefix = f"cv_tmp_val_fold{fold_id}_pid{pid}"
    
    for pattern in [f"{tmp_prefix}*", f"{tmp_val_prefix}*"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except:
                pass
    
    iqtree_exe = os.path.join(iqtree_bin, "iqtree3")
    
    # Build training command
    cmd_train = [iqtree_exe,
                 "-s", train_file,
                 "-m", model,
                 "-pre", tmp_prefix,
                 "-quiet",
                 "--epsilon", "0.1",
                 "-T", str(n_threads)]
    
    # Add fixed tree for mixture models
    if fixed_tree is not None:
        cmd_train.extend(["-te", fixed_tree])
    
    result = subprocess.run(cmd_train, check=False, shell=False, 
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = f"IQ-TREE training failed for {tmp_prefix}\n"
        error_msg += f"STDOUT: {result.stdout}\n"
        error_msg += f"STDERR: {result.stderr}\n"
        raise RuntimeError(error_msg)
    
    iqtree_output = f"{tmp_prefix}.iqtree"
    max_wait = 10
    for _ in range(max_wait):
        if os.path.exists(iqtree_output):
            break
        time_module.sleep(1)
    
    if not os.path.exists(iqtree_output):
        raise FileNotFoundError(f"IQ-TREE did not produce {iqtree_output} file after {max_wait}s wait")

    trained_model = extract_trained_model(iqtree_output, single=single)

    cmd_val = [iqtree_exe,
               "-s", test_file,
               "-m", trained_model,
               "-te", f"{tmp_prefix}.treefile",
               "-pre", tmp_val_prefix,
               "-blfix",
               "-quiet",
               "--epsilon", "0.1",
               "-T", "1"]
    
    result_val = subprocess.run(cmd_val, check=False, shell=False,
                               capture_output=True, text=True)
    
    if result_val.returncode != 0:
        error_msg = f"IQ-TREE validation failed for {tmp_val_prefix}\n"
        error_msg += f"STDOUT: {result_val.stdout}\n"
        error_msg += f"STDERR: {result_val.stderr}\n"
        raise RuntimeError(error_msg)

    ll = None
    val_output = f"{tmp_val_prefix}.iqtree"
    
    for _ in range(max_wait):
        if os.path.exists(val_output):
            break
        time_module.sleep(1)
    
    if not os.path.exists(val_output):
        raise FileNotFoundError(f"IQ-TREE did not produce {val_output} file")
    
    with open(val_output, "r", encoding='utf-8') as f:
        for line in f:
            m = re.search(r"Log-likelihood of the tree:\s*([-\d\.Ee]+)", line)
            if m:
                ll = float(m.group(1))
                break

    if ll is None:
        raise ValueError("Could not read log-likelihood from validation output")
    
    # Clean up all temporary files
    for pattern in [f"{tmp_prefix}*", f"{tmp_val_prefix}*"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except:
                pass
    
    return ll

# --- Helper: process single fold ---
def process_fold(fold_idx, model, train_file, val_file, iqtree_bin, single, n_threads, fixed_tree=None):
    """Process a single fold for cross-validation"""
    ll = iqtree_cv(train_file, val_file, iqtree_bin, model, single=single, n_threads=n_threads, fold_id=fold_idx, fixed_tree=fixed_tree)
    return ll

# --- Run CV for single-matrix models ---
print("\n" + "="*60)
print("TESTING SINGLE-MATRIX MODELS")
print("="*60)
single_start = time.time()

cv_results = []
for model in single_models:
    model_start = time.time()
    print(f"Testing model: {model}")
    
    if n_cores > 1:
        with Pool(processes=parallel_folds) as pool:
            fold_args = [(i, model, train_files[i], fold_files[i], iqtree_bin, True, cores_per_fold_list[i], None) 
                         for i in range(k)]
            lls = pool.starmap(process_fold, fold_args)
        cv_ll = sum(lls)
    else:
        cv_ll = 0.0
        for i in range(k):
            ll = process_fold(i, model, train_files[i], fold_files[i], iqtree_bin, True, 1, None)
            cv_ll += ll

    cv_results.append((model, cv_ll))
    model_time = time.time() - model_start
    print(f"Model {model} CV score: {cv_ll} (Time: {model_time:.2f}s)")

single_time = time.time() - single_start
print(f"\nSingle-matrix models completed in {single_time:.2f}s ({single_time/60:.2f} minutes)")

# --- Find best single-matrix model ---
best_single_model, best_single_ll = max(cv_results, key=lambda x: x[1])
print(f"\nBest single-matrix model: {best_single_model} with CV score: {best_single_ll}")

# --- Infer tree topology using best single model on FULL dataset ---
print(f"\nInferring tree topology using {best_single_model} on full dataset...")

iqtree_exe = os.path.join(iqtree_bin, "iqtree3")
cmd_full_tree = [iqtree_exe,
                 "-s", seq_file,
                 "-m", best_single_model,
                 "-pre", "full_tree",
                 "-quiet",
                 "--epsilon", "0.1",
                 "-nt", "AUTO"]

result = subprocess.run(cmd_full_tree, check=False, shell=False,
                       capture_output=True, text=True)

if result.returncode != 0:
    error_msg = f"IQ-TREE failed on full dataset\n"
    error_msg += f"STDOUT: {result.stdout}\n"
    error_msg += f"STDERR: {result.stderr}\n"
    raise RuntimeError(error_msg)

full_tree_file = "full_tree.treefile"
if not os.path.exists(full_tree_file):
    raise FileNotFoundError(f"IQ-TREE did not produce {full_tree_file}")

print(f"Tree topology saved: {full_tree_file}")

def parse_model(model_str):
    rate_het = ''
    for part in ['+I', '+G4'] + [f'+R{i}' for i in range(2, 11)]:
        if part in model_str:
            rate_het += part
    base_with_freq = model_str.replace(rate_het, '')
    return base_with_freq, rate_het

best_component, rate_het_modifiers = parse_model(best_single_model)
print(f"Best component: {best_component}")
print(f"Rate heterogeneity modifiers: {rate_het_modifiers}")

# Build component options: all combinations of base models + frequency options
component_options = []
for base in base_models:
    for f in freq_options:
        component_options.append(base + f)
print(f"Component options for new additions: {component_options}")

# --- Test mixture models ---
print("\n" + "="*60)
print("TESTING MIXTURE MODELS")
print("="*60)
mixture_start = time.time()

best_at_order = {1: (best_single_model, best_single_ll)}
best_components = [best_component]
best_overall_order = 1
best_overall_ll = best_single_ll
no_improvement_count = 0

for order in range(2, max_order + 1):
    order_start = time.time()
    print(f"\n=== Testing mixtures with {order} components ===")
    print(f"Fixed components: {best_components}")
    
    best_mixture_this_order = None
    best_ll_this_order = float('-inf')
    best_new_component = None
    
    for new_component in component_options:
        test_components = best_components + [new_component]
        mixture_model = f"MIX{{{','.join(test_components)}}}{rate_het_modifiers}"
        print(f"\nTesting: {mixture_model}")
        
        cv_ll = 0.0
        try:
            if n_cores > 1:
                with Pool(processes=parallel_folds) as pool:
                    fold_args = [(i, mixture_model, train_files[i], fold_files[i], iqtree_bin, False, cores_per_fold_list[i], full_tree_file) 
                                 for i in range(k)]
                    lls = pool.starmap(process_fold, fold_args)
                cv_ll = sum(lls)
            else:
                for i in range(k):
                    ll = process_fold(i, mixture_model, train_files[i], fold_files[i], iqtree_bin, False, 1, full_tree_file)
                    cv_ll += ll
            
            print(f"  CV score: {cv_ll}")
            
            if cv_ll > best_ll_this_order:
                best_ll_this_order = cv_ll
                best_mixture_this_order = mixture_model
                best_new_component = new_component
                
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    order_time = time.time() - order_start
    
    if best_mixture_this_order is None:
        print(f"No valid mixture found for order {order}. Stopping.")
        break
    
    best_at_order[order] = (best_mixture_this_order, best_ll_this_order)
    print(f"\nBest mixture with {order} components: {best_mixture_this_order}")
    print(f"CV score: {best_ll_this_order}")
    print(f"Added component: {best_new_component}")
    print(f"Time for order {order}: {order_time:.2f}s ({order_time/60:.2f} minutes)")
    
    improvement = best_ll_this_order - best_overall_ll
    
    if best_ll_this_order > best_overall_ll:
        print(f"Improvement: {improvement:.4f}")
        best_overall_order = order
        best_overall_ll = best_ll_this_order
        best_components.append(best_new_component)
        no_improvement_count = 0
    else:
        print(f"No improvement over best (delta: {improvement:.4f})")
        no_improvement_count += 1
        
        # Check stopping criteria
        if order >= min_components and no_improvement_count >= stopping_patience:
            print(f"\nStopping: {no_improvement_count} consecutive orders without improvement (patience={stopping_patience})")
            print(f"Minimum components ({min_components}) requirement satisfied")
            break
        elif order < min_components:
            print(f"Continuing: have not reached min_components={min_components}")
            best_components.append(best_new_component)
        else:
            print(f"Continuing: no_improvement_count={no_improvement_count} < patience={stopping_patience}")
            best_components.append(best_new_component)

mixture_time = time.time() - mixture_start
print(f"\nMixture models completed in {mixture_time:.2f}s ({mixture_time/60:.2f} minutes)")

best_model, best_ll = best_at_order[best_overall_order]
total_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model}")
print(f"Components: {best_overall_order}")
print(f"CV score: {best_ll}")
print(f"{'='*60}")
print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"{'='*60}")

# --- Write summary ---
with open(out_file, "w") as f:
    f.write("Model\tCV_score\tComponents\n")
    for model, score in cv_results:
        f.write(f"{model}\t{score}\t1\n")
    for order in range(2, len(best_at_order) + 1):
        if order in best_at_order:
            model, score = best_at_order[order]
            f.write(f"{model}\t{score}\t{order}\n")
    f.write(f"\n# Best model: {best_model} ({best_overall_order} components) with CV score: {best_ll}\n")
    f.write(f"# Stopping rules: min_components={min_components}, stopping_patience={stopping_patience}\n")

# --- Cleanup ---
if save_folds:
    # Create Folds directory and move files there
    folds_dir = "Folds"
    os.makedirs(folds_dir, exist_ok=True)
    print(f"\nMoving fold files to {folds_dir}/")
    
    for fold_file in fold_files:
        if os.path.exists(fold_file):
            shutil.move(fold_file, os.path.join(folds_dir, fold_file))
    
    for train_file in train_files:
        if os.path.exists(train_file):
            shutil.move(train_file, os.path.join(folds_dir, train_file))
    
    # Move the full tree file too
    if os.path.exists(full_tree_file):
        shutil.move(full_tree_file, os.path.join(folds_dir, full_tree_file))
    
    # Move any full_tree.* files
    for f in glob.glob("full_tree.*"):
        if os.path.exists(f):
            shutil.move(f, os.path.join(folds_dir, os.path.basename(f)))
else:
    # Delete all fold files
    print("\nCleaning up fold files...")
    for fold_file in fold_files:
        if os.path.exists(fold_file):
            os.remove(fold_file)
    
    for train_file in train_files:
        if os.path.exists(train_file):
            os.remove(train_file)
    
    # Delete full tree files
    for f in glob.glob("full_tree.*"):
        if os.path.exists(f):
            os.remove(f)

print(f"CV summary written to {out_file}")