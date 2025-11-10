configfile: "config.yml"

rule all:
    input:
        "cv_results.txt"

rule run_kfold_cv:
    input:
        seq_file=config["sequence_file"]
    output:
        "cv_results.txt"
    params:
        sequence_format=config["sequence_format"],
        k=config["k"],
        shuffle_alignment=config["shuffle_alignment"],
        seed=config.get("seed", None),
        n_cores=config.get("n_cores", 1),
        iqtree_bin=config["iqtree_bin"],
        model=config["model"],
        freq=config["freq"],
        invariant=config["invariant"],
        rate_heterogeneity=config["rate_heterogeneity"],
        max_order=config["max_order"],
        min_components=config.get("min_components", 1),
        stopping_patience=config.get("stopping_patience", 1),
        save_folds=config.get("save_folds", False)
    script:
        "k-fold_CV.py"