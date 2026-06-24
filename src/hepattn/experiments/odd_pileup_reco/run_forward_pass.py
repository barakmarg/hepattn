import os

from hepattn.experiments.odd_pileup_reco.eval_data import (
    analyze_tf_mask_strategies,
    compare_tf_vs_pred_loss,
    diagnose_track_class_attribution,
    run_forward_pass,
    sweep_tf_mask_schemes,
    verify_tf_selection,
)

# Held-out events to evaluate on. NOTE: for the closest comparison to the logged
# train/val numbers, use the training distribution (ttbar PU200) val split. The
# TF-vs-pred delta on identical events tests the exposure-bias hypothesis regardless.
# val_files = [
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00000.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00001.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00002.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00003.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00004.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00005.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00006.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00007.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00008.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00009.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00010.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00011.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00012.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00013.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00014.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00015.parquet',
#     '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00016.parquet',
# ]
val_files = path_list = [
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00686.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00214.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00363.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00379.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00166.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00373.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00854.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00650.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00464.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00954.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00947.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00103.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00887.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00978.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00875.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00081.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00296.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00791.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00233.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00677.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00046.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00071.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00721.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00196.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00591.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00370.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00882.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00906.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00633.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00643.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00849.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00300.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00565.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00080.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00387.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00127.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00549.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00470.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00747.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00044.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00826.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00270.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00618.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00352.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00867.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00367.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00099.parquet",
    "/storage/agrp/barakma/PileupODD/data/ttbar_pu200/target_particles-00389.parquet"
]
#CKPT = '/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260615-T105829/ckpts/epoch=072-val_loss=12.74524.ckpt'
# CKPT can be overridden at submit time via the FWD_CKPT env var (see submit_forward.sh).
CKPT = os.environ.get(
    'FWD_CKPT',
    '/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260615-T233516/ckpts/epoch=083-val_loss=11.27779.ckpt',
)
#CKPT = '/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260616-T142037/ckpts/epoch=030-val_loss=13.34838.ckpt'
#CKPT = '/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260620-T125301/ckpts/epoch=028-val_loss=10.15650.ckpt'
CONFIG = '/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/configs/base.yaml'

# Evaluation dataset (all-vertices paper sample) — forward pass runs over the
# first DATA_NUM_EVENTS events in this directory.
#DATA_DIR = '/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_paper'
DATA_DIR = '/storage/agrp/barakma/PileupODD/data/dihiggs_pu200_all_vertices_paper'
DATA_NUM_EVENTS = 10000


if __name__ == "__main__":
    # Forward pass over the all-vertices paper sample, written as 1000-event H5
    # shards inside a folder next to the checkpoint (last shard holds the
    # remainder). devices=1 so the prediction writer writes a single event stream.
    print("started running")
    h5_paths = run_forward_pass(
        ckpt_path=CKPT,
        config_path=CONFIG,
        data_dir=DATA_DIR,
        num_events=DATA_NUM_EVENTS,
        batch_size=128,
        num_workers=8,
        accelerator="gpu",
        devices=1,
        inference_mode=True,
        events_per_file=1000,
        test_suff="ttbar_pu200_ggf_finetune_paper",
        predict_only=False,
    )
    print(f"Wrote {len(h5_paths)} shard(s):")
    for _p in h5_paths:
        print(f"  {_p}")

    # --- Reference diagnostics (kept for reference) ---
    # verify = verify_tf_selection(
    #     ckpt_path=CKPT,
    #     config_path=CONFIG,
    #     files=val_files,
    #     num_events=2000,
    #     batch_size=16,
    # )
    # print(verify["means"])

    # #--- Idea 1: factorial track/calo ablation (kept for reference) ---
    # attribution = diagnose_track_class_attribution(
    #     ckpt_path=CKPT, config_path=CONFIG, files=val_files, num_events=2000, batch_size=32,
    # )
    # print(attribution["gap_tracks"], attribution["gap_calo"], attribution["total_gap"])

    # #--- TF-mask FP+FN scheme sweep (kept for reference) ---
    # sweep = sweep_tf_mask_schemes(
    #     ckpt_path=CKPT,
    #     config_path=CONFIG,
    #     files=val_files,
    #     num_events=4000,
    #     batch_size=32,
    #     fn_fracs=(0.10,),
    #     fp_counts=(170,),
    #     fp_count_stds=(75.0, 100.0, 125.0, 150.0),
    #     track_fn_frac=0.1,
    #     track_fp_frac=0.1,
    # )
    # print("BEST:", sweep["best"]["name"])

    # #--- Exposure-bias loss comparison (TF vs pred), kept for reference ---
    # results = compare_tf_vs_pred_loss(
    #     ckpt_path=CKPT,
    #     config_path=CONFIG,
    #     files=val_files,
    #     num_events=1000,
    #     batch_size=32,
    # )
    # print(results)

    # --- Original H5 forward-pass invocation (kept for reference) ---
    # hp_path = run_forward_pass(
    #     ckpt_path=CKPT,
    #     config_path=CONFIG,
    #     files=val_files,
    #     num_events=1000,
    #     batch_size=32,
    #     test_suff='ggft_pu200_test',
    #     inference_mode=True,
    # )
    # print(hp_path)
