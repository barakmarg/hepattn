from hepattn.experiments.odd_pileup_reco.eval_data import run_forward_pass

val_files = [
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00000.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00001.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00002.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00003.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00004.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00005.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00006.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00007.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00008.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00009.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00010.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00011.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00012.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00013.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00014.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00015.parquet',
    '/storage/agrp/barakma/PileupODD/data/ggf_pu200/target_particles-00016.parquet',
]


if __name__ == "__main__":
    hp_path = run_forward_pass(
        ckpt_path='/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/logs/odd_pflow_reco_20260519-T142453/ckpts/epoch=073-val_loss=13.67910.ckpt',
        config_path='/storage/agrp/barakma/hepattn/src/hepattn/experiments/odd_pileup_reco/configs/base.yaml',
        files=val_files,
        num_events=1000,
        batch_size=32,
        test_suff='ggft_pu200_test',
        inference_mode=True,
    )
    print(hp_path)
