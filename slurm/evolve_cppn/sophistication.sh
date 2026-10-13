python evolve.py --multirun \
    hydra/launcher=mila_eric \
    save_dir="/home/mila/e/eric.elmoznino/scratch/generative_compositional/logs" \
    hydra.launcher.gres="gpu:rtx8000:1" \
    hydra.launcher.timeout_min=10080 \
    experiment=evolve_cppn/sophistication