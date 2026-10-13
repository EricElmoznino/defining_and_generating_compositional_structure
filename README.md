# Defining and generating compositional structure

Code for estimating compositional structure and using it as a reward for evolutionary generation of symbolic strings and images. See the paper for the theoretical definitions and experimental details.

## Implementation

The measure compares models fitted to an object and its parts. A two-part coding procedure fits a model while accounting for model complexity and the cost of the data given the model. A conditional coding procedure then estimates the additional cost of describing the whole-model given the part-models. Applying this at each join in a decomposition tree gives a compositional structure profile, which a reward function aggregates for optimization.

These empirical costs depend on the fitting and conditional coding procedures. They capture relative complexity rather than providing formal bounds on the underlying algorithmic-information quantities.

| Location | Contents |
| --- | --- |
| `two_part_codes/` | Model fitting and two-part code lengths. `TwoPartCode.compress` returns fitted models, model costs, and data-given-model costs. |
| `conditional_compressors/` | Conditional model costs. `ConditionalCompressor.compress` describes a target model given models of its parts. |
| `decompositions/` | Decomposition trees. `parts` enumerates parts; `joins` links children to their parent. |
| `rewards/` | Profile computation and aggregation in `DecompositionReward` and its subclasses, plus sophistication and debug rewards. |
| `tasks/` | String evolution with pymoo and image generation through NEAT evolution of compositional pattern-producing networks (CPPNs). |
| `lib/` | Modified RePair, batched neural fitting, optimization utilities, CPPN coordinates and statistics, and visualization helpers. |
| `configs/` | Hydra configurations for tasks, rewards, compressors, logging, and cluster resources. |
| `evolve.py` | Entry point for evolutionary optimization. |
| `slurm/` | Submission commands using Hydra's Submitit launcher. |

Start with a YAML file in `configs/experiment/` and follow its `_target_` entries to the relevant implementations. Each component's `base.py` defines its interface.

String models are deterministic grammars fitted with modified RePair; conditional compression aligns rules and measures their differences. Image models are coordinate MLPs whose complexity is estimated from learning curves. Their conditional compressor adapts a spatially routed mixture of child models to the parent model. These fitted MLPs are separate from the evolved CPPNs that generate candidate images.

Inputs have a leading batch dimension: `(batch, length)` for strings and `(batch, height, width, channels)` for images. Symbols are integers in `[0, vocab_size)`.

## Dependencies

Use Python 3.11 with:

- NumPy, SciPy, and PyTorch.
- Hydra (`hydra-core`) and OmegaConf.
- `pymoo`, Dask, and `distributed` for string evolution.
- `neat-python` 2.0 for CPPN evolution and checkpoint restoration.
- `wandb`, pandas, Matplotlib, Seaborn, Pillow, and `soundfile` for logging and visualizations.
- `hydra-submitit-launcher` and Submitit for SLURM submission.

The GPU configurations require a CUDA-compatible PyTorch installation. Direct local runs do not require Submitit or SLURM.

## Running experiments

Run commands from the repository root with your Python environment active. Select a configuration from `configs/experiment/` and override settings on the command line:

```sh
python evolve.py experiment=evolve_string/hierarchical \
    save_dir=/path/to/logs logger.entity=YOUR_ENTITY logger.project=YOUR_PROJECT
```

Set `logger.entity` and `logger.project` to your W&B destination, or pass `logger=null` to disable logging.

Overrides use full configuration paths, such as `experiment.n_generations` or `experiment.algorithm.pop_size`. For a small local run, reduce the population and generation count and set `experiment.task.n_workers=1`. String tasks use Dask processes with multiple workers; `n_workers=-1` uses all CPUs reported by Python. Set the worker count explicitly to match a cluster allocation.

For CPPN tasks, population size is `experiment.task.pop_size`, and reward evaluation batches are controlled by `experiment.task.reward_batch_size`. Use `experiment.task.reward_device=cpu` to run GPU configurations on a CPU. Neural fitting dominates the cost; its step limits are configured under `experiment.task.reward.two_part_code`.

Use `seed=...` to select a replicate. Hydra's `--multirun` accepts sweeps such as `seed=27,28,29`.

### SLURM

The commands in `slurm/` submit jobs through Hydra and Submitit; they are not `#SBATCH` scripts. Before using them, adapt a launcher profile under `configs/hydra/launcher/` to your cluster's account, partition, CPU and memory allocation, GPU resources, time limits, and environment activation commands. Update the submission command's absolute `save_dir` and any resource overrides as well.

Submit from the repository root with `--multirun hydra/launcher=YOUR_PROFILE` and the chosen experiment. Compute nodes need access to the repository, Python environment, and output directory. The data are generated during evolution, so no external dataset is required.

### Outputs and checkpoints

Hydra saves run metadata under `${save_dir}/outputs/` or `${save_dir}/outputs_multiruns/`, including configurations and overrides in each job's `.hydra/` directory. W&B records rewards, population statistics, and selected examples. With `logger=null`, these metrics and examples are not saved; `evolve.py` does not separately serialize the optimization result.

CPPN checkpoints are stored under `${save_dir}/evolve_cppn_checkpoints/<run_id>/` when checkpointing is enabled and a run ID is available from W&B, `WANDB_RUN_ID`, SLURM, or an explicit resume ID. Pass `experiment.task.resume_run_id=THE_PREVIOUS_RUN_ID` with the same checkpoint root to load the latest checkpoint. `experiment.n_generations` then specifies additional generations. String evolution does not support checkpoint restoration.

## License

[MIT](LICENSE).
