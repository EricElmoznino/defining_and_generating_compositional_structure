import os
from typing import Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from lib import ast_eval


@hydra.main(config_path="./configs/", config_name="evolve", version_base=None)
def evolve(cfg: DictConfig) -> None:
    task = hydra.utils.instantiate(cfg.experiment.task)
    algorithm_cfg = OmegaConf.select(cfg, "experiment.algorithm")
    algorithm = (
        hydra.utils.instantiate(algorithm_cfg) if algorithm_cfg is not None else None
    )

    if cfg.logger:
        logger_run_id = _resolve_wandb_run_id(cfg)
        logger_resume = OmegaConf.select(cfg, "logger.resume")
        if logger_resume is None and logger_run_id is not None:
            logger_resume = "allow"
        logger = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            id=logger_run_id,
            name=cfg.logger.name,
            dir=cfg.logger.save_dir,
            reinit="finish_previous",
            resume=logger_resume,
            tags=cfg.logger.tags,
            config=OmegaConf.to_container(cfg.experiment, resolve=True),
            allow_val_change=logger_resume is not None,
        )
        logger.config.update({"seed": cfg.seed})
    else:
        logger = None

    optimize_kwargs = {
        "n_generations": cfg.experiment.n_generations,
        "logger": logger,
        "seed": cfg.seed,
    }
    if algorithm is not None:
        optimize_kwargs["algorithm"] = algorithm
    task.optimize(**optimize_kwargs)


def _resolve_wandb_run_id(cfg: DictConfig) -> str | None:
    explicit_id = _none_if_empty(OmegaConf.select(cfg, "logger.id"))
    if explicit_id is not None:
        return explicit_id

    resume_run_id = _none_if_empty(
        OmegaConf.select(cfg, "experiment.task.resume_run_id")
    )
    if resume_run_id is not None:
        return resume_run_id

    env_run_id = _none_if_empty(os.environ.get("WANDB_RUN_ID"))
    if env_run_id is not None:
        return env_run_id

    if _is_evolve_cppn_task(cfg):
        return _slurm_run_id()

    return None


def _is_evolve_cppn_task(cfg: DictConfig) -> bool:
    return (
        OmegaConf.select(cfg, "experiment.task._target_")
        == "tasks.evolve_cppn.EvolutionaryOptimizationCPPN"
    )


def _slurm_run_id() -> str | None:
    array_job_id = _none_if_empty(os.environ.get("SLURM_ARRAY_JOB_ID"))
    array_task_id = _none_if_empty(os.environ.get("SLURM_ARRAY_TASK_ID"))
    if array_job_id is not None and array_task_id is not None:
        return f"slurm-{array_job_id}-{array_task_id}"

    job_id = _none_if_empty(os.environ.get("SLURM_JOB_ID"))
    if job_id is not None:
        return f"slurm-{job_id}"

    return None


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    string_value = str(value).strip()
    return string_value or None


if __name__ == "__main__":
    """Run with:
    `python evolve.py experiment=[experiment_folder]/[experiment_name].yaml [overrides]`
    """
    evolve()
