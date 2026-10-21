from combined_main import run_experiment
import numpy as np
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

CPU_THROTTLE = 0.6
THREADS_PER_WORKER = 1
CUDA_ENABLED = True

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = str(THREADS_PER_WORKER)

class ExperimentConfig():
    def __init__(
        self,
        env_name: str = "lunar_lander",
        gamma: float = 0.995,
        lr: float = 0.0003,
        seed: int = 123456,
        batch_size: int = 256,
        mini_batch_size: int = 256,
        num_steps: int = 200_000,
        hidden_size: int = 256,
        replay_size: int = 1_000_000,
        start_steps: int = 10_000,
        cuda: bool = True,
        horizon: int = 5,
        cbf_gamma: float = 0.7,
        red_dim: int = 20,
        no_safety: bool = False,
        render: bool = False,
        percentile: int = 99,
        dynamics: str = None,
        cbf_alpha: float = 0.1,
        noise: float = 0.1,
        verbosity: int = 0,
        policy: str = "Gaussian",
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = False,
        updates_per_step: int = 1,
        target_update_interval: int = 1,
        adaptive_error: bool = False,
        wp: bool = False,
        aspos: bool = True,
        ppo: bool = True,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.lr = lr
        self.seed = seed
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.replay_size = replay_size
        self.start_steps = start_steps
        self.cuda = cuda
        self.horizon = horizon
        self.cbf_gamma = cbf_gamma
        self.red_dim = red_dim
        self.no_safety = no_safety
        self.render = render
        self.percentile = percentile
        self.dynamics = dynamics
        self.cbf_alpha = cbf_alpha
        self.noise = noise
        self.verbosity = verbosity

        self.policy = policy
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.updates_per_step = updates_per_step
        self.target_update_interval = target_update_interval

        self.adaptive_error = adaptive_error
        self.wp = wp
        self.aspos = aspos
        self.ppo = ppo

def _run_group(job):
    env, approach, method_name, alpha, seeds = job
    violations = { "real": [], "sim": [], "test": [] }

    for seed in seeds:
        config = ExperimentConfig(env_name=env, seed=seed, cbf_alpha=alpha, noise=alpha, red_dim=0, ppo=(method_name=='PPO'), aspos=(approach=='ASPOS'))
        if method_name == "PPO":
            config.batch_size = 2048

        results = run_experiment(config)

        for split in violations:
            violations[split].append( results[split]["unsafe"] / (results[split]["total"] + 1e-8) )
        
    summary = {
        split: {
            "mean": float(np.mean( split_val )),
            "std": float(np.std( split_val ))
        }
        for split, split_val in violations.items()
    }

    return { "env": env, "method": method_name, "alpha": alpha, "summary": summary}

def _init_worker():
    try:
        import torch

        torch.set_num_threads(THREADS_PER_WORKER)
        try:
            torch.set_num_interop_threads(THREADS_PER_WORKER)
        except RuntimeError:
            # PyTorch may already have initialized this setting.
            pass
    except ImportError:
        pass

def _get_available_cpus():
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)

def _worker_count(job_count):
    cpus = _get_available_cpus()

    workers = max(1, int(cpus * CPU_THROTTLE))
    return min(job_count, workers)

def run_exps():
    approaches = ["RAMPS", "ASPOS"]
    methods = ["PPO"]
    envs = ["pendulum", "cheetah", "hopper", "ant"]
    seeds = [1, 3, 5, 11, 42]
    cbf_alphas = [ 0.1, 0.01, 0.001, 0.0 ]
    metrics_file = "./runs/combined_results.txt"

    jobs = [
        (env, approach, method_name, alpha, seeds)
        for env in envs
        for approach in approaches
        for method_name in methods
        for alpha in cbf_alphas
    ]
    workers = _worker_count(len(jobs))

    context = mp.get_context("spawn")
    indexed_results = {}
    with ProcessPoolExecutor( max_workers = workers, mp_context = context, initializer = _init_worker ) as executor:
        future_to_job = {
            executor.submit(_run_group, job): (index, job)
            for index, job in enumerate(jobs)
        }
        for future in as_completed(future_to_job):
            index, job = future_to_job[future]
            env, approach, method_name, alpha, _ = job
            try:
                indexed_results[index] = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Experiment group failed: env={env}, method={method_name}, alpha={alpha}"
                ) from exc

    with open(metrics_file, 'w') as f:
        f.write("Mean + STD for experimental run\n" +\
            "------------------------------------------------------------------------------"
            )
        
        for index in range(len(jobs)):
            result = indexed_results[index]
            f.write(f"{env.capitalize()} {'PPO' if method == main_ppo else 'SAC'} - Noise: {alpha}\n")
            for split in ("real", "sim", "test"):
                stats = result["summary"][split]
                f.write( f"Total unsafe {split}: {stats['real']}  ± {stats['real']} \n")
            f.write("\n\n")


if __name__ == "__main__":
    run_exps()
