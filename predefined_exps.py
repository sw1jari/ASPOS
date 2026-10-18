from main_ppo import main as main_ppo
from main_sac import main as main_sac
import numpy as np

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


def run_exps():
    methods = [main_ppo, main_sac]
    envs = ["pendulum", "cheetah", "hopper", "ant"]
    seeds = [1, 3, 5, 11, 42]
    cbf_alphas = [ 0.1, 0.01, 0.001, 0.0 ]
    metrics_file = "./runs/combined_results.txt"

    with open(metrics_file, 'w') as f:
        f.write("Mean + STD for experimental run\n\n" +\
            "------------------------------------------------------------------------------"
            )

    for env in envs:
        for method in methods:
            for alpha in cbf_alphas:
                to_write = f"{env.capitalize()} {'PPO' if method == main_ppo else 'SAC'} - Noise: {alpha}\n\n\n"
                violations = {
                    "real": [],
                    "sim": [],
                    "test": []
                }
                for seed in seeds:
                    config = ExperimentConfig(env_name=env, seed=seed, cbf_alpha=alpha, noise=alpha, red_dim=34)
                    if method == main_ppo:
                        config.batch_size = 2048
                    results = method(config)
                    violations["real"].append( results["real"]["unsafe"] / (results["real"]["total"] + 1e-8) )
                    violations["sim"].append( results["sim"]["unsafe"] / (results["sim"]["total"] + 1e-8) )
                    violations["test"].append( results["test"]["unsafe"] / (results["test"]["total"] + 1e-8) )
                to_write += "Total unsafe real: " + np.mean(violations["real"]) + " ±" + np.std(violations["real"] + "\n")
                to_write += "Total unsafe sim: " + np.mean(violations["sim"]) + " ±" + np.std(violations["sim"] + "\n")
                to_write += "Total unsafe test: " + np.mean(violations["test"]) + " ±" + np.std(violations["test"] + "\n\n\n")
                with open(metrics_file, 'a') as f:
                    f.write(to_write)
            



if __name__ == "__main__":
    run_exps()
