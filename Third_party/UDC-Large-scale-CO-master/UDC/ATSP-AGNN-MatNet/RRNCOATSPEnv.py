import numpy as np

from ATSPEnv import ATSPEnv


class RRNCOATSPEnv(ATSPEnv):
    def __init__(self, problem_sampler, device, sampler_seed=1234, **env_params):
        super().__init__(**env_params)
        self.problem_sampler = problem_sampler
        self.device = device
        self.size_rng = np.random.default_rng(sampler_seed)
        self.problem_sizes = list(range(
            self.problem_size_low,
            self.problem_size_high + 1,
            self.problem_size,
        ))
        if not self.problem_sizes or self.problem_sizes[-1] != self.problem_size_high:
            raise ValueError('Problem size range must be divisible by sub_size')

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None):
        if nodes_coords is not None:
            self.raw_problems = nodes_coords[episode:episode + batch_size].to(self.device)
            self.raw_problem_size = self.raw_problems.size(1)
            return

        self.raw_problem_size = int(self.size_rng.choice(self.problem_sizes))
        self.raw_problems = self.problem_sampler.sample(
            batch_size=batch_size,
            node_count=self.raw_problem_size,
            split='train',
        ).to(self.device)
