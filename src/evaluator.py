import numpy as np
import torch
from itertools import chain


class Evaluator(object):
    def __init__(self, environment, model, logger, agents, max_steps):
        self.env = environment
        self.model = model
        self.logger = logger
        self.agents = agents
        self.max_steps = max_steps
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def play_n_episodes(self, render=False, fixed_spawn=None, silent=False):
        """
        wraps play_one_episode, playing a single episode at a time and logs
        results used when playing demos.
        """
        if fixed_spawn is None:
            # rand_positions = np.random.random(10).reshape((-1, 2))
            # fixed_spawn = np.array([0.5, 0.5] * 20).reshape((-1, 2))
            # fixed_spawn = np.concatenate((rand_positions, fixed_spawn))
            # np.random.shuffle(fixed_spawn)
            # num_runs = fixed_spawn.shape[0]
            # fixed_spawn = np.stack([fixed_spawn for _ in range(self.agents)], axis=-1)
            fixed_spawn = np.array([0.5, 0.5]).reshape((-1, 2))
            num_runs = fixed_spawn.shape[0]
            fixed_spawn = np.stack([fixed_spawn for _ in range(self.agents)], axis=-1)

        else:
            # fixed_spawn should be, for example, [0.5 , 0.5 , 0, 0] for 2 runs in 2D
            # In the first run agents spawn in the middle and in the second they will spawn from the corner
            fixed_spawn = np.array(fixed_spawn).reshape((-1, 2))  # 2 dimensions
            num_runs = fixed_spawn.shape[0]
            # Set all the agents to the same spawn point
            fixed_spawn = np.stack([fixed_spawn for _ in range(self.agents)], axis=-1)

        num_files = self.env.files.num_files
        self.model.train(False)
        # headers = ["number"] + list(chain.from_iterable(zip(
        #     [f"Filename {i}" for i in range(self.agents)],
        #     [f"Agent {i} pos x" for i in range(self.agents)],
        #     [f"Agent {i} pos y" for i in range(self.agents)],
        #     [f"Agent {i} pos z" for i in range(self.agents)],
        #     [f"Landmark {i} pos x" for i in range(self.agents)],
        #     [f"Landmark {i} pos y" for i in range(self.agents)],
        #     [f"Landmark {i} pos z" for i in range(self.agents)],
        #     [f"Distance {i}" for i in range(self.agents)])))
        headers = ["number"] + list(chain.from_iterable(zip(
            [f"Filename {i}" for i in range(self.agents)],
            [f"Agent {i} pos x" for i in range(self.agents)],
            [f"Agent {i} pos y" for i in range(self.agents)],
            [f"Landmark {i} pos x" for i in range(self.agents)],
            [f"Landmark {i} pos y" for i in range(self.agents)],
            [f"Distance {i}" for i in range(self.agents)])))
        self.logger.write_locations(headers)
        distances = []
        for j in range(num_runs):
            for k in range(num_files):
                score, start_dists, q_values, info = self.play_one_episode(render, fixed_spawn=fixed_spawn[j])
                row = [j * num_files + k + 1] + list(chain.from_iterable(zip(
                    [info[f"filename_{i}"] for i in range(self.agents)],
                    [info[f"agent_xpos_{i}"] for i in range(self.agents)],
                    [info[f"agent_ypos_{i}"] for i in range(self.agents)],
                    [info[f"landmark_xpos_{i}"] for i in range(self.agents)],
                    [info[f"landmark_ypos_{i}"] for i in range(self.agents)],
                    [info[f"distError_{i}"] for i in range(self.agents)])))
                distances.append([info[f"distError_{i}"] for i in range(self.agents)])

                self.logger.write_locations(row)
        mean = np.nanmean(distances, 0)
        std = np.nanstd(distances, 0, ddof=1)
        if not silent:
            self.logger.log(f"mean distances {mean}")
            self.logger.log(f"Std distances {std}")
        return mean, std

    def play_one_episode(self, render=False, frame_history=4, fixed_spawn=None):

        def predict(obs_stack, agents_wtargets=None):
            """
            Run a full episode, mapping observation to action,
            using greedy policy.
            """
            # from (agent, imagex, imagey, framehist) -> (agent, framehist, imagex, imagey)
            inputs = torch.tensor(obs_stack).permute(0, 3, 1, 2).unsqueeze(0)
            q_vals = self.model.forward(inputs.to(self.device)).detach().squeeze(0)
            q_vals = q_vals.cpu()
            idx = torch.max(q_vals, -1)[1]
            greedy_steps = np.array(idx, dtype=np.int32).flatten()
            return greedy_steps, q_vals.data.numpy()

        obs_stack, targets = self.env.reset(fixed_spawn)
        # agents_wtargets = np.argwhere(~np.isnan(np.asarray(targets)[:, 0])).reshape([-1, ])
        # Here obs have shape (agent, *image_size, frame_history)
        sum_r = np.zeros(self.agents)
        isOver = [False] * self.agents
        start_dists = None
        steps = 0
        while steps < self.max_steps and not np.all(isOver):
            acts, q_values = predict(obs_stack)
            # obs_stack, r, isOver, info, agents_wtarget = self.env.step(acts, q_values, isOver)
            obs_stack, r, isOver, info, _ = self.env.step(acts, q_values, isOver)
            steps += 1
            if start_dists is None:
                start_dists = [
                    info['distError_' + str(i)] for i in range(self.agents)]
            if render:
                self.env.render()
                # self.env.viewer.render()
            for i in range(self.agents):
                if not isOver[i]:
                    sum_r[i] += r[i]
        return sum_r, start_dists, q_values, info


