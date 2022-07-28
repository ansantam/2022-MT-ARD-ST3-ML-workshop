import gym
from gym import spaces
from gym.wrappers import RecordVideo
from IPython import display
from ipywidgets import GridspecLayout, Output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_training_videos():
    grid = GridspecLayout(1, 4)
    for i, eval_episode in enumerate([0, 60, 120, 300]):
        out = Output()
        with out:
            display.display(display.Video(f"utils/rl/lunar_lander_recordings/rl-video-episode-{eval_episode}.mp4"))
        grid[0, i] = out

    return grid


def plot_training_metrics():
    monitor = pd.read_csv("utils/rl/lunar_lander_evaluations/0.monitor.csv", index_col=False, skiprows=1)

    plt.figure(figsize=(13,4))
    plt.subplot(121)
    plt.title("Episode Rewards")
    plt.axhline(-100, color="tab:red", ls="--")
    plt.axhspan(-1000, -100, color="tab:red", alpha=0.2)
    plt.text(230/60, -700, "Crashing", color="tab:red")
    plt.axhline(140, color="tab:green", ls="--")
    plt.axhspan(140, 400, color="tab:green", alpha=0.2)
    plt.text(70/60, 250, "Landing", color="tab:green")
    plt.plot(monitor["t"]/60, monitor["r"])
    plt.ylim(-950, 350)
    plt.xlabel("Wall time (min)")
    plt.ylabel("Reward")
    plt.grid()
    plt.subplot(122)
    plt.title("Episode Lengths")
    plt.plot(monitor["t"]/60, monitor["l"], c="tab:orange")
    plt.xlabel("Wall time (min)")
    plt.ylabel("Steps")
    plt.grid()
    plt.tight_layout()
    plt.show()


def record_video(env):
    return RecordVideo(
        env,
        video_folder="utils/rl/lunar_lander_recordings",
        episode_trigger=lambda i: (i % 60) == 0
    )


class FilterAction(gym.ActionWrapper):

    def __init__(self, env, filter_indicies, replace="random"):
        super().__init__(env)

        self.filter_indicies = filter_indicies
        self.replace = replace

        self.action_space = spaces.Box(
            low=env.action_space.low[filter_indicies],
            high=env.action_space.high[filter_indicies],
            shape=env.action_space.low[filter_indicies].shape,
            dtype=env.action_space.dtype,
        )
    
    def action(self, action):
        if self.replace == "random":
            unfiltered = self.env.action_space.sample()
        else:
            unfiltered = np.full(self.env.action_space.shape, self.replace, dtype=self.env.action_space.dtype)
        
        unfiltered[self.filter_indicies] = action

        return unfiltered
