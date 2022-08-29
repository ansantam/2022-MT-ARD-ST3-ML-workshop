from datetime import datetime
import os
import pickle
import time

import cheetah
import cv2
import gym
from gym import spaces
from gym.wrappers import FlattenObservation, FilterObservation, FrameStack, RecordVideo, RescaleAction, TimeLimit
import imageio
from IPython import display
from ipywidgets import GridspecLayout, Output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import unwrap_wrapper
from stable_baselines3.common.monitor import Monitor
import yaml

import utils.rl.dummypydoocs as pydoocs
from .ARESlatticeStage3v1_9 import cell as ares_lattice


# ----- Notebook utilities ----------------------------------------------------

def display_acc_training_videos():
    grid = GridspecLayout(1, 4)
    for i, eval_episode in enumerate([0, 60, 120, 300]):
        out = Output()
        with out:
            display.display(display.Video(f"utils/rl/accelerator_recordings/rl-video-episode-{eval_episode}.mp4"))
        grid[0, i] = out

    return grid


def display_ll_training_videos():
    grid = GridspecLayout(1, 4)
    for i, eval_episode in enumerate([0, 60, 120, 300]):
        out = Output()
        with out:
            display.display(display.Video(f"utils/rl/lunar_lander_recordings/rl-video-episode-{eval_episode}.mp4"))
        grid[0, i] = out

    return grid


def display_video(filename):
    return display.Video(filename)


def make_lunar_lander_gif(model_path, gif_path):
    env = gym.make("LunarLander-v2")
    model = PPO.load(model_path)

    images = []
    obs = env.reset()
    img = env.render(mode="rgb_array")
    done = False
    while not done:
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, done ,_ = env.step(action)
        img = env.render(mode="rgb_array")

    imageio.mimsave(f"{gif_path}.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)
    env.close()

    
def make_lunar_lander_training_gifs():
    for steps in [1e5, 3e5, 1e6]:
        make_lunar_lander_gif(
            model_path=f"utils/rl/lunar_lander/checkpoints/rl_model_{int(steps)}_steps",
            gif_path=f"img/lunar_lander_trainig_{int(steps)}_steps",
        )


def plot_acc_training_metrics():
    monitor = pd.read_csv("utils/rl/accelerator_evaluations/0.monitor.csv", index_col=False, skiprows=1)

    plt.figure(figsize=(13,4))
    plt.subplot(121)
    plt.title("Episode Rewards")
    plt.plot(monitor["t"]/60, monitor["r"])
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


def plot_ll_training_metrics():
    monitor = pd.read_csv("utils/rl/lunar_lander/monitors/0.monitor.csv", index_col=False, skiprows=1)

    plt.figure(figsize=(13,4))
    plt.subplot(121)
    plt.title("Episode Rewards")
    plt.axhline(-100, color="tab:red", ls="--")
    plt.axhspan(-1000, -100, color="tab:red", alpha=0.2)
    plt.text(120/60, -700, "Crashing", color="tab:red")
    plt.axhline(140, color="tab:green", ls="--")
    plt.axhspan(140, 400, color="tab:green", alpha=0.2)
    plt.text(15/60, 250, "Landing", color="tab:green")
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


class ARESEA(gym.Env):
    """
    Base class for beam positioning and focusing on AREABSCR1 in the ARES EA.

    Parameters
    ----------
    action_mode : str
        How actions work. Choose `"direct"`, `"direct_unidirectional_quads"` or `"delta"`.
    magnet_init_mode : str
        Magnet initialisation on `reset`. Set to `None`, `"random"` or `"constant"`. The
        `"constant"` setting requires `magnet_init_values` to be set.
    magnet_init_values : np.ndarray
        Values to set magnets to on `reset`. May only be set when `magnet_init_mode` is set to 
        `"constant"`.
    reward_mode : str
        How to compute the reward. Choose from `"feedback"` or `"differential"`.
    target_beam_mode : str
        Setting of target beam on `reset`. Choose from `"constant"` or `"random"`. The `"constant"`
        setting requires `target_beam_values` to be set.
    """

    metadata = {
        "render.modes": ["rgb_array"],
        "video.frames_per_second": 2
    }

    def __init__(
        self,
        action_mode="direct",
        include_beam_image_in_info=True,
        magnet_init_mode=None,
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=2.4469e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=2.4469e-6,
        threshold_hold=1,
        w_done=1.0,
        w_mu_x=1.0,
        w_mu_x_in_threshold=1.0,
        w_mu_y=1.0,
        w_mu_y_in_threshold=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_x_in_threshold=1.0,
        w_sigma_y=1.0,
        w_sigma_y_in_threshold=1.0,
        w_time=1.0
    ):
        self.action_mode = action_mode
        self.include_beam_image_in_info = include_beam_image_in_info
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.target_mu_x_threshold = target_mu_x_threshold
        self.target_mu_y_threshold = target_mu_y_threshold
        self.target_sigma_x_threshold = target_sigma_x_threshold
        self.target_sigma_y_threshold = target_sigma_y_threshold
        self.threshold_hold = threshold_hold
        self.w_done = w_done
        self.w_mu_x = w_mu_x
        self.w_mu_x_in_threshold = w_mu_x_in_threshold
        self.w_mu_y = w_mu_y
        self.w_mu_y_in_threshold = w_mu_y_in_threshold
        self.w_on_screen = w_on_screen
        self.w_sigma_x = w_sigma_x
        self.w_sigma_x_in_threshold = w_sigma_x_in_threshold
        self.w_sigma_y = w_sigma_y
        self.w_sigma_y_in_threshold = w_sigma_y_in_threshold
        self.w_time = w_time

        # Create action space
        if self.action_mode == "direct":
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
            )
        elif self.action_mode == "direct_unidirectional_quads":
            self.action_space = spaces.Box(
                low=np.array([0, -72, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 0, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
            )
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32) * 0.1,
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32) * 0.1
            )
        else:
            raise ValueError(f"Invalid value \"{self.action_mode}\" for action_mode")

        # Create observation space
        obs_space_dict = {
            "beam": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            ),
            "magnets": self.action_space if self.action_mode.startswith("direct") else spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            ),
            "target": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            )
        }
        obs_space_dict.update(self.get_accelerator_observation_space())
        self.observation_space = spaces.Dict(obs_space_dict)

        # Setup the accelerator (either simulation or the actual machine)
        self.setup_accelerator()
    
    def reset(self):
        self.reset_accelerator()

        if self.magnet_init_mode == "constant":
            self.set_magnets(self.magnet_init_values)
        elif self.magnet_init_mode == "random":
            self.set_magnets(self.observation_space["magnets"].sample())
        elif self.magnet_init_mode == None:
            pass    # This really is intended to do nothing
        else:
            raise ValueError(f"Invalid value \"{self.magnet_init_mode}\" for magnet_init_mode")

        if self.target_beam_mode == "constant":
            self.target_beam = self.target_beam_values
        elif self.target_beam_mode == "random":
            self.target_beam = self.observation_space["target"].sample()
        else:
            raise ValueError(f"Invalid value \"{self.target_beam_mode}\" for target_beam_mode")

        # Update anything in the accelerator (mainly for running simulations)
        self.update_accelerator()

        self.initial_screen_beam = self.get_beam_parameters()
        self.previous_beam = self.initial_screen_beam
        self.is_in_threshold_history = []
        self.steps_taken = 0

        observation = {
            "beam": self.initial_screen_beam.astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
            "target": self.target_beam.astype("float32")
        }
        observation.update(self.get_accelerator_observation())

        return observation

    def step(self, action):
        # Perform action
        if self.action_mode == "direct":
            self.set_magnets(action)
        elif self.action_mode == "direct_unidirectional_quads":
            self.set_magnets(action)
        elif self.action_mode == "delta":
            magnet_values = self.get_magnets()
            self.set_magnets(magnet_values + action)
        else:
            raise ValueError(f"Invalid value \"{self.action_mode}\" for action_mode")

        # Run the simulation
        self.update_accelerator()

        current_beam = self.get_beam_parameters()
        self.steps_taken += 1

        # Build observation
        observation = {
            "beam": current_beam.astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
            "target": self.target_beam.astype("float32")
        }
        observation.update(self.get_accelerator_observation())

        # For readibility in computations below
        cb = current_beam
        ib = self.initial_screen_beam
        pb = self.previous_beam
        tb = self.target_beam

        # Compute if done (beam within threshold for a certain time)
        threshold = np.array([
            self.target_mu_x_threshold,
            self.target_sigma_x_threshold,
            self.target_mu_y_threshold,
            self.target_sigma_y_threshold,
        ])
        is_in_threshold = np.abs(cb - tb) < threshold
        self.is_in_threshold_history.append(is_in_threshold)
        is_stable_in_threshold = bool(np.array(self.is_in_threshold_history[-self.threshold_hold:]).all())
        done = is_stable_in_threshold and len(self.is_in_threshold_history) > 5

        # Compute reward
        on_screen_reward = -(not self.is_beam_on_screen())
        time_reward = -1
        done_reward = done * (25 - self.steps_taken) / 25
        if self.reward_mode == "differential":
            mu_x_reward = (abs(pb[0] - tb[0]) - abs(cb[0] - tb[0])) / abs(ib[0] - tb[0])
            sigma_x_reward = (abs(pb[1] - tb[1]) - abs(cb[1] - tb[1])) / abs(ib[1] - tb[1])
            mu_y_reward = (abs(pb[2] - tb[2]) - abs(cb[2] - tb[2])) / abs(ib[2] - tb[2])
            sigma_y_reward = (abs(pb[3] - tb[3]) - abs(cb[3] - tb[3])) / abs(ib[3] - tb[3])
        elif self.reward_mode == "feedback":
            mu_x_reward = - abs((cb[0] - tb[0]) / (ib[0] - tb[0]))
            sigma_x_reward = - abs((cb[1] - tb[1]) / (ib[1] - tb[1]))
            mu_y_reward = - abs((cb[2] - tb[2]) / (ib[2] - tb[2]))
            sigma_y_reward = - abs((cb[3] - tb[3]) / (ib[3] - tb[3]))
        else:
            raise ValueError(f"Invalid value \"{self.reward_mode}\" for reward_mode")

        reward = 0
        reward += self.w_on_screen * on_screen_reward
        reward += self.w_mu_x * mu_x_reward
        reward += self.w_sigma_x * sigma_x_reward
        reward += self.w_mu_y * mu_y_reward
        reward += self.w_sigma_y * sigma_y_reward * self.w_time * time_reward
        reward += self.w_mu_x_in_threshold * is_in_threshold[0]
        reward += self.w_sigma_x_in_threshold * is_in_threshold[1]
        reward += self.w_mu_y_in_threshold * is_in_threshold[2]
        reward += self.w_sigma_y_in_threshold * is_in_threshold[3]
        reward += self.w_done * done_reward
        reward = float(reward)

        # Put together info
        info = {
            "binning": self.get_binning(),
            "mu_x_reward": mu_x_reward,
            "mu_y_reward": mu_y_reward,
            "on_screen_reward": on_screen_reward,
            "pixel_size": self.get_pixel_size(),
            "screen_resolution": self.get_screen_resolution(),
            "sigma_x_reward": sigma_x_reward,
            "sigma_y_reward": sigma_y_reward,
            "time_reward": time_reward,
        }
        if self.include_beam_image_in_info:
            info["beam_image"] = self.get_beam_image()
        info.update(self.get_accelerator_info())
        
        self.previous_beam = current_beam

        return observation, reward, done, info
    
    def render(self, mode="human"):
        assert mode == "rgb_array" or mode == "human"

        binning = self.get_binning()
        pixel_size = self.get_pixel_size()
        resolution = self.get_screen_resolution()
        
        # Read screen image and make 8-bit RGB
        img = self.get_beam_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:,:,np.newaxis], 3, axis=-1)

        # Redraw beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw desired ellipse
        tb = self.target_beam
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(tb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(tb[1] / pixel_size_b4[0])
        e_pos_y = int(-tb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(tb[3] / pixel_size_b4[1])
        blue = (255, 204, 79)
        img = cv2.ellipse(img, (e_pos_x,e_pos_y), (e_width_x,e_width_y), 0, 0, 360, blue, 2)

        # Draw beam ellipse
        cb = self.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(cb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(cb[1] / pixel_size_b4[0])
        e_pos_y = int(-cb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(cb[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(img, (e_pos_x,e_pos_y), (e_width_x,e_width_y), 0, 0, 360, red, 2)
        
        # Adjust aspect ratio
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width,img.shape[0]))

        # Add magnet values and beam parameters
        magnets = self.get_magnets()
        padding = np.full((int(img.shape[0]*0.27),img.shape[1],3), fill_value=255, dtype=np.uint8)
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        red = (0, 0, 255)
        green = (0, 255, 0)
        img = cv2.putText(img, f"Q1={magnets[0]:.2f}", (15,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"Q2={magnets[1]:.2f}", (215,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"CV={magnets[2]*1e3:.2f}", (415,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"Q3={magnets[3]:.2f}", (615,545), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        img = cv2.putText(img, f"CH={magnets[4]*1e3:.2f}", (15,585), cv2.FONT_HERSHEY_SIMPLEX, 1, black)
        mu_x_color = black
        if self.target_mu_x_threshold != np.inf:
            mu_x_color = green if abs(cb[0] - tb[0]) < self.target_mu_x_threshold else red
        img = cv2.putText(img, f"mx={cb[0]*1e3:.2f}", (15,625), cv2.FONT_HERSHEY_SIMPLEX, 1, mu_x_color)
        sigma_x_color = black
        if self.target_sigma_x_threshold != np.inf:
            sigma_x_color = green if abs(cb[1] - tb[1]) < self.target_sigma_x_threshold else red
        img = cv2.putText(img, f"sx={cb[1]*1e3:.2f}", (215,625), cv2.FONT_HERSHEY_SIMPLEX, 1, sigma_x_color)
        mu_y_color = black
        if self.target_mu_y_threshold != np.inf:
            mu_y_color = green if abs(cb[2] - tb[2]) < self.target_mu_y_threshold else red
        img = cv2.putText(img, f"my={cb[2]*1e3:.2f}", (415,625), cv2.FONT_HERSHEY_SIMPLEX, 1, mu_y_color)
        sigma_y_color = black
        if self.target_sigma_y_threshold != np.inf:
            sigma_y_color = green if abs(cb[3] - tb[3]) < self.target_sigma_y_threshold else red
        img = cv2.putText(img, f"sy={cb[3]*1e3:.2f}", (615,625), cv2.FONT_HERSHEY_SIMPLEX, 1, sigma_y_color)

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def is_beam_on_screen(self):
        """
        Return `True` when the beam is on the screen and `False` when it isn't.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError
    
    def setup_accelerator(self):
        """
        Prepare the accelerator for use with the environment. Should mostly be used for setting up
        simulations.

        Override with backend-specific imlementation. Optional.
        """
    
    def get_magnets(self):
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the accelerator.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def set_magnets(self, magnets):
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets appear in
        the accelerator.

        When applicable, this method should block until the magnet values are acutally set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def reset_accelerator(self):
        """
        Code that should set the accelerator up for a new episode. Run when the `reset` is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or simular
        things.

        Override with backend-specific imlementation. Optional.
        """
    
    def update_accelerator(self):
        """
        Update accelerator metrics for later use. Use this to run the simulation or cache the beam
        image.

        Override with backend-specific imlementation. Optional.
        """
    
    def get_beam_parameters(self):
        """
        Get the beam parameters measured on the diagnostic screen as NumPy array grouped by
        dimension (e.g. mu_x, sigma_x, mu_y, sigma_y).

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError
    
    def get_incoming_parameters(self):
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order energy, mu_x,
        mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s, sigma_p.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_misalignments(self):
        """
        Get misalignments of the quadrupoles and the diagnostic screen as NumPy array in order
        AREAMQZM1.misalignment.x, AREAMQZM1.misalignment.y, AREAMQZM2.misalignment.x,
        AREAMQZM2.misalignment.y, AREAMQZM3.misalignment.x, AREAMQZM3.misalignment.y,
        AREABSCR1.misalignment.x, AREABSCR1.misalignment.y.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_beam_image(self):
        """
        Retreive the beam image as a 2-dimensional NumPy array.

        Note that if reading the beam image is expensive, it is best to cache the image in the
        `update_accelerator` method and the read the cached variable here.

        Ideally, the pixel values should look somewhat similar to the 12-bit values from the real
        screen camera.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_binning(self):
        """
        Return binning currently set on the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError
    
    def get_screen_resolution(self):
        """
        Return (binned) resolution of the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError
    
    def get_pixel_size(self):
        """
        Return the (binned) size of the area on the diagnostic screen covered by one pixel as NumPy
        array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_accelerator_observation_space(self):
        """
        Return a dictionary of aditional observation spaces for observations from the accelerator
        backend, e.g. incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_observation(self):
        """
        Return a dictionary of aditional observations from the accelerator backend, e.g. incoming
        beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_info(self):
        """
        Return a dictionary of aditional info from the accelerator backend, e.g. incoming beam and
        misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}


class ARESEACheetah(ARESEA):

    def __init__(
        self,
        incoming_mode="random",
        incoming_values=None,
        misalignment_mode="random",
        misalignment_values=None,
        action_mode="direct",
        include_beam_image_in_info=False,
        magnet_init_mode="zero",
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=2.4469e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=2.4469e-6,
        threshold_hold=1,
        w_done=1.0,
        w_mu_x=1.0,
        w_mu_x_in_threshold=1.0,
        w_mu_y=1.0,
        w_mu_y_in_threshold=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_x_in_threshold=1.0,
        w_sigma_y=1.0,
        w_sigma_y_in_threshold=1.0,
        w_time=1.0,
    ):
        super().__init__(
            action_mode=action_mode,
            include_beam_image_in_info=include_beam_image_in_info,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            reward_mode=reward_mode,
            target_beam_mode=target_beam_mode,
            target_beam_values=target_beam_values,
            target_mu_x_threshold=target_mu_x_threshold,
            target_mu_y_threshold=target_mu_y_threshold,
            target_sigma_x_threshold=target_sigma_x_threshold,
            target_sigma_y_threshold=target_sigma_y_threshold,
            threshold_hold=threshold_hold,
            w_done=w_done,
            w_mu_x=w_mu_x,
            w_mu_x_in_threshold=w_mu_x_in_threshold,
            w_mu_y=w_mu_y,
            w_mu_y_in_threshold=w_mu_y_in_threshold,
            w_on_screen=w_on_screen,
            w_sigma_x=w_sigma_x,
            w_sigma_x_in_threshold=w_sigma_x_in_threshold,
            w_sigma_y=w_sigma_y,
            w_sigma_y_in_threshold=w_sigma_y_in_threshold,
            w_time=w_time,
        )

        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values

        # Create particle simulation
        self.simulation = cheetah.Segment.from_ocelot(
            ares_lattice,
            warnings=False,
            device="cpu"
        ).subcell("AREASOLA1", "AREABSCR1")
        self.simulation.AREABSCR1.resolution = (2448, 2040)
        self.simulation.AREABSCR1.pixel_size = (3.3198e-6, 2.4469e-6)
        self.simulation.AREABSCR1.is_active = True
        self.simulation.AREABSCR1.binning = 4
        self.simulation.AREABSCR1.is_active = True

    def is_beam_on_screen(self):
        screen = self.simulation.AREABSCR1
        beam_position = np.array([screen.read_beam.mu_x, screen.read_beam.mu_y])
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        return np.all(np.abs(beam_position) < limits)
    
    def get_magnets(self):
        return np.array([
            self.simulation.AREAMQZM1.k1,
            self.simulation.AREAMQZM2.k1,
            self.simulation.AREAMCVM1.angle,
            self.simulation.AREAMQZM3.k1,
            self.simulation.AREAMCHM1.angle
        ])

    def set_magnets(self, magnets):
        self.simulation.AREAMQZM1.k1 = magnets[0]
        self.simulation.AREAMQZM2.k1 = magnets[1]
        self.simulation.AREAMCVM1.angle = magnets[2]
        self.simulation.AREAMQZM3.k1 = magnets[3]
        self.simulation.AREAMCHM1.angle = magnets[4]

    def reset_accelerator(self):
        # New domain randomisation
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            incoming_parameters = self.observation_space["incoming"].sample()
        else:
            raise ValueError(f"Invalid value \"{self.incoming_mode}\" for incoming_mode")
        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=incoming_parameters[0],
            mu_x=incoming_parameters[1],
            mu_xp=incoming_parameters[2],
            mu_y=incoming_parameters[3],
            mu_yp=incoming_parameters[4],
            sigma_x=incoming_parameters[5],
            sigma_xp=incoming_parameters[6],
            sigma_y=incoming_parameters[7],
            sigma_yp=incoming_parameters[8],
            sigma_s=incoming_parameters[9],
            sigma_p=incoming_parameters[10],
        )

        if self.misalignment_mode == "constant":
            misalignments = self.misalignment_values
        elif self.misalignment_mode == "random":
            misalignments = self.observation_space["misalignments"].sample()
        else:
            raise ValueError(f"Invalid value \"{self.misalignment_mode}\" for misalignment_mode")
        self.simulation.AREAMQZM1.misalignment = misalignments[0:2]
        self.simulation.AREAMQZM2.misalignment = misalignments[2:4]
        self.simulation.AREAMQZM3.misalignment = misalignments[4:6]
        self.simulation.AREABSCR1.misalignment = misalignments[6:8]
    
    def update_accelerator(self):
        self.simulation(self.incoming)
    
    def get_beam_parameters(self):
        return np.array([
            self.simulation.AREABSCR1.read_beam.mu_x,
            self.simulation.AREABSCR1.read_beam.sigma_x,
            self.simulation.AREABSCR1.read_beam.mu_y,
            self.simulation.AREABSCR1.read_beam.sigma_y
        ])
    
    def get_incoming_parameters(self):
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array([
            self.incoming.energy,
            self.incoming.mu_x,
            self.incoming.mu_xp,
            self.incoming.mu_y,
            self.incoming.mu_yp,
            self.incoming.sigma_x,
            self.incoming.sigma_xp,
            self.incoming.sigma_y,
            self.incoming.sigma_yp,
            self.incoming.sigma_s,
            self.incoming.sigma_p
        ])

    def get_misalignments(self):
        return np.array([
            self.simulation.AREAMQZM1.misalignment[0],
            self.simulation.AREAMQZM1.misalignment[1],
            self.simulation.AREAMQZM2.misalignment[0],
            self.simulation.AREAMQZM2.misalignment[1],
            self.simulation.AREAMQZM3.misalignment[0],
            self.simulation.AREAMQZM3.misalignment[1],
            self.simulation.AREABSCR1.misalignment[0],
            self.simulation.AREABSCR1.misalignment[1]
        ], dtype=np.float32)

    def get_beam_image(self):
        # Beam image to look like real image by dividing by goodlooking number and scaling to 12 bits)
        return self.simulation.AREABSCR1.reading / 1e9 * 2**12

    def get_binning(self):
        return np.array(self.simulation.AREABSCR1.binning)
    
    def get_screen_resolution(self):
        return np.array(self.simulation.AREABSCR1.resolution) / self.get_binning()
    
    def get_pixel_size(self):
        return np.array(self.simulation.AREABSCR1.pixel_size) * self.get_binning()

    def get_accelerator_observation_space(self):
        return {
            "incoming": spaces.Box(
                low=np.array([80e6, -1e-3, -1e-4, -1e-3, -1e-4, 1e-5, 1e-6, 1e-5, 1e-6, 1e-6, 1e-4], dtype=np.float32),
                high=np.array([160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3], dtype=np.float32)
            ),
            "misalignments": spaces.Box(low=-2e-3, high=2e-3, shape=(8,)),
        }

    def get_accelerator_observation(self):
        return {
            "incoming": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }


# ----- From ea_train.py (with slight modifications) --------------------------

def make_env(config, record_video=False, monitor_filename=None):
    env = ARESEACheetah(
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        misalignment_mode=config["misalignment_mode"],
        misalignment_values=config["misalignment_values"],
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        reward_mode=config["reward_mode"],
        target_beam_mode=config["target_beam_mode"],
        target_beam_values=config["target_beam_values"],
        target_mu_x_threshold=config["target_mu_x_threshold"],
        target_mu_y_threshold=config["target_mu_y_threshold"],
        target_sigma_x_threshold=config["target_sigma_x_threshold"],
        target_sigma_y_threshold=config["target_sigma_y_threshold"],
        threshold_hold=config["threshold_hold"],
        w_mu_x=config["w_mu_x"],
        w_mu_x_in_threshold=config["w_mu_x_in_threshold"],
        w_mu_y=config["w_mu_y"],
        w_mu_y_in_threshold=config["w_mu_y_in_threshold"],
        w_on_screen=config["w_on_screen"],
        w_sigma_x=config["w_sigma_x"],
        w_sigma_x_in_threshold=config["w_sigma_x_in_threshold"],
        w_sigma_y=config["w_sigma_y"],
        w_sigma_y_in_threshold=config["w_sigma_y_in_threshold"],
        w_time=config["w_time"],
    )
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    if config["time_limit"] is not None:
        env = TimeLimit(env, config["time_limit"])
    env = FlattenObservation(env)
    if config["frame_stack"] is not None:
        env = FrameStack(env, config["frame_stack"])
    if config["rescale_action"] is not None:
        env = RescaleAction(env, config["rescale_action"][0], config["rescale_action"][1])
    if not os.path.exists("utils/rl/accelerator_evaluations"):
        os.mkdir("utils/rl/accelerator_evaluations")
    env = Monitor(env, filename=monitor_filename)
    if record_video:
        env = RecordVideo(env, video_folder="utils/rl/accelerator_recordings")
    return env


def read_from_yaml(path):
    with open(f"{path}.yaml", "r") as f:
        data = yaml.parse(f.read())
    return data


def save_to_yaml(data, path):
    with open(f"{path}.yaml", "w") as f:
        yaml.dump(data, f)


# ----- From ea_optimize.py (with slight modificiations) ----------------------

def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=3.3198e-6,
    target_mu_y_threshold=3.3198e-6,
    target_sigma_x_threshold=3.3198e-6,
    target_sigma_y_threshold=3.3198e-6,
    max_steps=50,
    model_name="polished-donkey-996",
    logbook=False,
    callback=None,
):
    """
    Function used for optimisation during operation.

    Note: Current version only works for polished-donkey-996.
    """
    # config = read_from_yaml(f"models/{model}/config")
    assert model_name == "polished-donkey-996", "Current version only works for polished-donkey-996."
    
    # Load the model
    model = TD3.load(f"utils/rl/{model_name}/model")
    
    # Create the environment
    env = ARESEADOOCS(
        action_mode="delta",
        magnet_init_mode="constant",
        magnet_init_values=np.array([10, -10, 0, 10, 0]),
        reward_mode="differential",
        target_beam_mode="constant",
        target_beam_values=np.array([target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]),
        target_mu_x_threshold=target_mu_x_threshold,
        target_mu_y_threshold=target_mu_y_threshold,
        target_sigma_x_threshold=target_sigma_x_threshold,
        target_sigma_y_threshold=target_sigma_y_threshold,
    )
    if max_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_steps)
    env = RecordEpisode(env)
    env = RecordVideo(env, "utils/rl/polished-donkey_recordings")
    env = FlattenObservation(env)
    env = PolishedDonkeyCompatibility(env)
    env = NotVecNormalize(env, f"utils/rl/{model_name}/vec_normalize.pkl")
    env = RescaleAction(env, -1, 1)

    # Actual optimisation
    t_start = datetime.now()
    observation = env.reset()
    beam_image_before = env.get_beam_image()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    t_end = datetime.now()

    recording = unwrap_wrapper(env, RecordEpisode)
    if logbook:
        report_ea_optimization_to_logbook(
            model_name,
            t_start,
            t_end,
            recording.observations,
            recording.infos,
            beam_image_before,
            target_mu_x_threshold,
            target_sigma_x_threshold,
            target_mu_y_threshold,
            target_sigma_y_threshold,
        )

    env.close()


def report_ea_optimization_to_logbook(
    model_name,
    t_start,
    t_end,
    observations,
    infos,
    beam_image_before,
    target_mu_x_threshold,
    target_sigma_x_threshold,
    target_mu_y_threshold,
    target_sigma_y_threshold,
):
    # Create text message
    beam_before = observations[0]["beam"]
    beam_after = observations[-1]["beam"]
    target_beam = observations[0]["target"]
    target_threshold = np.array([
        target_mu_x_threshold,
        target_sigma_x_threshold,
        target_mu_y_threshold,
        target_sigma_y_threshold,
    ])
    final_magnets = observations[-1]["magnets"]
    steps_taken = len(observations) - 1
    success = np.abs(beam_after - target_beam) < target_threshold

    msg = f"""Reinforcement learning agent optimised beam on AREABSCR1
    
Agent: {model_name}
Start time: {t_start}
Time taken: {t_end - t_start}
No. of steps: {steps_taken}

Beam before:
    mu_x    = {beam_before[0] * 1e3: 5.4f} mm
    sigma_x = {beam_before[1] * 1e3: 5.4f} mm
    mu_y    = {beam_before[2] * 1e3: 5.4f} mm
    sigma_y = {beam_before[3] * 1e3: 5.4f} mm

Beam after:
    mu_x    = {beam_after[0] * 1e3: 5.4f} mm
    sigma_x = {beam_after[1] * 1e3: 5.4f} mm
    mu_y    = {beam_after[2] * 1e3: 5.4f} mm
    sigma_y = {beam_after[3] * 1e3: 5.4f} mm

Target beam:
    mu_x    = {target_beam[0] * 1e3: 5.4f} mm    (e = {target_threshold[0] * 1e3:5.4f} mm) {';)' if success[0] else ':/'}
    sigma_x = {target_beam[1] * 1e3: 5.4f} mm    (e = {target_threshold[0] * 1e3:5.4f} mm) {';)' if success[1] else ':/'}
    mu_y    = {target_beam[2] * 1e3: 5.4f} mm    (e = {target_threshold[0] * 1e3:5.4f} mm) {';)' if success[2] else ':/'}
    sigma_y = {target_beam[3] * 1e3: 5.4f} mm    (e = {target_threshold[0] * 1e3:5.4f} mm) {';)' if success[3] else ':/'}

Final magnet settings:
    AREAMQZM1 strength = {final_magnets[0]: 8.4f} 1/m^2
    AREAMQZM2 strength = {final_magnets[1]: 8.4f} 1/m^2
    AREAMCVM1 kick     = {final_magnets[2] * 1e3: 8.4f} mrad
    AREAMQZM3 strength = {final_magnets[3]: 8.4f} 1/m^2
    AREAMCHM1 kick     = {final_magnets[4] * 1e3: 8.4f} mrad
    """

    # Create plot as jpg
    fig, axs = plt.subplots(1, 5, figsize=(30,4))
    plot_quadrupole_history(axs[0], observations)
    plot_steerer_history(axs[1], observations)
    plot_beam_history(axs[2], observations)
    plot_beam_image(axs[3], beam_image_before, screen_resolution=infos[0]["screen_resolution"],
                    pixel_size=infos[0]["pixel_size"], title="Beam at Reset (Background Removed)")
    plot_beam_image(axs[4], infos[-1]["beam_image"], screen_resolution=infos[-1]["screen_resolution"],
                    pixel_size=infos[-1]["pixel_size"], title="Beam After (Background Removed)")
    fig.tight_layout()
    
    # buf = BytesIO()
    # fig.savefig(buf, dpi=300, format="jpg")
    # buf.seek(0)
    # img = bytes(buf.read())

    # Send to logbook
    # send_to_elog(
    #     elog="areslog",
    #     author="Autonomous ARES",
    #     title="RL-based Beam Optimisation on AREABSCR1",
    #     severity="NONE",
    #     text=msg,
    #     image=img,
    # )

    print(msg)
    plt.show()


def plot_quadrupole_history(ax, observations):
    areamqzm1 = [obs["magnets"][0] for obs in observations]
    areamqzm2 = [obs["magnets"][1] for obs in observations]
    areamqzm3 = [obs["magnets"][3] for obs in observations]

    steps = np.arange(len(observations))

    ax.set_title("Quadrupoles")
    ax.set_xlim([0, len(steps)])
    ax.set_xlabel("Step")
    ax.set_ylabel("Strength (1/m^2)")
    ax.plot(steps, areamqzm1, label="AREAMQZM1")
    ax.plot(steps, areamqzm2, label="AREAMQZM2")
    ax.plot(steps, areamqzm3, label="AREAMQZM3")
    ax.legend()
    ax.grid(True)


def plot_steerer_history(ax, observations):
    areamcvm1 = np.array([obs["magnets"][2] for obs in observations])
    areamchm2 = np.array([obs["magnets"][4] for obs in observations])

    steps = np.arange(len(observations))

    ax.set_title("Steerers")
    ax.set_xlabel("Step")
    ax.set_ylabel("Kick (mrad)")
    ax.set_xlim([0, len(steps)])
    ax.plot(steps, areamcvm1*1e3, label="AREAMCVM1")
    ax.plot(steps, areamchm2*1e3, label="AREAMCHM2")
    ax.legend()
    ax.grid(True)


def plot_beam_history(ax, observations):
    mu_x = np.array([obs["beam"][0] for obs in observations])
    sigma_x = np.array([obs["beam"][1] for obs in observations])
    mu_y = np.array([obs["beam"][2] for obs in observations])
    sigma_y = np.array([obs["beam"][3] for obs in observations])

    target_beam = observations[0]["target"]

    steps = np.arange(len(observations))

    ax.set_title("Beam Parameters")
    ax.set_xlim([0, len(steps)])
    ax.set_xlabel("Step")
    ax.set_ylabel("(mm)")
    ax.plot(steps, mu_x*1e3, label=r"$\mu_x$", c="tab:blue")
    ax.plot(steps, [target_beam[0]*1e3]*len(steps), ls="--", c="tab:blue")
    ax.plot(steps, sigma_x*1e3, label=r"$\sigma_x$", c="tab:orange")
    ax.plot(steps, [target_beam[1]*1e3]*len(steps), ls="--", c="tab:orange")
    ax.plot(steps, mu_y*1e3, label=r"$\mu_y$", c="tab:green")
    ax.plot(steps, [target_beam[2]*1e3]*len(steps), ls="--", c="tab:green")
    ax.plot(steps, sigma_y*1e3, label=r"$\sigma_y$", c="tab:red")
    ax.plot(steps, [target_beam[3]*1e3]*len(steps), ls="--", c="tab:red")
    ax.legend()
    ax.grid(True)
    

def plot_beam_image(ax, img, screen_resolution, pixel_size, title="Beam Image"):
    screen_size = screen_resolution * pixel_size

    ax.set_title(title)
    ax.set_xlabel("(mm)")
    ax.set_ylabel("(mm)")
    ax.imshow(
        img,
        vmin=0,
        aspect="equal",
        interpolation="none",
        extent=(
            -screen_size[0] / 2 * 1e3,
            screen_size[0] / 2 * 1e3,
            -screen_size[1] / 2 * 1e3,
            screen_size[1] / 2 * 1e3,
        ),
    )


class ARESEADOOCS(ARESEA):
    
    def __init__(
        self,
        action_mode="direct",
        include_beam_image_in_info=True,
        magnet_init_mode="zero",
        magnet_init_values=None,
        reward_mode="differential",
        target_beam_mode="random",
        target_beam_values=None,
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=2.4469e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=2.4469e-6,
        threshold_hold=1,
        w_done=1.0,
        w_mu_x=1.0,
        w_mu_x_in_threshold=1.0,
        w_mu_y=1.0,
        w_mu_y_in_threshold=1.0,
        w_on_screen=1.0,
        w_sigma_x=1.0,
        w_sigma_x_in_threshold=1.0,
        w_sigma_y=1.0,
        w_sigma_y_in_threshold=1.0,
        w_time=1.0,
    ):
        super().__init__(
            action_mode=action_mode,
            include_beam_image_in_info=include_beam_image_in_info,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            reward_mode=reward_mode,
            target_beam_mode=target_beam_mode,
            target_beam_values=target_beam_values,
            target_mu_x_threshold=target_mu_x_threshold,
            target_mu_y_threshold=target_mu_y_threshold,
            target_sigma_x_threshold=target_sigma_x_threshold,
            target_sigma_y_threshold=target_sigma_y_threshold,
            threshold_hold=threshold_hold,
            w_done=w_done,
            w_mu_x=w_mu_x,
            w_mu_x_in_threshold=w_mu_x_in_threshold,
            w_mu_y=w_mu_y,
            w_mu_y_in_threshold=w_mu_y_in_threshold,
            w_on_screen=w_on_screen,
            w_sigma_x=w_sigma_x,
            w_sigma_x_in_threshold=w_sigma_x_in_threshold,
            w_sigma_y=w_sigma_y,
            w_sigma_y_in_threshold=w_sigma_y_in_threshold,
            w_time=w_time,
        )

    def is_beam_on_screen(self):
        return True # TODO find better logic

    def get_magnets(self):
        return np.array([
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV")["data"] / 1000,
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV")["data"],
            pydoocs.read("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV")["data"] / 1000
        ])
    
    def set_magnets(self, magnets):
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.SP", magnets[0])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.SP", magnets[1])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP", magnets[2] * 1000)
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.SP", magnets[3])
        pydoocs.write("SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.SP", magnets[4] * 1000)

        # Wait until magnets have reached their setpoints
        
        time.sleep(3.0) # Wait for magnets to realise they received a command

        magnets = ["AREAMQZM1", "AREAMQZM2", "AREAMCVM1", "AREAMQZM3", "AREAMCHM1"]

        are_busy = [True] * 5
        are_ps_on = [True] * 5
        while any(are_busy) or not all(are_ps_on):
            are_busy = [pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/BUSY")["data"] for magnet in magnets]
            are_ps_on = [pydoocs.read(f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/PS_ON")["data"] for magnet in magnets]

    def update_accelerator(self):
        self.beam_image = self.capture_clean_beam_image()

    def get_beam_parameters(self):
        img = self.get_beam_image()
        pixel_size = self.get_pixel_size()

        parameters = {}
        for axis, direction in zip([0,1], ["x","y"]):
            projection = img.sum(axis=axis)
            minfiltered = minimum_filter1d(projection, size=5, mode="nearest")
            filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

            half_values, = np.where(filtered >= 0.5 * filtered.max())

            if len(half_values) > 0:
                fwhm_pixel = half_values[-1] - half_values[0]
                center_pixel = half_values[0] + fwhm_pixel / 2
            else:
                fwhm_pixel = 42     # TODO figure out what to do with these
                center_pixel = 42

            parameters[f"mu_{direction}"] = (center_pixel - len(filtered) / 2) * pixel_size[axis]
            parameters[f"sigma_{direction}"] = fwhm_pixel / 2.355 * pixel_size[axis]
            
        parameters["mu_y"] = -parameters["mu_y"]

        return np.array([
            parameters["mu_x"],
            parameters["sigma_x"],
            parameters["mu_y"],
            parameters["sigma_y"]
        ])

    def get_beam_image(self):
        return self.beam_image

    def get_binning(self):
        return np.array((
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL")["data"],
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL")["data"]
        ))

    def get_screen_resolution(self):
        return np.array([
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH")["data"],
            pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT")["data"]
        ])
    
    def get_pixel_size(self):
        return np.array([
            abs(pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/X.POLY_SCALE")["data"][2]) / 1000,
            abs(pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/Y.POLY_SCALE")["data"][2]) / 1000
        ]) * self.get_binning()

    def capture_clean_beam_image(self, average=5):
        """
        Capture a clean image of the beam from the screen using `average` images with beam on and
        `average` images of the background and then removing the background.
        
        Saves the image to a property of the object.
        """
         # Laser off
        self.set_cathode_laser(False)
        background_images = self.capture_interval(n=average, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        self.set_cathode_laser(True)
        beam_images = self.capture_interval(n=average, dt=0.1)
        median_beam = np.median(beam_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16-1)
        flipped = np.flipud(removed)
        
        return flipped
    
    def capture_interval(self, n, dt):
        """Capture `n` images from the screen and wait `dt` seconds in between them."""
        images = []
        for _ in range(n):
            images.append(self.capture_screen())
            time.sleep(dt)
        return np.array(images)
    
    def capture_screen(self):
        """Capture and image from the screen."""
        return pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ")["data"]

    def set_cathode_laser(self, setto):
        """Sets the bool switch of the cathode laser event to `setto` and waits a second."""
        address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
        bits = pydoocs.read(address)["data"]
        bits[0] = 1 if setto else 0
        pydoocs.write(address, bits)
        time.sleep(1)


# ----- From ares-ea-rl utils.py ----------------------------------------------

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


class NotVecNormalize(gym.Wrapper):
    """
    Normal Gym wrapper that replicates the functionality of Stable Baselines3's VecNormalize wrapper
    for non VecEnvs (i.e. `gym.Env`) in production.
    """
    
    def __init__(self, env, path):
        super().__init__(env)

        with open(path, "rb") as file_handler:
            self.vec_normalize = pickle.load(file_handler)

    def reset(self):
        observation = self.env.reset()
        return self.vec_normalize.normalize_obs(observation)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.vec_normalize.normalize_obs(observation)
        reward = self.vec_normalize.normalize_reward(reward)
        return observation, reward, done, info


class PolishedDonkeyCompatibility(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=np.array([
                super().observation_space.low[4],
                super().observation_space.low[5],
                super().observation_space.low[7],
                super().observation_space.low[6],
                super().observation_space.low[8],
                super().observation_space.low[9],
                super().observation_space.low[11],
                super().observation_space.low[10],
                super().observation_space.low[12],
                super().observation_space.low[0],
                super().observation_space.low[2],
                super().observation_space.low[1],
                super().observation_space.low[3],
            ]),
            high=np.array([
                super().observation_space.high[4],
                super().observation_space.high[5],
                super().observation_space.high[7],
                super().observation_space.high[6],
                super().observation_space.high[8],
                super().observation_space.high[9],
                super().observation_space.high[11],
                super().observation_space.high[10],
                super().observation_space.high[12],
                super().observation_space.high[0],
                super().observation_space.high[2],
                super().observation_space.high[1],
                super().observation_space.high[3],
            ])
        )

        self.action_space = spaces.Box(
            low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32) * 0.1,
            high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32) * 0.1,
        )

    def reset(self):
        return self.observation(super().reset())
    
    def step(self, action):
        observation, reward, done, info = super().step(self.action(action))
        return self.observation(observation), reward, done, info
    
    def observation(self, observation):
        return np.array([
            observation[4],
            observation[5],
            observation[7],
            observation[6],
            observation[8],
            observation[9],
            observation[11],
            observation[10],
            observation[12],
            observation[0],
            observation[2],
            observation[1],
            observation[3],
        ])
    
    def action(self, action):
        return np.array([
            action[0],
            action[1],
            action[3],
            action[2],
            action[4],
        ])


class RecordEpisode(gym.Wrapper):
    """Wrapper for recording epsiode data such as observations, rewards, infos and actions."""

    def __init__(self, env):
        super().__init__(env)
        
        self.has_previously_run = False

    def reset(self):
        if self.has_previously_run:
            self.previous_observations = self.observations
            self.previous_rewards = self.rewards
            self.previous_infos = self.infos
            self.previous_actions = self.actions
            self.previous_t_start = self.t_start
            self.previous_t_end = datetime.now()
            self.previous_steps_taken = self.steps_taken

        observation = self.env.reset()

        self.observations = [observation]
        self.rewards = []
        self.infos = []
        self.actions = []
        self.t_start = datetime.now()
        self.t_end = None
        self.steps_taken = 0

        self.has_previously_run = True

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.infos.append(info)
        self.actions.append(action)
        self.steps_taken += 1

        return observation, reward, done, info
