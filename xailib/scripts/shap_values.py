from __future__ import annotations

import glob
import json
import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import shap
import supersuit as ss
from pettingzoo.mpe import simple_push_v3, simple_tag_v3, simple_v3
from stable_baselines3 import PPO

from utils.common.constants import ENV_SIMPLE_PUSH_FEATURES


def main(env_fn, render_mode: str = "human", feature_names: List[str] | None = None):
    figure_path = os.path.join("assets", "figures")
    observation_path = os.path.join("assets", "observations")

    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = ss.pad_observations_v0(env)

    env.reset()

    logging.info(
        f"\nStarting collecting observations on {str(env.metadata['name'])} (render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        logging.warning("Policy not found.")
        exit(0)
    model = PPO.load(latest_policy)

    with open(os.path.join(observation_path, "observations.json"), "r") as file:
        observations = json.load(file)

    def predict_fn(observations):
        action, _ = model.predict(observations)
        return action

    observations = np.array(observations)

    np.random.shuffle(observations)
    explainer = shap.KernelExplainer(predict_fn, observations)
    shap_values = explainer.shap_values(observations)
    shap.summary_plot(shap_values, observations, feature_names=feature_names)
    plt.savefig(figure_path)


if __name__ == "__main__":
    env_fn = simple_push_v3
    env_kwargs = {}

    feature_names = ENV_SIMPLE_PUSH_FEATURES

    main(env_fn, feature_names=feature_names)
