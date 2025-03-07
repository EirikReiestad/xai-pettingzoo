from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
import shap
from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo.mpe import simple_tag_v3, simple_v3


def main(env_fn, render_mode: str = "human"):
    figure_path = os.path.join("assets", "figures")

    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = ss.pad_observations_v0(env)

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

    with open("assets/observations/observations.json") as file:
        observations = json.load(file)

    def predict_fn(observations):
        action, _ = model.predict(observations)
        return action

    observations = np.array(observations)

    np.random.shuffle(observations)
    explainer = shap.KernelExplainer(predict_fn, observations)
    shap_values = explainer.shap_values(observations)
    shap.summary_plot(shap_values, observations)


if __name__ == "__main__":
    env_fn = simple_v3
    env_kwargs = {}

    main(env_fn)
