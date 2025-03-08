from __future__ import annotations

import glob
import json
import logging
import os
from itertools import count
from typing import List

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO

from utils.common.numpy_collection import NumpyEncoder


def generate_observations(
    env_fn,
    num_observations: int = 1000,
    render_mode: str | None = None,
    observations_path: str = os.path.join(
        "assets", "observations", "observations.json"
    ),
    **env_kwargs,
) -> List[np.ndarray]:
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = ss.pad_observations_v0(env)

    logging.info(
        f"\nStarting collecting observations on {str(env.metadata['name'])} (num_observations={num_observations}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        logging.warning("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    observations = []

    for i in count():
        env.reset(seed=i)
        if len(observations) >= num_observations:
            observations = observations[:num_observations]
            break

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if np.random.rand() < 0.5:
                observations.append(obs)

            if termination or truncation:
                break
            else:
                if obs is None:
                    act = env.action_space(agent).sample()
                else:
                    obs = obs.reshape((1, -1))
                    act, _ = model.predict(obs, deterministic=True)
                    act = (
                        act
                        if env.action_space(agent).contains(act)
                        else env.action_space(agent).sample()
                    )

            env.step(act)
    env.close()

    with open(observations_path, "w") as file:
        json.dump(observations, file, cls=NumpyEncoder)

    return observations
