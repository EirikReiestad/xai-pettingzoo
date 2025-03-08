from __future__ import annotations

import glob
import json
import logging
import os
from collections import defaultdict
from itertools import count
from typing import List, Literal

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO

from utils.common.concepts import has_concept
from utils.common.numpy_collection import NumpyEncoder


def collect_concepts(
    env_fn,
    concepts: List[str],
    num_concepts: int = 1000,
    render_mode: Literal["rgb_array", "human"] | None = "human",
    concept_path: str = os.path.join("assets", "concepts"),
    **env_kwargs,
):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = ss.pad_observations_v0(env)

    logging.info(
        f"\nStarting collecting concepts on {str(env.metadata['name'])} (num_concepts={num_concepts}, concepts={concepts}, render_mode={render_mode})"
    )
    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        logging.warning("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    negative_concepts = [f"negative_{c}" for c in concepts]

    all_concepts = concepts + negative_concepts

    observations = defaultdict(list)
    observations_filled = {c: False for c in all_concepts}

    for i in count():
        env.reset(seed=i)
        if all(observations_filled.values()):
            break

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for concept in all_concepts:
                if observations_filled[concept]:
                    continue
                if len(observations[concept]) == num_concepts:
                    observations_filled[concept] = True
                    continue
                if has_concept(concept, obs):
                    observations[concept].append(obs)

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


def generate_observations(
    observations_path: str = os.path.join(
        "assets", "observations", "observations.json"
    ),
) -> List[np.ndarray]:
    with open(observations_path, "w") as file:
        json.dump(observations, file, cls=NumpyEncoder)

    return observations
