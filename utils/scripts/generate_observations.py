from __future__ import annotations

import logging

from pettingzoo.mpe import simple_push_v3, simple_spread_v3, simple_tag_v3, simple_v3

from utils.core.generate_observations import generate_observations

logging.basicConfig(level=logging.INFO)


def main():
    env_fn = simple_push_v3
    env_kwargs = {}

    generate_observations(
        env_fn, num_observations=100, render_mode="human", **env_kwargs
    )


if __name__ == "__main__":
    main()
