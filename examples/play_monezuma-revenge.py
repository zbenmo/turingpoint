import ale_py
import gymnasium as gym
from gymnasium.utils.play import play


def main():
    gym.register_envs(ale_py)

    gym_environment = "ALE/MontezumaRevenge-v5"
    env = gym.make(gym_environment, frameskip=1, render_mode='rgb_array')

    play(env, zoom=3)


if __name__ == "__main__":
    main()
