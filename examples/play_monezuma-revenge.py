import ale_py
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import gymnasium as gym
from gymnasium.utils.play import play


def main():
    gym.register_envs(ale_py)

    gym_environment = "ALE/MontezumaRevenge-v5"
    env = gym.make(gym_environment, frameskip=1, render_mode='rgb_array')
    # # keys_to_action={
    # #     chr(ord('a') + action): action for action in range(18)
    # # })
    # env = AtariPreprocessing(
    #     env,
    #     noop_max=30,
    #     frame_skip=3, # 3 + 1 = 4 ?
    #     screen_size=84,
    #     terminal_on_life_loss=False,
    #     grayscale_obs=True,
    #     scale_obs=True,
    # )

    # print(env.action_space)

    # env = gym.make(gym_environment, render_mode='rgb_array')
    # play(env, zoom=3)
    play(env, zoom=3)


if __name__ == "__main__":
    main()
