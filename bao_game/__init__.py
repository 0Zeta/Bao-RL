from gym.envs.registration import register

register(
    id='Bao-v0',
    entry_point='bao_game.envs:BaoEnv',
)
