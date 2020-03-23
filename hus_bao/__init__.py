from gym.envs.registration import register

register(
    id='hus_bao-v0',
    entry_point='hus_bao.envs:HusBaoEnv',
)
