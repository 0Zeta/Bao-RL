from gym.envs.registration import register

register(
    id='HusBao-v0',
    entry_point='hus_bao.envs:HusBaoEnv',
)
