from gym.envs.registration import register

register(
    id='platoon_beam-v0',
    entry_point='gym_platoon_beamformer.envs:PlatoonBeamEnv'
)


