def get_env_from_name(name, alpha=0):

    if name == 'acc':
        from .acc import AccEnv
        return AccEnv(alpha=alpha)
    if name == 'car_racing':
        from .car_racing import CarRacingEnv
        return CarRacingEnv(alpha=alpha)
    if name == 'mid_obstacle':
        from .mid_obstacle import MidObstacleEnv
        return MidObstacleEnv(alpha=alpha)
    if name == 'mountain_car':
        from .mountain_car import MountainCarEnv
        return MountainCarEnv(alpha=alpha)
    if name == 'noisy_road':
        from .noisy_road import NoisyRoadEnv
        return NoisyRoadEnv(alpha=alpha)
    if name == 'noisy_road_2d':
        from .noisy_road_2d import NoisyRoad2dEnv
        return NoisyRoad2dEnv(alpha=alpha)
    if name == 'obstacle':
        from .obstacle import ObstacleEnv
        return ObstacleEnv(alpha=alpha)
    if name == 'pendulum':
        from .pendulum import PendulumEnv
        return PendulumEnv(alpha=alpha)
    if name == 'road':
        from .road import RoadEnv
        return RoadEnv(alpha=alpha)
    if name == 'road_2d':
        from .road_2d import Road2dEnv
        return Road2dEnv(alpha=alpha)
    if name == 'lunar_lander':
        from .lunar_lander import LunarLanderEnv
        return LunarLanderEnv(alpha=alpha)
    if name == 'lunar_lander_R':
        from .lunar_lander_RedDim import LunarLanderEnv2
        return LunarLanderEnv2(alpha=alpha)
    if name == 'bipedal_walker':
        from .bipedal_walker import BipedalWalkerEnv
        return BipedalWalkerEnv(alpha=alpha)
    if name == 'inverted_pendulum':
        from .inverted_pendulum import InvertedPendulumEnv
        return InvertedPendulumEnv(alpha=alpha)
    if name == 'hopper':
        from .hopper import HopperEnv
        return HopperEnv(alpha=alpha)
    if name == 'walker':
        from .walker import WalkerEnv
        return WalkerEnv(alpha=alpha)
    if name == 'cheetah':
        from .cheetah import CheetahEnv
        return CheetahEnv(alpha=alpha)
    if name == 'safety_point':
        from .safety_gym import SafetyPointGoalEnv
        return SafetyPointGoalEnv(alpha=alpha)
    if name == 'carplatoon4':
        from .CarPlatoon4 import CarPlatoonEnv
        return CarPlatoonEnv(alpha=alpha)
    if name == 'Oscillator':
        from .Oscillator import OscillatorEnv
        return OscillatorEnv(alpha=alpha)
    if name == 'humanoid':
        from .humanoid import HumanoidEnv
        return HumanoidEnv(alpha=alpha)
    if name == 'ant':
        from .ant import AntEnv
        return AntEnv(alpha=alpha)
    else:
        raise RuntimeError("Unkonwn environment: " + name)
