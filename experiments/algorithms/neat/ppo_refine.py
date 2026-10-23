from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class TanhMeanPolicy(ActorCriticPolicy):
    """Standard Gaussian PPO with a bounded mean, matching the student actor."""
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # The Linear parameters are unchanged, so the existing optimizer keeps
        # the correct references. Tanh itself introduces no parameters.
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())
