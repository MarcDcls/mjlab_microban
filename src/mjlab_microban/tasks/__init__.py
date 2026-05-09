from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.tasks.microban_velocity_env_cfg import (
    make_microban_velocity_env_cfg,
    MicrobanVelocityRlCfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Microban-K1",
    env_cfg=make_microban_velocity_env_cfg(),
    play_env_cfg=make_microban_velocity_env_cfg(play=True),
    rl_cfg=MicrobanVelocityRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)
