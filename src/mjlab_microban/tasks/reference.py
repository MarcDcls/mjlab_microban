import pickle
from dataclasses import dataclass
from mjlab.managers.reward_manager import RewardTermCfg
import math
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.tasks.velocity.mdp import Entity


@dataclass(kw_only=True)
class ReferenceCommandCfg(CommandTermCfg):
    """Configuration for the Reference motion-tracking command."""

    poses_path: str = "poses.pkl"
    """Path to the pickle file containing reference poses, shape [S, J]."""

    twist_command_name: str = "twist"
    """Name of the velocity command used to decide whether the robot is commanded to move."""

    velocity_threshold: float = 0.05
    """Minimum twist magnitude to consider the robot commanded to move."""

    def build(self, env: ManagerBasedRlEnv) -> "ReferenceCommand":
        return ReferenceCommand(self, env)


class ReferenceCommand(CommandTerm):
    """Tracks a cyclic reference motion tied to the current velocity command.

    At each step:
    - If the magnitude of the twist command exceeds *velocity_threshold*, the per-environment
      integer counter ``step`` is incremented.
    - Otherwise the counter is reset to 0.

    The ``command`` property returns the reference joint positions indexed by
    ``step % n_steps`` for each environment.
    """

    cfg: ReferenceCommandCfg

    def __init__(self, cfg: ReferenceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        # Load reference poses from disk
        with open(cfg.poses_path, "rb") as f:
            poses_np = pickle.load(f)
        self.poses: torch.Tensor = torch.tensor(
            poses_np, dtype=torch.float32, device=self.device
        )  # [S, J]
        self.n_steps: int = self.poses.shape[0]

        # Integer step counter, one per environment
        self.step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Buffer for the current command (target joint positions)
        self._command = self.poses[0].unsqueeze(0).expand(self.num_envs, -1).clone()

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Current target joint positions for each environment, shape [num_envs, J]."""
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Called on environment reset — zero out the step counter."""
        self.step[env_ids] = 0
        self._sync_command(env_ids)

    def _update_command(self) -> None:
        """Called every step — advance or reset each environment's phase counter."""
        cmd = self._env.command_manager.get_term(self.cfg.twist_command_name)

        if hasattr(cmd, "moving_magnitude"):
            magnitude = cmd.moving_magnitude  # [N]
        else:
            twist_cmd = self._env.command_manager.get_command(
                self.cfg.twist_command_name
            )  # [N, D]

            magnitude = torch.linalg.norm(twist_cmd, dim=1)  # [N]

        moving = magnitude > self.cfg.velocity_threshold
        self.step[moving] += 1
        self.step[~moving] = 0

        all_ids = torch.arange(self.num_envs, device=self.device)
        self._sync_command(all_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_command(self, env_ids: torch.Tensor) -> None:
        """Update the command buffer for the given environments."""
        indices = self.step[env_ids] % self.n_steps
        self._command[env_ids] = self.poses[indices]


# ---------------------------------------------------------------------------
# Reference-phase observation
# ---------------------------------------------------------------------------


def reference_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "reference",
) -> torch.Tensor:
    """Return [cos, sin] of the normalised reference phase for each environment.

    Output shape: [num_envs, 2]
    """
    term: ReferenceCommand = env.command_manager.get_term(command_name)
    phase = 2.0 * math.pi * term.step.float() / term.n_steps  # [N]
    return torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)  # [N, 2]


# ---------------------------------------------------------------------------
# Reference-pose reward
# ---------------------------------------------------------------------------


class reference_pose_reward:
    """Reward the robot for matching the reference joint pose.

    Per-joint stds are resolved from ``std_standing`` and ``std_walking``
    dicts (same format as the velocity pose reward).  At runtime the two
    std vectors are blended linearly based on the magnitude of the twist
    command relative to ``velocity_threshold``.

    reward = exp(-mean(error_sq / std_blended**2))
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names, preserve_order=True)

        _, _, std_standing_vals = resolve_matching_names_values(
            data=cfg.params["std_standing"], list_of_strings=joint_names
        )
        _, _, std_walking_vals = resolve_matching_names_values(
            data=cfg.params["std_walking"], list_of_strings=joint_names
        )
        self.std_standing = torch.tensor(
            std_standing_vals, device=env.device, dtype=torch.float32
        )  # [J]
        self.std_walking = torch.tensor(
            std_walking_vals, device=env.device, dtype=torch.float32
        )  # [J]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del std_standing, std_walking  # resolved at init

        term: ReferenceCommand = env.command_manager.get_term(command_name)
        target_pose = term.command  # [N, J]

        asset: Entity = env.scene[asset_cfg.name]
        actual_pose = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, J]

        # Blend stds: fully standing when below threshold, fully walking above it
        cmd = env.command_manager.get_term(term.cfg.twist_command_name)

        if hasattr(cmd, "moving_magnitude"):
            magnitude = cmd.moving_magnitude  # [N]
        else:
            twist_cmd = env.command_manager.get_command(
                term.cfg.twist_command_name
            )  # [N, D]

            magnitude = torch.linalg.norm(twist_cmd, dim=1)  # [N]

        blend = torch.clamp(magnitude / term.cfg.velocity_threshold, 0.0, 1.0)  # [N]
        std = self.std_standing + blend.unsqueeze(1) * (
            self.std_walking - self.std_standing
        )  # [N, J]

        error_squared = torch.square(actual_pose - target_pose)  # [N, J]
        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
