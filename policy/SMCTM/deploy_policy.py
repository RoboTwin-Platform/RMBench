"""RMBench deployment adapter for frozen-DINOv3 SM-CTM checkpoints."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch


_SMCTM_ROOT = Path(__file__).resolve().parents[3] / "smctm_robot_policy"
_SMCTM_SITE = _SMCTM_ROOT / ".venv/lib/python3.10/site-packages"
for path in (_SMCTM_ROOT, _SMCTM_SITE):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from smctm import DINOv3ImageEncoder, JointMinMaxStats, SMCTMConfig, SMCTMPolicy


def env_to_model_state(vector: np.ndarray) -> torch.Tensor:
    """Env [LA6,LGrip,RA6,RGrip] -> training [LA6,0,RA6,0,LGrip,RGrip]."""

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size != 14:
        raise ValueError(f"Expected 14-D RMBench joint vector, got {vector.size}")
    state = np.zeros(16, dtype=np.float32)
    state[0:6] = vector[0:6]
    state[7:13] = vector[7:13]
    state[14] = vector[6]
    state[15] = vector[13]
    return torch.from_numpy(state)


def model_to_env_action(vector: torch.Tensor) -> np.ndarray:
    """Training [LA6,0,RA6,0,LGrip,RGrip] -> env qpos layout."""

    vector = vector.detach().float().cpu().numpy().reshape(-1)
    if vector.size != 16:
        raise ValueError(f"Expected 16-D model action, got {vector.size}")
    action = np.empty(14, dtype=np.float32)
    action[0:6] = vector[0:6]
    action[6] = np.clip(vector[14], 0.0, 1.0)
    action[7:13] = vector[7:13]
    action[13] = np.clip(vector[15], 0.0, 1.0)
    return action


class SMCTMDeployment:
    def __init__(self, args: dict) -> None:
        self.device = torch.device(args.get("device", "cuda:0"))
        checkpoint_path = Path(args["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = SMCTMConfig.from_mapping(checkpoint["config"])
        self.policy = SMCTMPolicy(self.config).to(self.device)
        self.policy.load_state_dict(checkpoint["model"])
        self.policy.eval()

        metadata = checkpoint.get("feature_metadata")
        if not metadata or "normalization" not in metadata:
            raise KeyError("Checkpoint is missing feature normalization metadata")
        normalization = metadata["normalization"]
        self.stats = JointMinMaxStats(
            minimum=torch.tensor(normalization["minimum"], dtype=torch.float32),
            maximum=torch.tensor(normalization["maximum"], dtype=torch.float32),
        )
        dino_checkpoint = args.get("dino_checkpoint") or metadata.get("checkpoint")
        self.encoder = DINOv3ImageEncoder(
            model_name=metadata.get(
                "model_name", "vit_base_patch16_dinov3.lvd1689m"
            ),
            checkpoint_path=dino_checkpoint,
            image_size=int(metadata.get("image_size", 224)),
        ).to(self.device)
        self.encoder.eval()
        self.execute_steps = max(1, int(args.get("execute_steps", 1)))
        self.seed = int(args.get("policy_seed", 2026))
        self.stream = None
        self.generator = None
        self.reset()

    def reset(self) -> None:
        dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.stream = self.policy.init_stream_state(1, self.device, dtype)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def encode(self, observation: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(
            observation["observation"]["head_camera"]["rgb"], copy=True
        )
        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
            .to(self.device)
        )
        vector = observation["joint_action"]["vector"]
        state = env_to_model_state(vector)
        normalized_state = self.stats.normalize(state).unsqueeze(0).to(self.device)
        visual_tokens = self.encoder(image_tensor)
        return visual_tokens, normalized_state

    @torch.inference_mode()
    def plan(self, observation: dict) -> tuple[torch.Tensor, np.ndarray]:
        visual_tokens, proprio = self.encode(observation)
        language = torch.zeros(
            1,
            1,
            self.config.observation.language_dim,
            device=self.device,
            dtype=visual_tokens.dtype,
        )
        use_bfloat16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            output = self.policy.act_step(
                visual_tokens=visual_tokens,
                language_tokens=language,
                proprio=proprio,
                stream_state=self.stream,
                generator=self.generator,
            )
        normalized = output.action_chunk[0]
        denormalized = self.stats.denormalize(normalized)
        self.stream = output.stream_state
        return normalized, np.stack(
            [model_to_env_action(action) for action in denormalized], axis=0
        )

    def record_executed(self, normalized_action: torch.Tensor) -> None:
        self.stream = self.policy.record_executed_action(
            self.stream, normalized_action.unsqueeze(0)
        )


def get_model(usr_args: dict) -> SMCTMDeployment:
    return SMCTMDeployment(usr_args)


def eval(TASK_ENV, model: SMCTMDeployment, observation: dict):
    normalized, actions = model.plan(observation)
    steps = min(model.execute_steps, actions.shape[0])
    for index in range(steps):
        TASK_ENV.take_action(actions[index], action_type="qpos")
        model.record_executed(normalized[index])
        observation = TASK_ENV.get_obs()
        if TASK_ENV.eval_success:
            break
    return observation


def reset_model(model: Optional[SMCTMDeployment] = None) -> None:
    if model is not None:
        model.reset()
