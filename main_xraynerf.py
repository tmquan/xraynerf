import os
import warnings

warnings.filterwarnings("ignore")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))

import hydra

from typing import Optional
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from typing import Any, Callable, Dict, Optional, Tuple, List
from lightning.pytorch import seed_everything, Trainer, LightningModule

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)
from pytorch3d.renderer.camera_utils import join_cameras_as_batch

# from monai.networks.nets import Unet
# from monai.networks.layers.factories import Norm
# from generative.networks.nets import DiffusionModelUNet
from omegaconf import OmegaConf

from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.implicit_function.base import ImplicitFunctionBase
from pytorch3d.implicitron.models.renderer.raymarcher import AccumulativeRaymarcherBase
from pytorch3d.implicitron.models.renderer.base import (
    RendererOutput,
    EvaluationMode,
)
from pytorch3d.implicitron.tools.config import (
    registry,
    remove_unused_components,
)
from pytorch3d.renderer.implicit.raymarching import (
    _check_raymarcher_inputs,
    _shifted_cumprod,
)


from datamodule import UnpairedDataModule
from dvr.renderer import ReverseXRayVolumeRenderer


def make_cameras_dea(
    dist: torch.Tensor,
    elev: torch.Tensor,
    azim: torch.Tensor,
    fov: int = 40,
    znear: int = 4.0,
    zfar: int = 8.0,
    is_orthogonal: bool = False,
):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist, elev=elev * 90, azim=azim * 180)
    if is_orthogonal:
        return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)


@registry.register
class AbsorptionEmissionRaymarcher(  # pyre-ignore: 13
    AccumulativeRaymarcherBase, torch.nn.Module
):
    background_opacity: float = 1e10

    @property
    def capping_function_type(self) -> str:
        return "exponential"

    @property
    def weight_function_type(self) -> str:
        return "product"

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        aux: Dict[str, Any],
        ray_lengths: torch.Tensor,
        ray_deltas: Optional[torch.Tensor] = None,
        density_noise_std: float = 0.0,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
            ray_lengths: Per-ray depth values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            ray_deltas: Optional differences between consecutive elements along the ray bundle
                represented with a tensor of shape `(..., n_points_per_ray)`. If None,
                these differences are computed from ray_lengths.
            density_noise_std: the magnitude of the noise added to densities.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            depth: A tensor of shape `(..., 1)` containing estimated depth.
            opacities: A tensor of shape `(..., 1)` containing rendered opacities.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific non-negative opacity weights. In general, they
                don't sum to 1 but do not overcome it, i.e.
                `(weights.sum(dim=-1) <= 1.0).all()` holds.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            ray_lengths,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )

        if ray_deltas is None:
            ray_lengths_diffs = torch.diff(ray_lengths, dim=-1)
            if self.replicate_last_interval:
                last_interval = ray_lengths_diffs[..., -1:]
            else:
                last_interval = torch.full_like(
                    ray_lengths[..., :1], self.background_opacity
                )
            deltas = torch.cat((ray_lengths_diffs, last_interval), dim=-1)
        else:
            deltas = ray_deltas

        rays_densities = rays_densities[..., 0]

        if density_noise_std > 0.0:
            noise: _TTensor = torch.randn_like(rays_densities).mul(density_noise_std)
            rays_densities = rays_densities + noise
        if self.density_relu:
            rays_densities = torch.relu(rays_densities)

        weighted_densities = deltas * rays_densities
        capped_densities = self._capping_function(weighted_densities)

        rays_opacities = self._capping_function(
            torch.cumsum(weighted_densities, dim=-1)
        )
        opacities = rays_opacities[..., -1:]
        # absorption_shifted = (-rays_opacities + 1.0).roll(
        #     self.surface_thickness, dims=-1
        # )
        # absorption_shifted[..., : self.surface_thickness] = 1.0
        eps = 1e-6
        absorption_shifted = _shifted_cumprod(
            (1.0 + eps) - rays_opacities.flip(dims=(-1,)), shift=-self.surface_thickness
        ).flip(dims=(-1,))
        weights = self._weight_function(capped_densities, absorption_shifted)
        features = (weights[..., None] * rays_features).sum(dim=-2)
        depth = (weights * ray_lengths)[..., None].sum(dim=-2)

        alpha = opacities if self.blend_output else 1
        if self._bg_color.shape[-1] not in [1, features.shape[-1]]:
            raise ValueError("Wrong number of background color channels.")
        features = alpha * features + (1 - opacities) * self._bg_color

        return RendererOutput(
            features=features,
            depths=depth,
            masks=opacities,
            weights=weights,
            aux=aux,
        )


class InverseXrayVolumeRenderer(ImplicitFunctionBase, nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        img_shape=256,
        vol_shape=256,
        n_pts_per_ray=256,
        resample=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.n_pts_per_ray = n_pts_per_ray
        self.resample = resample
        self.gm = GenericModel(
            num_passes=1,
            loss_weights={
                # "loss_rgb_l1": 1.0,
                "loss_rgb_mse": 1.0,
                "loss_mask_bce": 0.0,
                "loss_rgb_psnr": 0.0,
                "loss_rgb_huber": 0.0,
                "loss_depth_abs": 0.0,
                # "loss_prev_stage_rgb_l1": 1.0,
                "loss_prev_stage_rgb_mse": 1.0,
                "loss_prev_stage_mask_bce": 0.0,
                "loss_prev_stage_rgb_psnr": 0.0,
                "loss_prev_stage_rgb_huber": 0.0,
                "loss_prev_stage_depth_abs": 0.0,
            },
            image_feature_extractor_class_type="ResNetFeatureExtractor",
            image_feature_extractor_ResNetFeatureExtractor_args={
                "add_images": True,
                "add_masks": False,
                "first_max_pool": True,
                "image_rescale": 1,
                "l2_norm": True,
                "name": "resnet101",
                "normalize_image": True,
                "pretrained": True,
                "stages": (1, 2, 3, 4),
                "proj_dim": 64,
            },
            raysampler_class_type="AdaptiveRaySampler",
            raysampler_AdaptiveRaySampler_args={
                "scene_extent": 1.0,
                "n_pts_per_ray_training": 256,
                "n_pts_per_ray_evaluation": 256,
            },
            chunk_size_grid=16384,
            tqdm_trigger_threshold=100000,
            render_features_dimensions=3,
            n_train_target_views=1,
            sampling_mode_training="mask_sample",
            sampling_mode_evaluation="full_grid",
            render_image_height=256,
            render_image_width=256,
            renderer_class_type="MultiPassEmissionAbsorptionRenderer",
            renderer_MultiPassEmissionAbsorptionRenderer_args={
                "raymarcher_class_type": "AbsorptionEmissionRaymarcher",  # Front To Back
                "raymarcher_AbsorptionEmissionRaymarcher_args": {},
            },
            implicit_function_class_type="NeuralRadianceFieldImplicitFunction",
            implicit_function_NeuralRadianceFieldImplicitFunction_args={
                "n_harmonic_functions_xyz": 10,
                "n_harmonic_functions_dir": 4,
                "n_hidden_neurons_dir": 128,
                "input_xyz": True,
                "xyz_ray_dir_in_camera_coords": False,
                "use_integrated_positional_encoding": False,
                "transformer_dim_down_factor": 1.0,
                "n_hidden_neurons_xyz": 256,
                "n_layers_xyz": 8,
            },
            view_pooler_enabled=True,
        )
        # In this case we can get the equivalent DictConfig cfg object to the way gm is configured as follows
        cfg = OmegaConf.structured(self.gm)
        # We can display the configuration in use as follows.
        remove_unused_components(cfg)
        yaml = OmegaConf.to_yaml(cfg, sort_keys=False)
        # Specify the file path
        file_path = "genericmodel.yaml"
        # Write the YAML content to the file
        with open(file_path, "w") as file:
            file.write(yaml)

    # def forward(self, frames, evaluation_mode=EvaluationMode.EVALUATION):
    #     return self.gm(**frames, evaluation_mode=evaluation_mode)

    def forward(
        self,
        image2d=None,
        cameras=None,
        sequence_name=None,
        evaluation_mode=EvaluationMode.EVALUATION,
    ):
        if image2d.shape[1] == 1:
            image2d = image2d.repeat(1, 3, 1, 1)
        return self.gm(
            image_rgb=image2d,
            camera=cameras,
            sequence_name=sequence_name,
            evaluation_mode=evaluation_mode,
        )


class NVLightningModule(LightningModule):
    def __init__(
        self, model_cfg: DictConfig, train_cfg: DictConfig, infer_cfg: DictConfig
    ):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.infer_cfg = infer_cfg

        self.fwd_renderer = ReverseXRayVolumeRenderer(
            image_width=self.model_cfg.img_shape,
            image_height=self.model_cfg.img_shape,
            n_pts_per_ray=self.model_cfg.n_pts_per_ray,
            min_depth=4.0,
            max_depth=8.0,
            ndc_extent=1.0,
        )

        self.inv_renderer = InverseXrayVolumeRenderer(
            in_channels=1,
            out_channels=1,
            img_shape=self.model_cfg.img_shape,
            vol_shape=self.model_cfg.vol_shape,
            n_pts_per_ray=self.model_cfg.n_pts_per_ray,
        )

        if self.train_cfg.ckpt:
            print("Loading.. ", self.train_cfg.ckpt)
            checkpoint = torch.load(
                self.train_cfg.ckpt, map_location=torch.device("cpu")
            )["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=False)

        self.save_hyperparameters()
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def correct_window(self, T_old, a_min=-1024, a_max=3071, b_min=-512, b_max=3071):
        # Calculate the range for the old and new scales
        range_old = a_max - a_min
        range_new = b_max - b_min

        # Reverse the incorrect scaling
        T_raw = (T_old * range_old) + a_min

        # Apply the correct scaling
        T_new = (T_raw - b_min) / range_new
        return T_new.clamp_(0, 1)

    def forward_screen(self, image3d, cameras, is_training=False):
        image3d = self.correct_window(
            image3d, a_min=-1024, a_max=3071, b_min=-512, b_max=3071
        )
        return self.fwd_renderer(
            image3d, cameras, norm_type="standardized", stratified_sampling=is_training
        )

    def forward_volume(
        self,
        image2d,
        cameras,
        n_views=[2, 1],
        resample=True,
        timesteps=None,
        is_training=False,
    ):
        pass

    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

        image3d_pth = str(batch["image3d_pth"][0]).split("/")[-1]
        image2d_pth = str(batch["image2d_pth"][0]).split("/")[-1]
        image3d_idx = str(batch["image3d_idx"][0])
        image2d_idx = str(batch["image2d_idx"][0])

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(B, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # from [0 1) to [-1 1)
        azim_random = 0.75 * azim_random - 0.25  #  from [-1, 1) to [-1, 0.5).
        view_random = make_cameras_dea(
            dist_random, elev_random, azim_random, fov=16, znear=4, zfar=8
        )

        dist_hidden = 6.0 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(
            dist_hidden, elev_hidden, azim_hidden, fov=16, znear=4, zfar=8
        )

        # Construct the samples in 2D
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)

        if self.model_cfg.phase == "ctproj":
            pass
        else:
            # TODO
            figure_xr_hidden_hidden = torch.cat(
                [figure_xr_hidden, figure_xr_hidden], dim=0
            )
            figure_ct_hidden_hidden = torch.cat(
                [figure_ct_hidden, figure_ct_hidden], dim=0
            )
            figure_ct_random_random = torch.cat(
                [figure_ct_random, figure_ct_random], dim=0
            )
            figure_ct_hidden_random = torch.cat(
                [figure_ct_hidden, figure_ct_random], dim=0
            )
            figure_ct_random_hidden = torch.cat(
                [figure_ct_random, figure_ct_hidden], dim=0
            )

            camera_hidden_hidden = join_cameras_as_batch([view_hidden, view_hidden])
            camera_random_random = join_cameras_as_batch([view_random, view_random])
            camera_hidden_random = join_cameras_as_batch([view_hidden, view_random])
            camera_random_hidden = join_cameras_as_batch([view_random, view_hidden])

            evaluation_mode = (
                EvaluationMode.EVALUATION
                if stage == "validation"
                else EvaluationMode.TRAINING
            )

            output_xr_hidden_hidden = self.inv_renderer.forward(
                image2d=figure_xr_hidden_hidden,
                cameras=camera_hidden_hidden,
                sequence_name=[image2d_idx + "_" + image2d_pth] * B,
                evaluation_mode=evaluation_mode,
            )
            output_ct_hidden_hidden = self.inv_renderer.forward(
                image2d=figure_ct_hidden_hidden,
                cameras=camera_hidden_hidden,
                sequence_name=[image3d_idx + "_" + image3d_pth] * B,
                evaluation_mode=evaluation_mode,
            )
            output_ct_random_random = self.inv_renderer.forward(
                image2d=figure_ct_random_random,
                cameras=camera_random_random,
                sequence_name=[image3d_idx + "_" + image3d_pth] * B,
                evaluation_mode=evaluation_mode,
            )
            output_ct_hidden_random = self.inv_renderer.forward(
                image2d=figure_ct_hidden_random,
                cameras=camera_hidden_random,
                sequence_name=[image3d_idx + "_" + image3d_pth] * B,
                evaluation_mode=evaluation_mode,
            )
            output_ct_random_hidden = self.inv_renderer.forward(
                image2d=figure_ct_random_hidden,
                cameras=camera_random_hidden,
                sequence_name=[image3d_idx + "_" + image3d_pth] * B,
                evaluation_mode=evaluation_mode,
            )

            im2d_loss = (
                output_xr_hidden_hidden["objective"]
                + output_ct_hidden_hidden["objective"]
                + output_ct_random_random["objective"]
                + output_ct_random_hidden["objective"]
                + output_ct_hidden_random["objective"]
            )

            self.log(
                f"{stage}_im2d_loss",
                im2d_loss,
                on_step=(stage == "train"),
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=B,
            )
            
            # Visualization step
            if batch_idx == 0 and stage == "validation":
                viz2d = torch.cat(
                    [
                        torch.cat(
                            [
                                figure_xr_hidden.repeat(1, 3, 1, 1),
                                figure_ct_hidden.repeat(1, 3, 1, 1),
                                figure_ct_random.repeat(1, 3, 1, 1),
                                output_ct_hidden_random["images_render"],
                            ],
                            dim=-2,
                        ).transpose(2, 3),
                        torch.cat(
                            [
                                output_xr_hidden_hidden["images_render"],
                                output_ct_hidden_hidden["images_render"],
                                output_ct_random_random["images_render"],
                                output_ct_random_hidden["images_render"],
                            ],
                            dim=-2,
                        ).transpose(2, 3),
                    ],
                    dim=-2,
                )
                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(
                    viz2d, normalize=False, scale_each=True, nrow=1, padding=0
                )
                tensorboard.add_image(
                    f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx
                )
            loss = im2d_loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(
            f"train_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            f"validation_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()  # free memory

    def sample(self, **kwargs: dict):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.train_cfg.lr, betas=(0.5, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 200], gamma=0.1
        )
        return [optimizer], [scheduler]


@hydra.main(version_base=None, config_path="./conf")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # resolve all str interpolation
    seed_everything(42)
    datamodule = UnpairedDataModule(
        train_image3d_folders=cfg.data.train_image3d_folders,
        train_image2d_folders=cfg.data.train_image2d_folders,
        val_image3d_folders=cfg.data.val_image3d_folders,
        val_image2d_folders=cfg.data.val_image2d_folders,
        test_image3d_folders=cfg.data.test_image3d_folders,
        test_image2d_folders=cfg.data.test_image2d_folders,
        img_shape=cfg.data.img_shape,
        vol_shape=cfg.data.vol_shape,
        batch_size=cfg.data.batch_size,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples,
    )

    model = NVLightningModule(
        model_cfg=cfg.model, train_cfg=cfg.train, infer_cfg=cfg.infer
    )
    callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks]
    logger = [hydra.utils.instantiate(c) for c in cfg.logger]

    trainer = Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    trainer.fit(
        model,
        # datamodule=datamodule,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        # ckpt_path=cfg.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
