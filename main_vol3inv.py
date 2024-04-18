import os
import warnings

warnings.filterwarnings("ignore")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))
from itertools import chain

import hydra
from hydra.utils import instantiate

from typing import Optional
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torchmetrics.functional.image import image_gradients
from typing import Any, Callable, Dict, Optional, Tuple, List
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
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
from PIL import Image
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.utils import DATASET_TYPE_KNOWN, DATASET_TYPE_UNKNOWN
from pytorch3d.implicitron.dataset.rendered_mesh_dataset_map_provider import (
    RenderedMeshDatasetMapProvider,
)

from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.implicit_function.base import (
    ImplicitFunctionBase,
    ImplicitronRayBundle,
)
from pytorch3d.implicitron.models.renderer.raymarcher import (
    AccumulativeRaymarcherBase,
    RaymarcherBase,
)
from pytorch3d.implicitron.models.renderer.base import (
    BaseRenderer,
    RendererOutput,
    EvaluationMode,
    ImplicitFunctionWrapper,
)
from pytorch3d.implicitron.models.renderer.multipass_ea import (
    MultiPassEmissionAbsorptionRenderer,
)
from pytorch3d.implicitron.models.renderer.ray_point_refiner import RayPointRefiner
from pytorch3d.implicitron.tools.config import (
    get_default_args,
    registry,
    remove_unused_components,
    run_auto_creation,
)
from pytorch3d.renderer.implicit.renderer import VolumeSampler
from pytorch3d.vis.plotly_vis import plot_batch_individually, plot_scene
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)

from monai.losses import PerceptualLoss
from monai.networks.nets import Unet
from monai.networks.layers.factories import Norm
from generative.networks.nets import DiffusionModelUNet

from datamodule import UnpairedDataModule
from dvr.renderer import ReverseXRayVolumeRenderer
from dvr.renderer import normalized
from dvr.renderer import standardized


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


class InverseXrayVolumeRenderer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        img_shape=256,
        vol_shape=256,
        fov_depth=256,
        resample=True,
        gradient=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.fov_depth = fov_depth
        self.resample = resample
        self.gradient = gradient
        self.net2d3d_explicit = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2 if self.gradient else 1,  # Condition with straight/hidden view
            out_channels=self.fov_depth,
            num_channels=(128, 128, 256),
            attention_levels=(False, False, True),
            num_res_blocks=1,
            num_head_channels=256,
            with_conditioning=True,
            cross_attention_dim=12,  # flatR | flatT
        )

        self.net2d3d_implicit = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2 if self.gradient else 1,  # Condition with straight/hidden view
            out_channels=self.fov_depth,
            num_channels=(128, 128, 256),
            attention_levels=(False, False, True),
            num_res_blocks=1,
            num_head_channels=256,
            with_conditioning=True,
            cross_attention_dim=12,  # flatR | flatT
        )

        self.net3d3d = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2,
                out_channels=6,
                # channels=backbones[backbone],
                # strides=(2, 2, 2, 2, 2),
                # num_res_units=1,
                channels=(128, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.INSTANCE,
                dropout=0.5,
            ),
        )

    def forward(
        self,
        image2d,
        cameras,
        resample=True,
        timesteps=None,
        is_training=False,
        with_gradients=True,
    ):
        _device = image2d.device
        B = image2d.shape[0]
        dtype = image2d.dtype
        if timesteps is None:
            timesteps = torch.zeros((B), device=_device).long()

        assert self.gradient == with_gradients
        if with_gradients:
            image2d = torch.cat(image_gradients(image2d), dim=1)

        detcams = cameras.clone()
        R = detcams.R
        # T = detcams.T.unsqueeze_(-1)
        T = torch.zeros_like(detcams.T.unsqueeze_(-1))
        inv = torch.cat([torch.inverse(R), -T], dim=-1)

        mat = (
            torch.cat(
                [detcams.R.reshape(-1, 1, 9), detcams.T.reshape(-1, 1, 3)], dim=-1
            )
            .contiguous()
            .view(-1, 1, 12)
        )

        image2d = torch.rot90(image2d, 1, [2, 3])

        i03 = self.net2d3d_implicit(
            x=image2d, context=mat.reshape(B, 1, -1), timesteps=timesteps,
        ).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)
        
        e23 = self.net2d3d_explicit(
            x=image2d, context=mat.reshape(B, 1, -1), timesteps=timesteps,
        ).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)

        if resample:
            grd = F.affine_grid(inv, i03.size()).type(dtype)
            m23 = F.grid_sample(e23, grd)
            # sum = m23 + i03
            sum = torch.cat([m23, i03], dim=1)
            vol = self.net3d3d(sum)
            out = (
                torch.concat(
                    [
                        torch.permute(vol[:, [0], ...], (0, 1, 2, 3, 4)),
                        torch.permute(vol[:, [1], ...], (0, 1, 2, 4, 3)),
                        torch.permute(vol[:, [2], ...], (0, 1, 3, 2, 4)),
                        torch.permute(vol[:, [3], ...], (0, 1, 3, 4, 2)),
                        torch.permute(vol[:, [4], ...], (0, 1, 4, 2, 3)),
                        torch.permute(vol[:, [5], ...], (0, 1, 4, 3, 2)),
                    ],
                    dim=1,
                ).mean(dim=1, keepdim=True)
                # + sum
            )
        else:
            pass
        return out


class NVLightningModule(LightningModule):
    def __init__(
        self, model_cfg: DictConfig, train_cfg: DictConfig,
    ):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

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
            fov_depth=self.model_cfg.fov_depth,
            gradient=True,
        )

        self.p2dloss = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            is_fake_3d=False,
            pretrained=True,
        )

        self.p3dloss = PerceptualLoss(
            spatial_dims=3,
            network_type="medicalnet_resnet50_23datasets",
            is_fake_3d=False,
            pretrained=True,
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
        with_gradients=False,
    ):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections

        results = self.inv_renderer(
            image2d, cameras, resample, timesteps, is_training, with_gradients
        )
        return results

    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 8 * torch.ones(B, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # from [0 1) to [-1 1)
        view_random = make_cameras_dea(
            dist_random, elev_random, azim_random, fov=16, znear=6, zfar=10
        )

        dist_second = 8 * torch.ones(B, device=_device)
        elev_second = torch.rand_like(dist_second) - 0.5
        azim_second = torch.rand_like(dist_second) * 2 - 1  # from [0 1) to [-1 1)
        view_second = make_cameras_dea(
            dist_second, elev_second, azim_second, fov=16, znear=6, zfar=10
        )

        dist_hidden = 8 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(
            dist_hidden, elev_hidden, azim_hidden, fov=16, znear=6, zfar=10
        )

        # Construct the samples in 2D
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_second = self.forward_screen(image3d=image3d, cameras=view_second)

        if self.model_cfg.phase == "multi":
            pass
        else:
            # Reconstruct the Encoder-Decoder
            volume_dx_concat = self.forward_volume(
                image2d=torch.cat(
                    [figure_xr_hidden, figure_ct_random, figure_ct_second]
                ),
                cameras=join_cameras_as_batch([view_hidden, view_random, view_second]),
                n_views=[1, 1, 1] * B,
                resample=True,
                timesteps=None,
                is_training=(stage == "train"),
                with_gradients=True,
            )
            (
                volume_xr_hidden_origin,
                volume_ct_random_origin,
                volume_ct_second_origin,
            ) = torch.split(volume_dx_concat, B)

            # Reconstruct the projection
            figure_xr_origin_hidden_random = self.forward_screen(
                image3d=volume_xr_hidden_origin,
                cameras=view_random,
                is_training=(stage == "train"),
            )
            figure_xr_origin_hidden_hidden = self.forward_screen(
                image3d=volume_xr_hidden_origin,
                cameras=view_hidden,
                is_training=(stage == "train"),
            )
            figure_ct_origin_random_random = self.forward_screen(
                image3d=volume_ct_random_origin,
                cameras=view_random,
                is_training=(stage == "train"),
            )
            figure_ct_origin_random_second = self.forward_screen(
                image3d=volume_ct_random_origin,
                cameras=view_second,
                is_training=(stage == "train"),
            )
            figure_ct_origin_second_random = self.forward_screen(
                image3d=volume_ct_second_origin,
                cameras=view_random,
                is_training=(stage == "train"),
            )
            figure_ct_origin_second_second = self.forward_screen(
                image3d=volume_ct_second_origin,
                cameras=view_second,
                is_training=(stage == "train"),
            )

            # Compute the losses
            im3d_loss = F.l1_loss(volume_ct_second_origin, image3d) \
                      + F.l1_loss(volume_ct_random_origin, image3d)

            im2d_loss = F.l1_loss(figure_ct_origin_random_random, figure_ct_random) \
                      + F.l1_loss(figure_ct_origin_random_second, figure_ct_second) \
                      + F.l1_loss(figure_ct_origin_second_random, figure_ct_random) \
                      + F.l1_loss(figure_ct_origin_second_second, figure_ct_second)

            pc3d_loss = self.p3dloss(volume_xr_hidden_origin, image3d)
            im3d_loss += self.train_cfg.lamda * pc3d_loss

            pc2d_loss = self.p2dloss(figure_xr_origin_hidden_random, figure_ct_random) \
                      + self.p2dloss(figure_xr_origin_hidden_hidden, image2d)
            im2d_loss += self.train_cfg.lamda * pc2d_loss

            self.log(
                f"{stage}_im3d_loss",
                im3d_loss,
                on_step=(stage == "train"),
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=B,
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

            loss = self.train_cfg.alpha * im3d_loss + self.train_cfg.gamma * im2d_loss

            # Visualization step
            if batch_idx == 0:
                zeros2d = torch.zeros_like(image2d)
                viz2d = torch.cat(
                    [
                        torch.cat(
                            [
                                zeros2d,
                                image2d,
                                volume_xr_hidden_origin[..., self.model_cfg.vol_shape // 2, :],
                                figure_xr_origin_hidden_random,
                                figure_xr_origin_hidden_hidden,
                            ],
                            dim=-2,
                        ).transpose(2, 3),
                        torch.cat(
                            [
                                image3d[..., self.model_cfg.vol_shape // 2, :],
                                figure_ct_random,
                                volume_ct_random_origin[..., self.model_cfg.vol_shape // 2, :],
                                figure_ct_origin_random_random,
                                figure_ct_origin_random_second,
                            ],
                            dim=-2,
                        ).transpose(2, 3),
                        torch.cat(
                            [
                                image3d[..., self.model_cfg.vol_shape // 2, :],
                                figure_ct_second,
                                volume_ct_second_origin[..., self.model_cfg.vol_shape // 2, :],
                                figure_ct_origin_second_random,
                                figure_ct_origin_second_second,
                            ],
                            dim=-2,
                        ).transpose(2, 3),
                    ],
                    dim=-2,
                )

                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(
                    viz2d, normalize=False, scale_each=False, nrow=1, padding=0
                ).clamp(0, 1)
                tensorboard.add_image(
                    f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx
                )

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

    model = NVLightningModule(model_cfg=cfg.model, train_cfg=cfg.train,)
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
