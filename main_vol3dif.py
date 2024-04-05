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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.fov_depth = fov_depth
        self.resample = resample

        self.net2d3d = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,  # Condition with straight/hidden view
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
                in_channels=1, 
                out_channels=1, 
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
                dropout=0.5
            ),
        )
        
        
    def forward(self, image2d, cameras, resample=True, timesteps=None, is_training=False):
        _device = image2d.device
        batch = image2d.shape[0]
        dtype = image2d.dtype
        if timesteps is None:
            timesteps = torch.zeros((batch), device=_device).long()
        
        detcams = cameras.clone()
        R = detcams.R
        # T = detcams.T.unsqueeze_(-1)
        T = torch.zeros_like(detcams.T.unsqueeze_(-1))
        inv = torch.cat([torch.inverse(R), -T], dim=-1)
        
        mat = torch.cat([detcams.R.reshape(-1, 1, 9), detcams.T.reshape(-1, 1, 3)], dim=-1).contiguous().view(-1, 1, 12)

        # Run forward pass
        fov = self.net2d3d(
            x=image2d, 
            context=mat.reshape(batch, 1, -1),
            timesteps=timesteps,
        ).view(-1, 1, self.fov_depth, self.img_shape, self.img_shape)

        if resample:
            grd = F.affine_grid(inv, fov.size()).type(dtype)
            if is_training:
                # Randomly choose between mid and fov with a 50% probability each using torch.rand()
                if torch.rand(1).item() < 0.5:
                    mid = F.grid_sample(fov, grd)
                    vol = mid + self.net3d3d(mid)
                else:
                    out = fov + self.net3d3d(fov)
                    vol = F.grid_sample(out, grd)
            else: 
                mid = F.grid_sample(fov, grd)
                vol = mid + self.net3d3d(mid)
            # mid = F.grid_sample(fov, grd)
            # vol = mid + self.net3d3d(mid)
        else:
            vol = fov + self.net3d3d(fov)
        return vol
        

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
            fov_depth=self.model_cfg.fov_depth,
        )

        self.unetmodel = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,  # Condition with straight/hidden view
            out_channels=1,
            num_channels=(32, 32, 64),
            attention_levels=(False, False, True),
            num_res_blocks=1,
            num_head_channels=64,
            with_conditioning=False,
            # cross_attention_dim=9,  # flatR | flatT
        )
        self.scheduler = hydra.utils.instantiate(self.model_cfg.scheduler)
        self.inferer = hydra.utils.instantiate(self.model_cfg.inferer)

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
        image3d = self.correct_window(image3d, a_min=-1024, a_max=3071, b_min=-512, b_max=3071)
        return self.fwd_renderer(image3d, cameras, norm_type="standardized", stratified_sampling=is_training)

    def forward_volume(self, image2d, cameras, n_views=[2, 1], resample=True, timesteps=None, is_training=False):
        _device = image2d.device
        B = image2d.shape[0]
        assert B == sum(n_views)  # batch must be equal to number of projections
        
        # Transpose on the fly to make it homogeneous
        if resample:
            image2d = torch.flip(image2d, [2, 3])
        else:
            image2d = torch.flip(image2d, [3])
        image2d = image2d.transpose(2, 3)
        
        results = self.inv_renderer(image2d, cameras, resample, timesteps, is_training)
        return results
    
    def forward_timing(self, image3d, cameras=None, n_views=[2, 1], noise=None, resample=False, timesteps=None):
        _device = image3d.device
        B = image3d.shape[0]
        assert B == sum(n_views)  

        if timesteps is None:
            timesteps = torch.zeros((B,), device=_device).long()
                    
        results = self.inferer(
            inputs=image3d * 2.0 - 1.0, 
            diffusion_model=self.unetmodel, 
            noise=noise * 2.0 - 1.0, 
            timesteps=timesteps
        ) * 0.5 + 0.5
        return results
    
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
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=16, znear=4, zfar=8)

        dist_hidden = 6.0 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=16, znear=4, zfar=8)

        # Construct the samples in 2D
        figure_xr_hidden = image2d
        figure_ct_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)

        if self.model_cfg.phase == "ctproj":
            pass
        else:
            # Reconstruct the Encoder-Decoder
            volume_dx_concat = self.forward_volume(
                image2d=torch.cat([figure_xr_hidden, figure_ct_hidden]), 
                cameras=join_cameras_as_batch([view_hidden, view_hidden]), 
                n_views=[1, 1] * B, 
                resample=True,
                timesteps=None, 
                is_training=(stage=="train"),
            )
            (
                volume_xr_hidden_latent, 
                volume_ct_hidden_latent
            ) = torch.split(volume_dx_concat, B)
            
            ### @ Diffusion step: 2 kinds of blending
            timesteps = torch.randint(0, self.model_cfg.scheduler.num_train_timesteps, (B,), device=_device).long()  # 3 views
            # volume_ct_random_interp = self.scheduler.add_noise(original_samples=image3d, noise=volume_ct_random_latent, timesteps=timesteps)
            # volume_ct_hidden_interp = self.scheduler.add_noise(original_samples=image3d, noise=volume_ct_hidden_latent, timesteps=timesteps)  
            
            # Run the backward diffusion (denoising + reproject)
            volume_dx_output = self.forward_timing(
                image3d=torch.cat([image3d]), 
                cameras=join_cameras_as_batch([view_hidden]), 
                noise=torch.cat([volume_ct_hidden_latent]), 
                n_views=[1] * B, 
                timesteps=timesteps.repeat(2),
            )
            volume_ct_hidden_output = torch.split(volume_dx_output, B)

            if self.scheduler.prediction_type == "sample":
                volume_ct_target = image3d
            elif self.scheduler.prediction_type == "epsilon":
                pass
            elif self.scheduler.prediction_type == "v_prediction":
                pass

            im3d_loss_dif = F.l1_loss(volume_ct_hidden_output, volume_ct_target) 
            im3d_loss = im3d_loss_dif
            self.log(f"{stage}_im3d_loss", im3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
            loss = self.train_cfg.alpha * im3d_loss 

            # Visualization step
            if batch_idx == 0:
                # Sampling step for X-ray
                with torch.no_grad():
                    volume_dx_sample = volume_dx_concat
                    volume_dx_sample = self.inferer.sample(
                        input_noise=volume_dx_sample * 2.0 - 1.0, 
                        diffusion_model=self.unetmodel, 
                        verbose=False
                    ) * 0.5 + 0.5
                    (
                        volume_xr_hidden_sample, 
                        volume_ct_hidden_sample
                    ) = torch.split(volume_dx_sample, B)
                    
                    view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=16, znear=4, zfar=8)
                    view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=16, znear=4, zfar=8)
                    figure_xr_sample_hidden_random = self.forward_screen(image3d=volume_xr_hidden_sample, cameras=view_random, is_training=(stage=="train"))
                    figure_xr_sample_hidden_hidden = self.forward_screen(image3d=volume_xr_hidden_sample, cameras=view_hidden, is_training=(stage=="train"))
                    figure_ct_sample_hidden_random = self.forward_screen(image3d=volume_ct_hidden_sample, cameras=view_random, is_training=(stage=="train"))
                    figure_ct_sample_hidden_hidden = self.forward_screen(image3d=volume_ct_hidden_sample, cameras=view_hidden, is_training=(stage=="train"))

                zeros = torch.zeros_like(image2d)
                viz2d = torch.cat([
                    torch.cat([
                        image2d, 
                        volume_xr_hidden_latent[..., self.model_cfg.vol_shape // 2, :], 
                        volume_xr_hidden_sample[..., self.model_cfg.vol_shape // 2, :], 
                        figure_xr_sample_hidden_random,
                        figure_xr_sample_hidden_hidden,
                    ], dim=-2).transpose(2, 3),
                    torch.cat([
                        figure_ct_hidden,
                        volume_ct_hidden_latent[..., self.model_cfg.vol_shape // 2, :],
                        volume_ct_hidden_sample[..., self.model_cfg.vol_shape // 2, :],
                        figure_ct_sample_hidden_random, 
                        figure_ct_sample_hidden_hidden
                    ], dim=-2).transpose(2, 3),
                ], dim=-2)

                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(0, 1)
                tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx)            
        
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
