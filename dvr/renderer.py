import torch
import torch.nn as nn

from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    VolumeRenderer,
    MonteCarloRaysampler,
    NDCMultinomialRaysampler,
)
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from dvr.raymarcher import AbsorptionEmissionRaymarcher

def rescaled(x, val=64, eps=1e-8):
    return (x + eps) / (val + eps)

def minimized(x, eps=1e-8):
    return (x + eps) / (x.max() + eps)

def normalized(x, eps=1e-8):
    return (x - x.min() + eps) / (x.max() - x.min() + eps)

def standardized(x, eps=1e-8):
    return (x - x.mean()) / (x.std() + eps)  # 1e-6 to avoid zero division

def equalized(x, eps=1e-8):
    return equalize(x)

class ForwardXRayVolumeRenderer(nn.Module):
    def __init__(
        self, 
        image_width: int = 256, 
        image_height: int = 256, 
        n_pts_per_ray: int = 320, 
        min_depth: float = 3.0, 
        max_depth: float = 9.0, 
        ndc_extent: float = 1.0, 
        data_dim: int = 1024,
        feat_dim: int = 3,
        stratified_sampling: bool = False,
    ):
        super().__init__()
        self.n_pts_per_ray = n_pts_per_ray
        self.image_width = image_width
        self.image_height = image_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        # self.raymarcher = EmissionAbsorptionRaymarcher()  # BackToFront Raymarcher
        # self.raysampler = NDCMultinomialRaysampler(image_width=image_width, 
        #                                            image_height=image_height, 
        #                                            n_pts_per_ray=n_pts_per_ray, 
        #                                            min_depth=min_depth, 
        #                                            max_depth=max_depth, 
        #                                            stratified_sampling=stratified_sampling,)
        # self.renderer = VolumeRenderer(raysampler=self.raysampler, raymarcher=self.raymarcher,)
        self.ndc_extent = ndc_extent
            

    def forward(self, 
        image3d, 
        cameras, 
        opacity=None, 
        norm_type="minimized", 
        scaling_factor=0.1, 
        is_grayscale=True, 
        return_bundle=False, 
        stratified_sampling=False,
    ) -> torch.Tensor:
        
        features = image3d.repeat(1, 3, 1, 1, 1) if image3d.shape[1] == 1 else image3d
        if opacity is None:
            densities = torch.ones_like(image3d[:, [0]]) * scaling_factor
        else:
            densities = opacity * scaling_factor
        # print(image3d.shape, densities.shape)
        shape = max(image3d.shape[1], image3d.shape[2])
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=2.0*float(self.ndc_extent)/shape,
            # volume_translation = [-0.5, -0.5, -0.5],
        )
        
        # screen_RGBA, ray_bundles = self.renderer(cameras=cameras, volumes=volumes) #[...,:3]
        # rays_points = ray_bundle_to_ray_points(ray_bundles)
        # self.raysampler._stratified_sampling = stratified_sampling
        # self.renderer = VolumeRenderer(raysampler=self.raysampler, raymarcher=self.raymarcher,)
        raymarcher = EmissionAbsorptionRaymarcher()  # BackToFront Raymarcher

        # if stratified_sampling:
        #     raysampler = NDCMultinomialRaysampler(
        #         image_width=self.image_width, 
        #         image_height=self.image_height, 
        #         n_pts_per_ray=self.n_pts_per_ray, 
        #         min_depth=self.min_depth, 
        #         max_depth=self.max_depth, 
        #         stratified_sampling=stratified_sampling
        #     )
        #     # raysampler = MonteCarloRaysampler(
        #     #     min_x=-1.0,
        #     #     max_x=+1.0,
        #     #     min_y=-1.0,
        #     #     max_y=+1.0,
        #     #     n_rays_per_image=self.image_width*self.image_height,
        #     #     n_pts_per_ray=self.n_pts_per_ray, 
        #     #     min_depth=self.min_depth,
        #     #     max_depth=self.max_depth,
        #     # )
        # else:
        #     raysampler = NDCMultinomialRaysampler(
        #         image_width=self.image_width, 
        #         image_height=self.image_height, 
        #         n_pts_per_ray=self.n_pts_per_ray, 
        #         min_depth=self.min_depth, 
        #         max_depth=self.max_depth, 
        #         stratified_sampling=stratified_sampling
        #     )
        raysampler = NDCMultinomialRaysampler(
                image_width=self.image_width, 
                image_height=self.image_height, 
                n_pts_per_ray=self.n_pts_per_ray, 
                min_depth=self.min_depth, 
                max_depth=self.max_depth, 
                stratified_sampling=stratified_sampling
            )
        renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)  
        screen_RGBA, bundle = renderer(cameras=cameras, volumes=volumes)  # [...,:3]

        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1)  # 3 for NeRF
        if is_grayscale:
            screen_RGB = screen_RGBA[:, :].mean(dim=1, keepdim=True)
        else:
            screen_RGB = screen_RGBA[:, :]

        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))

        if return_bundle:
            return screen_RGB, bundle
        return screen_RGB


class ReverseXRayVolumeRenderer(ForwardXRayVolumeRenderer):
    def __init__(
        self, 
        image_width: int = 256, 
        image_height: int = 256, 
        n_pts_per_ray: int = 320, 
        min_depth: float = 3.0, 
        max_depth: float = 9.0, 
        ndc_extent: float = 1.0, 
        data_dim: int = 1024,
        feat_dim: int = 3,
        stratified_sampling: bool = False,
    ):
        super().__init__()


    def forward(self, 
        image3d, 
        cameras, 
        opacity=None, 
        norm_type="minimized", 
        scaling_factor=1, 
        is_grayscale=True, 
        return_bundle=False, 
        stratified_sampling=False,
    ) -> torch.Tensor:
        
        features = image3d.repeat(1, 3, 1, 1, 1) if image3d.shape[1] == 1 else image3d
        if opacity is None:
            densities = torch.ones_like(image3d[:, [0]]) * scaling_factor
        else:
            densities = opacity * scaling_factor
        # print(image3d.shape, densities.shape)
        shape = max(image3d.shape[1], image3d.shape[2])
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=2.0*float(self.ndc_extent)/shape,
            # volume_translation = [-0.5, -0.5, -0.5],
        )
        
        raymarcher = AbsorptionEmissionRaymarcher()  # BackToFront Raymarcher

        # if stratified_sampling:
        #     raysampler = NDCMultinomialRaysampler(
        #         image_width=self.image_width, 
        #         image_height=self.image_height, 
        #         n_pts_per_ray=self.n_pts_per_ray, 
        #         min_depth=self.min_depth, 
        #         max_depth=self.max_depth, 
        #         stratified_sampling=stratified_sampling
        #     )
        #     # raysampler = MonteCarloRaysampler(
        #     #     min_x=-1.0,
        #     #     max_x=+1.0,
        #     #     min_y=-1.0,
        #     #     max_y=+1.0,
        #     #     n_rays_per_image=self.image_width*self.image_height,
        #     #     n_pts_per_ray=self.n_pts_per_ray, 
        #     #     min_depth=self.min_depth,
        #     #     max_depth=self.max_depth,
        #     # )
        # else:
        #     raysampler = NDCMultinomialRaysampler(
        #         image_width=self.image_width, 
        #         image_height=self.image_height, 
        #         n_pts_per_ray=self.n_pts_per_ray, 
        #         min_depth=self.min_depth, 
        #         max_depth=self.max_depth, 
        #         stratified_sampling=stratified_sampling
        #     )
        raysampler = NDCMultinomialRaysampler(
                image_width=self.image_width, 
                image_height=self.image_height, 
                n_pts_per_ray=self.n_pts_per_ray, 
                min_depth=self.min_depth, 
                max_depth=self.max_depth, 
                stratified_sampling=stratified_sampling
            )
        renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)  
        screen_RGBA, bundle = renderer(cameras=cameras, volumes=volumes)  # [...,:3]

        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1)  # 3 for NeRF
        if is_grayscale:
            screen_RGB = screen_RGBA[:, :].mean(dim=1, keepdim=True)
        else:
            screen_RGB = screen_RGBA[:, :]

        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "rescaled":
            screen_RGB = standardized(rescaled(screen_RGB, val=32))
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))

        if return_bundle:
            return screen_RGB, bundle
        return screen_RGB