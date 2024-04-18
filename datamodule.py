import os
import glob

from typing import Callable, Optional, Sequence
from argparse import ArgumentParser

# from torch.utils.data import Dataset, DataLoader
from monai.data import CacheDataset, Dataset, DataLoader
from monai.data import list_data_collate
from monai.utils import set_determinism
from monai.transforms import (
    apply_transform,
    Randomizable,
    Compose,
    OneOf,
    EnsureChannelFirstDict,
    LoadImageDict,
    SpacingDict,
    OrientationDict,
    DivisiblePadDict,
    CropForegroundDict,
    ResizeDict,
    RandZoomDict,
    ZoomDict,
    RandRotateDict,
    HistogramNormalizeDict,
    ScaleIntensityDict,
    ScaleIntensityRangeDict,
    ToTensorDict,
)

from pytorch_lightning import LightningDataModule


class UnpairedDataset(CacheDataset, Randomizable):
    def __init__(
        self,
        keys: Sequence,
        data: Sequence,
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None,
        batch_size: int = 32,
        is_training: bool = True,
    ) -> None:
        self.keys = keys
        self.data = data
        self.length = length
        self.batch_size = batch_size
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.data))
        else:
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        for key, dataset in zip(self.keys, self.data):
            if self.is_training:
                rand_idx = self.R.randint(0, len(dataset))
                data[key] = dataset[rand_idx]
                data[f"{key}_idx"] = rand_idx
            else:
                data[key] = dataset[index]
                data[f"{key}_idx"] = index
            # Add the filename to the data dictionary
            data[f"{key}_pth"] = data[key]  # Assuming "image3d" contains the filename

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class UnpairedDataModule(LightningDataModule):
    def __init__(
        self,
        train_image3d_folders: str = "path/to/folder",
        train_image2d_folders: str = "path/to/folder",
        val_image3d_folders: str = "path/to/folder",
        val_image2d_folders: str = "path/to/folder",
        test_image3d_folders: str = "path/to/folder",
        test_image2d_folders: str = "path/to/dir",
        train_samples: int = 1000,
        val_samples: int = 400,
        test_samples: int = 400,
        img_shape: int = 512,
        vol_shape: int = 256,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        # self.setup()
        self.train_image3d_folders = train_image3d_folders
        self.train_image2d_folders = train_image2d_folders
        self.val_image3d_folders = val_image3d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image3d_folders = test_image3d_folders
        self.test_image2d_folders = test_image2d_folders
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: str = None, extension: str = "*.nii.gz"):
            assert folders is not None
            paths = [
                glob.glob(os.path.join(folder, extension), recursive=True)
                for folder in folders
            ]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files

        self.train_image3d_files = glob_files(
            folders=train_image3d_folders, extension="**/*.nii.gz"
        )
        self.train_image2d_files = glob_files(
            folders=train_image2d_folders, extension="**/*.png"
        )

        self.val_image3d_files = glob_files(
            folders=val_image3d_folders, extension="**/*.nii.gz"
        )  # TODO
        self.val_image2d_files = glob_files(
            folders=val_image2d_folders, extension="**/*.png"
        )

        self.test_image3d_files = glob_files(
            folders=test_image3d_folders, extension="**/*.nii.gz"
        )  # TODO
        self.test_image2d_files = glob_files(
            folders=test_image2d_folders, extension="**/*.png"
        )

    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImageDict(keys=["image3d", "image2d"]),
                EnsureChannelFirstDict(keys=["image3d", "image2d"],),
                SpacingDict(
                    keys=["image3d"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=["bilinear"],
                    align_corners=True,
                ),
                # Rotate90Dict(keys=["image2d"], k=3),
                # RandFlipDict(keys=["image2d"], prob=1.0, spatial_axis=1),
                OneOf(
                    [
                        OrientationDict(keys=("image3d"), axcodes="ASL"),
                        # OrientationDict(keys=('image3d'), axcodes="ARI"),
                        # OrientationDict(keys=('image3d'), axcodes="PRI"),
                        # OrientationDict(keys=('image3d'), axcodes="ALI"),
                        # OrientationDict(keys=('image3d'), axcodes="PLI"),
                        # OrientationDict(keys=["image3d"], axcodes="LAI"),
                        # OrientationDict(keys=["image3d"], axcodes="RAI"),
                        # OrientationDict(keys=["image3d"], axcodes="LPI"),
                        # OrientationDict(keys=["image3d"], axcodes="RPI"),
                    ],
                ),
                ScaleIntensityDict(keys=["image2d"], minv=0.0, maxv=1.0,),
                HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
                OneOf(
                    [
                        # ScaleIntensityRangeDict(keys=["image3d"], clip=True,  # CTXR range
                        #         a_min=-200,
                        #         a_max=1500,
                        #         b_min=0.0,
                        #         b_max=1.0
                        # ),
                        ScaleIntensityRangeDict(
                            keys=["image3d"],
                            clip=True,
                            a_min=-1024,
                            a_max=512,
                            b_min=0.0,
                            b_max=1.0,
                        ),  # Full range  # -200,  # 1500,
                    ]
                ),
                RandRotateDict(
                    keys=["image3d"],
                    prob=1.0,
                    range_x=0.1,
                    range_y=0.1,
                    range_z=0.1,
                    padding_mode="zeros",
                    mode=["bilinear"],
                    align_corners=True,
                ),
                RandZoomDict(
                    keys=["image3d"],
                    prob=1.0,
                    min_zoom=0.85,
                    max_zoom=1.10,
                    padding_mode="constant",
                    mode=["trilinear"],
                    align_corners=True,
                ),
                RandZoomDict(
                    keys=["image2d"],
                    prob=1.0,
                    min_zoom=0.85,
                    max_zoom=1.10,
                    padding_mode="constant",
                    mode=["area"],
                ),
                CropForegroundDict(
                    keys=["image3d"],
                    source_key="image3d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                CropForegroundDict(
                    keys=["image2d"],
                    source_key="image2d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                # RandZoomDict(keys=["image3d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["trilinear"], align_corners=True),
                # RandZoomDict(keys=["image2d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]),
                # RandFlipdict(keys=["image3d"], prob=0.5, spatial_axis=0),
                # RandFlipdict(keys=["image3d"], prob=0.5, spatial_axis=1),
                # RandScaleCropDict(keys=["image3d"],
                #                roi_scale=(0.9, 0.9, 0.8),
                #                max_roi_scale=(1.0, 1.0, 0.8),
                #                random_center=False,
                #                random_size=False),
                # RandAffineDict(keys=["image3d"], rotate_range=None, shear_range=None, translate_range=20, scale_range=None),
                # CropForegroundDict(keys=["image3d"], source_key="image3d", select_fn=lambda x: x>0, margin=0),
                # CropForegroundDict(keys=["image2d"], source_key="image2d", select_fn=lambda x: x>0, margin=0),
                # ZoomDict(keys=["image3d"], zoom=0.9, padding_mode="constant", mode=["area"]),
                ZoomDict(
                    keys=["image2d"], zoom=0.9, padding_mode="constant", mode=["area"]
                ),
                ResizeDict(
                    keys=["image3d"],
                    spatial_size=self.vol_shape,
                    size_mode="longest",
                    mode=["trilinear"],
                    align_corners=True,
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.img_shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image3d"],
                    k=self.vol_shape,
                    mode="constant",
                    constant_values=0,
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.img_shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(keys=["image3d", "image2d"],),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.train_image3d_files, self.train_image2d_files],
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.train_loader = DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImageDict(keys=["image3d", "image2d"]),
                EnsureChannelFirstDict(keys=["image3d", "image2d"],),
                SpacingDict(
                    keys=["image3d"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=["bilinear"],
                    align_corners=True,
                ),
                # Rotate90Dict(keys=["image2d"], k=3),
                # RandFlipDict(keys=["image2d"], prob=1.0, spatial_axis=1), #Right cardio
                OneOf(
                    [
                        OrientationDict(keys=("image3d"), axcodes="ASL"),
                        # OrientationDict(keys=('image3d'), axcodes="ARI"),
                        # OrientationDict(keys=('image3d'), axcodes="PRI"),
                        # OrientationDict(keys=('image3d'), axcodes="ALI"),
                        # OrientationDict(keys=('image3d'), axcodes="PLI"),
                        # OrientationDict(keys=["image3d"], axcodes="LAI"),
                        # OrientationDict(keys=["image3d"], axcodes="RAI"),
                        # OrientationDict(keys=["image3d"], axcodes="LPI"),
                        # OrientationDict(keys=["image3d"], axcodes="RPI"),
                    ],
                ),
                ScaleIntensityDict(keys=["image2d"], minv=0.0, maxv=1.0,),
                HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
                OneOf(
                    [
                        # ScaleIntensityRangeDict(keys=["image3d"], clip=True,  # CTXR range
                        #         a_min=-200,
                        #         a_max=1500,
                        #         b_min=0.0,
                        #         b_max=1.0),
                        ScaleIntensityRangeDict(
                            keys=["image3d"],
                            clip=True,
                            a_min=-1024,
                            a_max=512,
                            b_min=0.0,
                            b_max=1.0,
                        ),  # Full range  # -200,  # 1500,
                    ]
                ),
                CropForegroundDict(
                    keys=["image3d"],
                    source_key="image3d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                CropForegroundDict(
                    keys=["image2d"],
                    source_key="image2d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                # ZoomDict(keys=["image3d"], zoom=0.9, padding_mode="constant", mode=["area"]),
                ZoomDict(
                    keys=["image2d"], zoom=0.9, padding_mode="constant", mode=["area"]
                ),
                ResizeDict(
                    keys=["image3d"],
                    spatial_size=self.vol_shape,
                    size_mode="longest",
                    mode=["trilinear"],
                    align_corners=True,
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.img_shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image3d"],
                    k=self.vol_shape,
                    mode="constant",
                    constant_values=0,
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.img_shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(keys=["image3d", "image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.val_image3d_files, self.val_image2d_files],
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.val_loader = DataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_transforms = Compose(
            [
                LoadImageDict(keys=["image3d", "image2d"]),
                EnsureChannelFirstDict(keys=["image3d", "image2d"],),
                SpacingDict(
                    keys=["image3d"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=["bilinear"],
                    align_corners=True,
                ),
                # Rotate90Dict(keys=["image2d"], k=3),
                # RandFlipDict(keys=["image2d"], prob=1.0, spatial_axis=1), #Right cardio
                OneOf(
                    [
                        OrientationDict(keys=("image3d"), axcodes="ASL"),
                        # OrientationDict(keys=('image3d'), axcodes="ARI"),
                        # OrientationDict(keys=('image3d'), axcodes="PRI"),
                        # OrientationDict(keys=('image3d'), axcodes="ALI"),
                        # OrientationDict(keys=('image3d'), axcodes="PLI"),
                        # OrientationDict(keys=["image3d"], axcodes="LAI"),
                        # OrientationDict(keys=["image3d"], axcodes="RAI"),
                        # OrientationDict(keys=["image3d"], axcodes="LPI"),
                        # OrientationDict(keys=["image3d"], axcodes="RPI"),
                    ],
                ),
                ScaleIntensityDict(keys=["image2d"], minv=0.0, maxv=1.0,),
                HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
                OneOf(
                    [
                        # ScaleIntensityRangeDict(keys=["image3d"], clip=True,  # CTXR range
                        #         a_min=-200,
                        #         a_max=1500,
                        #         b_min=0.0,
                        #         b_max=1.0),
                        ScaleIntensityRangeDict(
                            keys=["image3d"],
                            clip=False,
                            a_min=-1024,
                            a_max=512,
                            b_min=0.0,
                            b_max=1.0,
                        ),  # Full range  # -200,  # 1500,
                    ]
                ),
                CropForegroundDict(
                    keys=["image3d"],
                    source_key="image3d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                CropForegroundDict(
                    keys=["image2d"],
                    source_key="image2d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                # ZoomDict(keys=["image3d"], zoom=0.9, padding_mode="constant", mode=["area"]),
                ZoomDict(
                    keys=["image2d"], zoom=0.9, padding_mode="constant", mode=["area"]
                ),
                ResizeDict(
                    keys=["image3d"],
                    spatial_size=self.vol_shape,
                    size_mode="longest",
                    mode=["trilinear"],
                    align_corners=True,
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.img_shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image3d"],
                    k=self.vol_shape,
                    mode="constant",
                    constant_values=0,
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.img_shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(keys=["image3d", "image2d"],),
            ]
        )

        self.test_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.test_image3d_files, self.test_image2d_files],
            transform=self.test_transforms,
            length=self.test_samples,
            batch_size=self.batch_size,
            is_training=False,
        )

        self.test_loader = DataLoader(
            self.test_datasets,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return self.test_loader


class ExampleDataModule(LightningDataModule):
    def __init__(
        self,
        train_image3d_folders: str = "path/to/folder",
        train_image2d_folders: str = "path/to/folder",
        val_image3d_folders: str = "path/to/folder",
        val_image2d_folders: str = "path/to/folder",
        test_image3d_folders: str = "path/to/folder",
        test_image2d_folders: str = "path/to/dir",
        train_samples: int = 1000,
        val_samples: int = 400,
        test_samples: int = 400,
        img_shape: int = 512,
        vol_shape: int = 256,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        # self.setup()
        self.train_image3d_folders = train_image3d_folders
        self.train_image2d_folders = train_image2d_folders
        self.val_image3d_folders = val_image3d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image3d_folders = test_image3d_folders
        self.test_image2d_folders = test_image2d_folders
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: str = None, extension: str = "*.nii.gz"):
            assert folders is not None
            paths = [
                glob.glob(os.path.join(folder, extension), recursive=True)
                for folder in folders
            ]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files

        self.train_image3d_files = glob_files(
            folders=train_image3d_folders, extension="**/*.tif"
        )
        self.train_image2d_files = glob_files(
            folders=train_image2d_folders, extension="**/*.png"
        )

        self.val_image3d_files = glob_files(
            folders=val_image3d_folders, extension="**/*.tif"
        )  # TODO
        self.val_image2d_files = glob_files(
            folders=val_image2d_folders, extension="**/*.png"
        )

        self.test_image3d_files = glob_files(
            folders=test_image3d_folders, extension="**/*.tif"
        )  # TODO
        self.test_image2d_files = glob_files(
            folders=test_image2d_folders, extension="**/*.png"
        )

    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImageDict(keys=["image3d", "image2d"]),
                EnsureChannelFirstDict(keys=["image3d", "image2d"],),
                ScaleIntensityDict(keys=["image3d", "image2d"], minv=0.0, maxv=1.0,),
                HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
                ResizeDict(
                    keys=["image3d"],
                    spatial_size=self.vol_shape,
                    size_mode="longest",
                    mode=["trilinear"],
                    align_corners=True,
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.img_shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image3d"],
                    k=self.vol_shape,
                    mode="constant",
                    constant_values=0,
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.img_shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(keys=["image3d", "image2d"],),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.train_image3d_files, self.train_image2d_files],
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.train_loader = DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=32,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImageDict(keys=["image3d", "image2d"]),
                EnsureChannelFirstDict(keys=["image3d", "image2d"],),
                ScaleIntensityDict(keys=["image3d", "image2d"], minv=0.0, maxv=1.0,),
                HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
                ResizeDict(
                    keys=["image3d"],
                    spatial_size=self.vol_shape,
                    size_mode="longest",
                    mode=["trilinear"],
                    align_corners=True,
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.img_shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image3d"],
                    k=self.vol_shape,
                    mode="constant",
                    constant_values=0,
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.img_shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(keys=["image3d", "image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.val_image3d_files, self.val_image2d_files],
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.val_loader = DataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_transforms = Compose(
            [
                LoadImageDict(keys=["image3d", "image2d"]),
                EnsureChannelFirstDict(keys=["image3d", "image2d"],),
                ScaleIntensityDict(keys=["image3d", "image2d"], minv=0.0, maxv=1.0,),
                HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0,),
                ResizeDict(
                    keys=["image3d"],
                    spatial_size=self.vol_shape,
                    size_mode="longest",
                    mode=["trilinear"],
                    align_corners=True,
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.img_shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image3d"],
                    k=self.vol_shape,
                    mode="constant",
                    constant_values=0,
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.img_shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(keys=["image3d", "image2d"],),
            ]
        )

        self.test_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.test_image3d_files, self.test_image2d_files],
            transform=self.test_transforms,
            length=self.test_samples,
            batch_size=self.batch_size,
            is_training=False,
        )

        self.test_loader = DataLoader(
            self.test_datasets,
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return self.test_loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--datadir", type=str, default="data", help="data directory")
    parser.add_argument(
        "--img_shape", type=int, default=512, help="isotropic img shape"
    )
    parser.add_argument(
        "--vol_shape", type=int, default=256, help="isotropic vol shape"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    hparams = parser.parse_args()
    # Create data module
    train_image3d_folders = [
        os.path.join(
            hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"
        ),
        os.path.join(
            hparams.datadir,
            "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0",
        ),
        os.path.join(
            hparams.datadir,
            "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1",
        ),
        os.path.join(
            hparams.datadir,
            "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2",
        ),
        os.path.join(
            hparams.datadir,
            "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3",
        ),
        os.path.join(
            hparams.datadir,
            "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4",
        ),
        os.path.join(
            hparams.datadir, "ChestXRLungSegmentation/Imagenglab/processed/train/images"
        ),
    ]
    train_label3d_folders = []

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(
            hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/train/images/"
        ),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]
    train_label2d_folders = []

    val_image3d_folders = train_image3d_folders
    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(
            hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"
        ),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape,
        batch_size=hparams.batch_size,
    )
    datamodule.setup(seed=hparams.seed)

    #
    # Example
    #
    hparams = parser.parse_args()
    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, "Visualization/image3d"),
    ]
    train_image2d_folders = [
        os.path.join(hparams.datadir, "Visualization/image2d"),
    ]
    val_image3d_folders = [
        os.path.join(hparams.datadir, "Visualization/image3d"),
    ]
    val_image2d_folders = [
        os.path.join(hparams.datadir, "Visualization/image2d"),
    ]
    test_image3d_folders = [
        os.path.join(hparams.datadir, "Visualization/image3d"),
    ]
    test_image2d_folders = [
        os.path.join(hparams.datadir, "Visualization/image2d"),
    ]

    datamodule = ExampleDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape,
        batch_size=hparams.batch_size,
    )
    datamodule.setup(seed=hparams.seed)
