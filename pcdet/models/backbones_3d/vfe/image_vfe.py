import torch

from .vfe_template import VFETemplate
from .image_vfe_modules import ffn, f2v


class ImageVFE(VFETemplate):
    def __init__(self, model_cfg, grid_size, point_cloud_range, depth_downsample_factor, use_pooling=False, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.downsample_factor = depth_downsample_factor
        self.use_pooling = use_pooling
        self.module_topology = [
            'ffn', 'f2v'
        ]
        self.use_lidar_depth = model_cfg.FFN.get('USE_LIDAR_DEPTH', False)
        self.use_depth = model_cfg.get('USE_DEPTH', True)
        self.feature_num = model_cfg.F2V.get('FEATURE_NUM', None)
        self.fuse_mode = model_cfg.F2V.get('FUSE_MODE', None)
        self.out_channels_f2v = model_cfg.F2V.get('OUT_CHANNELS', None)
        self.in_channels_f2v = model_cfg.FFN.CHANNEL_REDUCE['out_channels']
        self.build_modules()

    def build_modules(self):
        """
        Builds modules
        """
        for module_name in self.module_topology:
            module = getattr(self, 'build_%s' % module_name)()
            self.add_module(module_name, module)

    def build_ffn(self):
        """
        Builds frustum feature network
        Returns:
            ffn_module: nn.Module, Frustum feature network
        """
        ffn_module = ffn.__all__[self.model_cfg.FFN.NAME](
            model_cfg=self.model_cfg.FFN,
            downsample_factor=self.downsample_factor,
            use_lidar_depth=self.use_lidar_depth,
            use_pooling=self.use_pooling,
            use_depth=self.use_depth
        )
        self.disc_cfg = ffn_module.disc_cfg
        return ffn_module

    def build_f2v(self):
        """
        Builds frustum to voxel transformation
        Returns:
            f2v_module: nn.Module, Frustum to voxel transformation
        """
        f2v_module = f2v.__all__[self.model_cfg.F2V.NAME](
            model_cfg=self.model_cfg.F2V,
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            disc_cfg=self.disc_cfg,
            use_depth=self.use_depth,
            feature_num=self.feature_num,
            fuse_mode=self.fuse_mode,
            in_channels=self.in_channels_f2v,
            out_channels=self.out_channels_f2v
        )
        return f2v_module

    def get_output_feature_dim(self):
        """
        Gets number of output channels
        Returns:
            out_feature_dim: int, Number of output channels
        """
        out_feature_dim = self.ffn.get_output_feature_dim()
        return out_feature_dim

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
            **kwargs:
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        batch_dict = self.ffn(batch_dict)
        batch_dict = self.f2v(batch_dict)
        return batch_dict

    def get_loss(self):
        """
        Gets DDN loss
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        if self.use_lidar_depth or (not self.use_depth):
            return None, None
        loss, tb_dict = self.ffn.get_loss()
        return loss, tb_dict
