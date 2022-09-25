import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Projection(object):
    def __init__(self,focal_ratio = (350.0/320.0,350.0/240.0),near = 5, far = 16,
    frustum_size = [128,128,128], device = device,nss_scale = 7, render_size = (64,64)):
        self.render_size  = render_size
        self.device       = device
        self.focal_ratio  = focal_ratio
        self.near         = near
        self.far          = far
        self.frustum_size = frustum_size

        self.nss_scale = nss_scale
        self.world2nss = torch.tensor([[1/nss_scale, 0, 0, 0],
                                        [0, 1/nss_scale, 0, 0],
                                        [0, 0, 1/nss_scale, 0],
                                        [0, 0, 0, 1]]).unsqueeze(0).to(device)
        focal_x = self.focal_ratio[0] * self.frustum_size[0]
        focal_y = self.focal_ratio[1] * self.frustum_size[1]
        bias_x  = (self.frustum_size[0] - 1.0)/2
        bias_y  = (self.frustum_size[1] - 1.0)/2
    
        intrinsic_mat = torch.tensor([[focal_x, 0, bias_x, 0],
                                      [0, focal_y, bias_y, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        self.cam2spixel = intrinsic_mat.to(self.device)
        self.spixel2cam = intrinsic_mat.inverse().to(self.device)
    
    def construct_frus_coor(self):
        x = torch.arange(self.frsutum_size[0])
        y = torch.arange(self.frustum_size[1])
        z = torch.arange(self.frsutum_size[2])
        x, y, z = torch.meshgrid([x,y,z])
        x_frus = x.flatten().to(self.device)
        y_frus = y.flatten().to(self.device)
        z_frus = z.flatten().to(self.device)
    
        # project the frustum points ot vol coord
        depth_range = torch.linspace(self.near,self.far,self.frustum_size[2])
        z_cam = depth_range[z_frus].to(self.device)

        x_unnorm_pix = x_frus * z_cam
        y_unnorm_pix = y_frus * z_cam
        z_unnorm_pix = z_cam
        pixel_coor = torch.stack([x_unnorm_pix,y_unnorm_pix,z_unnorm_pix,torch.ones_like(x_unnorm_pix)])
        return pixel_coor
    
    def construct_sampling_coor(self,cam2world,partitioned = False):
        """
        construct a sampling frustum coor in NSS space, and geneate z_val/ray_dir
        input:
            cam2world: Nx4x4, N: #images to render
        output:
            frus_nss_coor: (NxDxHxW) x 3
            z_vals: (NxHxW) x D
            ray_dir: (N,H,W) x 3 
        """
        N = cam2world.shape[0]
        W, H, D = self.frustum_size

        pixel_coor = self.construct_frus_coor()
        frus_cam_coor = torch.matmul(self.spixel2cam,pixel_coor.float()) # 4x(WxHxD)

        frus_world_coor = torch.matmul(cam2world,frus_cam_coor)      # Nx4x(WxHxD)
        frus_nss_coor   = torch.matmul(self.world2nss,frus_cam_coor) # Nx4x(WxHxD)
        frus_nss_coor   = frus_nss_coor.view(N,4,W,H,D).permute([0,4,3,2,1]) # NxDxHxWx4
        frus_nss_coor   = frus_nss_coor[...,:3] # NxDxHxWx3
        scale = H // self.render_size[0]
        if partitioned:
            frus_nss_coor_ = []