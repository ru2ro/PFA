import torch
import torch.nn as nn
import torch.nn.functional as F

# import numpy as np
# import matplotlib.pyplot as plt
#
# def set_axes_equal(ax):
#     '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
#     cubes as cubes, etc..  This is one possible solution to Matplotlib's
#     ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
#
#     Input
#       ax: a matplotlib axis, e.g., as output from plt.gca().
#     '''
#
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()
#
#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = np.mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = np.mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = np.mean(z_limits)
#
#     # The plot bounding box is a sphere in the sense of the infinity
#     # norm, hence I call half the max range the plot radius.
#     plot_radius = 0.5*max([x_range, y_range, z_range])
#
#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
#
#
# def show_voxel(pred_heatmap3d, ax=None):
#
#     if ax is None:
#         ax = plt.subplot(111, projection='3d')
#
#     view_angle = (-160, 30)
#     ht_map = pred_heatmap3d[0]
#     density = ht_map.flatten()
#     density = np.clip(density, 0, 1)
#     density /= density.sum()
#     selected_pt = np.random.choice(range(len(density)), 20000, p=density)
#     pt3d = np.unravel_index(selected_pt, ht_map.shape)
#     density_map = ht_map[pt3d]
#
#     # ax.set_aspect('equal')
#     ax.set_aspect('auto')
#     ax.scatter(pt3d[0], pt3d[2], pt3d[1], c=density_map, s=2, marker='.', linewidths=0)
#     set_axes_equal(ax)
#     # ax.set_xlabel('d', fontsize=10)
#     # ax.set_ylabel('w', fontsize=10)
#     # ax.set_zlabel('h', fontsize=10)
#     ax.view_init(*view_angle)
#
#     ax.xaxis.set_ticks([])
#     ax.yaxis.set_ticks([])
#     ax.zaxis.set_ticks([])
#
#     ax.set_xlabel('', fontsize=10)
#     ax.set_ylabel('', fontsize=10)
#     ax.set_zlabel('', fontsize=10)

class AddCoordsTh(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).cuda()
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).cuda()
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).cuda()
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).cuda()
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        # xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1).cpu()
        # yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1).cpu()

        if self.with_boundary and type(heatmap) != type(None):
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :],
                                        0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel>0.05,
                                              xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel>0.05,
                                              yy_channel, zero_tensor)
        if self.with_boundary and type(heatmap) != type(None):
            xx_boundary_channel = xx_boundary_channel.cuda()
            yy_boundary_channel = yy_boundary_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and type(heatmap) != type(None):
            ret = torch.cat([ret, xx_boundary_channel,
                             yy_boundary_channel], dim=1)
        return ret

class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r,
                                    with_boundary=with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class AddBoundaryCoord(nn.Module):
    def __init__(self):
        super(AddBoundaryCoord, self).__init__()

    def forward(self, xx_channel, yy_channel, heatmap):
        boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)

        zero_tensor = torch.zeros_like(xx_channel)
        xx_boundary_channel = torch.where(boundary_channel>0.05,
                                          xx_channel, zero_tensor)
        yy_boundary_channel = torch.where(boundary_channel>0.05,
                                          yy_channel, zero_tensor)

        xx_boundary_channel = xx_boundary_channel.cuda()
        yy_boundary_channel = yy_boundary_channel.cuda()
        ret = torch.cat([xx_boundary_channel, yy_boundary_channel], dim=1)

        return ret

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel / (x_dim - 1)
        yy_channel = yy_channel / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        if input_tensor.is_cuda:
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            if input_tensor.is_cuda:
                rr = rr.cuda()
            ret = torch.cat([ret, rr], dim=1)

        return ret



# class CoordConv(nn.Module):
#     def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
#         super().__init__()
#         self.addcoords = AddCoords(with_r=with_r)
#         self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
#
#     def forward(self, x):
#         ret = self.addcoords(x)
#         ret = self.conv(ret)
#         return ret


class AddCoordsTh3D(nn.Module):
    def __init__(self, x_dim=64, y_dim=64,z_dim=64, with_r=False, with_boundary=False):
        super(AddCoordsTh3D, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, z_dim, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).cuda()
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).cuda()
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)
        xx_channel = xx_channel.permute(0, 3, 2, 1).unsqueeze(2)
        xx_channel = xx_channel / (self.x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, self.z_dim, 1, 1)

        yy_channel = xx_channel.permute(0, 1, 2, 4, 3)
        zz_channel = xx_channel.permute(0, 1, 3, 2, 4)

        zz_channel = F.interpolate(zz_channel, size=(self.z_dim, self.x_dim, self.y_dim))


        # vox_chan = [yy_channel[0, 0].cpu().numpy() / 63]
        # show_voxel(vox_chan, ax=plt.subplot(111, projection='3d'))
        # plt.show()

        # import cv2
        # xx_tmp = xx_channel.detach().cpu().numpy()
        # xx_tmp = xx_tmp[0, 0, 0].astype('uint8') * 4
        # xx_tmp = cv2.resize(xx_tmp, (256, 256))
        # cv2.imshow('xx', xx_tmp)
        #
        # yy_tmp = yy_channel.detach().cpu().numpy()
        # yy_tmp = yy_tmp[0, 0, 0].astype('uint8') * 4
        # yy_tmp = cv2.resize(yy_tmp, (256, 256))
        # cv2.imshow('yy', yy_tmp)
        #
        # zz_tmp = zz_channel.detach().cpu().numpy()
        # zz_tmp = zz_tmp[0, 0, :, :, 0].astype('uint8') * 4
        # zz_tmp = cv2.resize(zz_tmp, (256, 256))
        # cv2.imshow('zz', zz_tmp)
        # cv2.waitKey(0)

        # yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).cuda()
        # yy_ones = yy_ones.unsqueeze(1)
        #
        # yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).cuda()
        # yy_range = yy_range.unsqueeze(-1)
        #
        # yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        # yy_channel = yy_channel.unsqueeze(-1)
        #
        # xx_channel = xx_channel.permute(0, 3, 2, 1)
        # yy_channel = yy_channel.permute(0, 3, 2, 1)
        #
        # xx_channel = xx_channel / (self.x_dim - 1)
        # yy_channel = yy_channel / (self.y_dim - 1)
        #
        # xx_channel = xx_channel * 2 - 1
        # yy_channel = yy_channel * 2 - 1
        #
        # xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        # yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        # xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1).cpu()
        # yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1).cpu()

        ret = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

        if self.with_boundary and type(heatmap) != type(None):
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :, :], 0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)
            zz_boundary_channel = torch.where(boundary_channel > 0.05, zz_channel, zero_tensor)

            xx_boundary_channel = xx_boundary_channel.cuda()
            yy_boundary_channel = yy_boundary_channel.cuda()
            zz_boundary_channel = zz_boundary_channel.cuda()

            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel, zz_boundary_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2) + torch.pow(zz_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConvTh3D(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, z_dim, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh3D, self).__init__()
        self.addcoords = AddCoordsTh3D(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, with_r=with_r,
                                    with_boundary=with_boundary)
        in_channels += 3
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 3
        self.conv = nn.Conv3d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        ret = self.conv(ret)
        return ret

if __name__ == "__main__":
    x_h = torch.zeros((2, 1, 64, 64, 64), device='cpu').cuda()

    addchan = CoordConvTh3D(x_dim=64, y_dim=64, z_dim=64, with_r=True, with_boundary=True,
                            in_channels=1, out_channels=256, kernel_size=1, stride=1, padding=0)
    addchan.cuda()
    x_h2 = addchan(x_h, x_h)

    # net = PCNet(68)
    #
    # pose_pred, coord_pred = net(x_h)

    a = 0
