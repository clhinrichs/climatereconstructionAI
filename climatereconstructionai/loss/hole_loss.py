import torch
from torch import nn


class HoleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, data_dict):
        loss_dict = {
            'hole': 0.0
        }

        output = data_dict['output']
        gt = data_dict['gt']
        mask = data_dict['mask']

        # calculate loss for all channels
        for channel in range(output.shape[1]):
            # only select first channel
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            # define different loss functions from output and output_comp
            loss_dict['hole'] += self.l1((1 - mask_ch) * output_ch, (1 - mask_ch) * gt_ch)
        return loss_dict

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, data_dict):
        output = data_dict["output"]
        gt = data_dict["gt"]
        mask = data_dict["mask"]

        loss = 0.0

        for channel in range(output.shape[1]):
            output_ch = output[:, channel:channel+1]
            gt_ch = gt[:, channel:channel+1]
            mask_ch = mask[:, channel:channel+1]

            # Spatial gradients
            output_dx = output_ch[:, :, :, 1:] - output_ch[:, :, :, :-1]
            gt_dx = gt_ch[:, :, :, 1:] - gt_ch[:, :, :, :-1]

            output_dy = output_ch[:, :, 1:, :] - output_ch[:, :, :-1, :]
            gt_dy = gt_ch[:, :, 1:, :] - gt_ch[:, :, :-1, :]

            # Only compare gradients where both pixels are reconstructed
            hole_dx = (mask_ch[:, :, :, 1:] == 0) & (mask_ch[:, :, :, :-1] == 0)
            hole_dy = (mask_ch[:, :, 1:, :] == 0) & (mask_ch[:, :, :-1, :] == 0)

            if hole_dx.any():
                loss += self.l1(output_dx[hole_dx], gt_dx[hole_dx])

            if hole_dy.any():
                loss += self.l1(output_dy[hole_dy], gt_dy[hole_dy])

        return {"grad": loss}