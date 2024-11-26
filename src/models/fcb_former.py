import os
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
from utils import calculate_iou, add_to_test_results
import config

# for fcbFormer
from timm.models.vision_transformer import _cfg
from functools import partial
from . import pvt_v2


class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=512,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        # checkpoint = torch.load("/home/mikolaj/oke_laptop/Programs/breast_cancer/segmentation/imports/FCBFormer/pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        # backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64), nn.Upsample(size=88)
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l


class FCBFormer(nn.Module):
    def __init__(self, size=512):

        super().__init__()

        self.TB = TB()

        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out# Import your FCBFormer model here

# Training function
def train_fcbformer(save_path, data_loader, input_size=512):
    # Initialize the FCBFormer model
    model = FCBFormer(size=input_size)
    
    # Move the model to GPU if available
    if torch.cuda.is_available():
        device_id = [0, 1]  # Choose GPU devices
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            model = nn.DataParallel(model, device_ids=device_id)
        torch.cuda.set_device(device_id[0])
        model = model.cuda()
    else:
        model = model.cpu()
        print("CUDA is not available. Training on CPU.")

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 20  # Set the number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iou_scores = []

        for batch_idx, (images, masks) in enumerate(data_loader):
            # Check for problematic batch
            if images.shape[0] < 2 or images.shape[2] < 2 or images.shape[3] < 2 or \
                    masks.shape[0] < 2 or masks.shape[2] < 2 or masks.shape[3] < 2:
                print(f"Skipping batch {batch_idx} due to unexpected size. "
                      f"Image shape: {images.shape}, Mask shape: {masks.shape}")
                continue

            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)

            if masks.dim() == 4:
                masks = masks.squeeze(1)

            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iou_score = calculate_iou(torch.sigmoid(outputs), masks)
            iou_scores.append(iou_score)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], "
                      f"Loss: {loss.item():.4f}, IoU: {iou_score:.4f}")

        epoch_loss = running_loss / len(data_loader)
        epoch_iou = np.mean(iou_scores)
        print(f"End of Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss:.4f}, Average IoU: {epoch_iou:.4f}")

    print('Finished Training')

    os.makedirs(save_path, exist_ok=True)
    torch.save(model, os.path.join(save_path, 'fcbformer_segmentation.pth'))


# Testing function
def test_fcbformer(result_path, dataset, feature_dataset_choice, data_loader, input_size=512):
    print("Testing FCBFormer on Feature: " + str(feature_dataset_choice) + " of " + dataset)

    # Load the trained model
    model_path = os.path.join(config.saved_models_path, 'FCBFormer', dataset, f'Feature_{feature_dataset_choice}', 'fcbformer_segmentation.pth')

    if not os.path.exists(model_path):
        print("The given model does not exist, Train the model before testing.")
        return

    model = torch.load(model_path, map_location=torch.device('cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    iou_scores = []

    for images, masks in data_loader:
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()

        with torch.no_grad():
            outputs = model(images)

        iou_score = calculate_iou(torch.sigmoid(outputs), masks)
        iou_scores.append(iou_score)

    average_iou = np.mean(iou_scores)
    print(f"Average IoU: {average_iou:.4f}")

    os.makedirs(config.results_path, exist_ok=True)
    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)

    print(f"Testing Successful. Results added to {result_path}")
