import torch
import torch.nn as nn
from torch.distributions import Normal
input_channel = 3

class encoder(nn.Module):
    def __init__(self, z_size=16):
        super(encoder, self).__init__()
        self.contract_layers = nn.Sequential(
            nn.Conv2d(input_channel+1, 32, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(64)
        )

        self.linear1 = nn.Linear(64 * 7 * 7, 256)
        self.linear2_logvar = nn.Linear(256, z_size)
        self.linea2_mu = nn.Linear(256, z_size)
        #self.score_net = FCBlock(132,3,256,1)
        #self.feature_net = FCBlock(132,3,256,100)
        self.relu = nn.ReLU()

    def reparameterize(self, logvar, mu):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, mask):
        x = torch.cat((image, mask), 1)
        x = self.contract_layers(x)
        bs, _, _, _ = x.shape
        x = x.view(bs, -1)
        x = self.linear1(x)
        x = self.relu(x)
        logvar = torch.nn.functional.softplus(self.linear2_logvar(x))+0.01
        mu = self.linea2_mu(x)
        # z = self.reparameterize(logvar, mu)
        m = Normal(mu, logvar)
        z = m.rsample()
        #score = torch.sigmoid(3 * self.score_net( x))
        #feature = self.feature_net(x)
        # z = mu
        return z, logvar, mu

# four convolutinal layer (conv1 - 4) to upsample the latent variable to size 128 * 128 * 32 and
# two convolutional layer (conv5_mask, conv5_img) to output the logit(mask) (1 channel) and image (3 channel)
# Get the mask by applying sigmoid on logit(mask)
class decoder(nn.Module):
    def __init__(self, inchannel):
        super(decoder, self).__init__()
        self.im_size = 128
        self.conv1 = nn.Conv2d(inchannel + 2, 32, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn4 = nn.BatchNorm2d(32)
        self.celu = nn.CELU()
        self.inchannel = inchannel
        self.conv5_img = nn.Conv2d(32, input_channel, 1)
        self.conv5_mask = nn.Conv2d(32, 1, 1)

        x = torch.linspace(-1, 1, self.im_size + 8)
        y = torch.linspace(-1, 1, self.im_size + 8)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
        self.bias = 0

    def forward(self, z):
        # z (bs, 32)
        bs, _ = z.shape
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size + 8, self.im_size + 8)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
                       self.y_grid.expand(bs, -1, -1, -1), z), dim=1)
        # x (bs, 32, image_h, image_w)
        x = self.conv1(x)
        x = self.celu(x)
        # x = self.bn1(x)
        x = self.conv2(x)
        x = self.celu(x)
        # x = self.bn2(x)
        x = self.conv3(x)
        x = self.celu(x)
        # x = self.bn3(x)
        x = self.conv4(x)
        x = self.celu(x)
        # x = self.bn4(x)
        img = self.conv5_img(x)
        img = .5 + 0.55 * torch.tanh(img + self.bias)
        logitmask = self.conv5_mask(x)

        return img, logitmask


inputs = torch.randn([10,3,128,128])
masks = torch.ones([10,1,128,128])
encoder_net = encoder(32)

outputs = encoder_net(inputs,masks)

print(outputs[0].shape)
print(outputs[1].shape)