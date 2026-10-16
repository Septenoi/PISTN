import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels,
                              kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x, h_prev, c_prev):
        if h_prev is None:
            size_h = [x.size(0), self.hidden_channels, x.size(2), x.size(3)]
            h_prev = torch.zeros(size_h, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(size_h, device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        ci, cf, cg, co = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        g = torch.tanh(cg)
        o = torch.sigmoid(co)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)

    def forward(self, x_seq, h0=None, c0=None):
        B, T, C, H, W = x_seq.shape
        hs, cs = [], []
        h, c = h0, c0
        for t in range(T):
            x_t = x_seq[:, t]
            h, c = self.cell(x_t, h, c)
            hs.append(h)
        hs = torch.stack(hs, dim=1)
        return hs, (h, c)


def conv_block2d(in_ch, out_ch, ks=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=ks, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvLSTM_EncDec(nn.Module):
    def __init__(self, shape_in, base_channels=64, kernel_size=3, dropout=0.5):
        super().__init__()
        T, C, H, W = shape_in
        self.T_in = T

        self.enc_ch1 = base_channels
        self.enc_ch2 = base_channels * 2
        self.enc_ch3 = base_channels * 4
        self.bottleneck_ch = base_channels * 8

        self.enc1 = conv_block2d(C, self.enc_ch1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block2d(self.enc_ch1, self.enc_ch2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block2d(self.enc_ch2, self.enc_ch3)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = conv_block2d(self.enc_ch3, self.bottleneck_ch)

        self.convlstm = ConvLSTM(self.bottleneck_ch, self.bottleneck_ch, kernel_size=kernel_size)

        self.up3 = nn.ConvTranspose2d(self.bottleneck_ch, self.enc_ch3, kernel_size=2, stride=2)
        self.dec3 = conv_block2d(self.enc_ch3 * 2, self.enc_ch3)

        self.up2 = nn.ConvTranspose2d(self.enc_ch3, self.enc_ch2, kernel_size=2, stride=2)
        self.dec2 = conv_block2d(self.enc_ch2 * 2, self.enc_ch2)

        self.up1 = nn.ConvTranspose2d(self.enc_ch2, self.enc_ch1, kernel_size=2, stride=2)
        self.dec1 = conv_block2d(self.enc_ch1 * 2, self.enc_ch1)

        self.out_conv = nn.Conv2d(self.enc_ch1, 1, kernel_size=1)
        self.alpha = 0.03
        self.skip_conv = nn.Conv2d(C, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def encode_per_frame(self, x):
        B, T, C, H, W = x.shape

        x_flat = x.view(B * T, C, H, W)
        e1 = self.enc1(x_flat)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        bridge = self.bridge(p3)

        def unflatten(tensor, channels, h, w):
            return tensor.view(B, T, channels, h, w)

        H1, W1 = H, W
        H2, W2 = H // 2, W // 2
        H3, W3 = H // 4, W // 4
        Hb, Wb = H // 8, W // 8

        e1_seq = unflatten(e1, self.enc_ch1, H1, W1)
        e2_seq = unflatten(e2, self.enc_ch2, H2, W2)
        e3_seq = unflatten(e3, self.enc_ch3, H3, W3)
        bridge_seq = unflatten(bridge, self.bottleneck_ch, Hb, Wb)

        return e1_seq, e2_seq, e3_seq, bridge_seq

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert T == self.T_in, f"Expected T={self.T_in}, got {T}"

        e1_seq, e2_seq, e3_seq, bridge_seq = self.encode_per_frame(x)

        hs_bridge, (h_last, c_last) = self.convlstm(bridge_seq)

        e1_last = e1_seq[:, -1]
        e2_last = e2_seq[:, -1]
        e3_last = e3_seq[:, -1]

        outputs = []
        for t in range(T):
            h_t = hs_bridge[:, t]

            d3 = self.up3(h_t)
            d3 = torch.cat([d3, e3_last], dim=1)
            d3 = self.dec3(d3)
            d3 = self.dropout(d3)

            d2 = self.up2(d3)
            d2 = torch.cat([d2, e2_last], dim=1)
            d2 = self.dec2(d2)
            d2 = self.dropout(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1_last], dim=1)
            d1 = self.dec1(d1)

            pred2d = self.out_conv(d1)

            skip = self.skip_conv(x[:, -1])
            pred2d = self.alpha * pred2d + skip

            outputs.append(pred2d)

        out = torch.stack(outputs, dim=1)
        return out


if __name__ == "__main__":
    shape_in = (12, 5, 80, 80)
    model = ConvLSTM_EncDec(shape_in, base_channels=32)
    x = torch.randn(2, *shape_in)
    y = model(x)
    print("Output shape:", y.shape)
