import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, self_att):
        super(Attention, self).__init__()
        self.self_att = self_att
        self.L = 512
        self.D = 128
        self.K = 10

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        if self.self_att:
            self.self_att = SelfAttention(self.L)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.Tensor(x)

        if self.self_att:
            H, self_attention, gamma, gamma_kernel = self.self_att(x)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M.flatten().unsqueeze(0))
        pred = torch.argmax(Y_prob)

        return Y_prob, pred, A

    # AUXILIARY METHODS
    def proper_values(self, X, Y):
        _, pred, _ = self.forward(X)
        proper = pred == torch.full_like(pred, Y)

        return proper

    def calculate_objective(self, X, Y, entropy):
        Y_prob, _, A = self.forward(X)
        loss = entropy(Y_prob[0], Y)

        return loss


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.zeros(1)).cuda())
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)).cuda())

    def forward(self, x):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att
