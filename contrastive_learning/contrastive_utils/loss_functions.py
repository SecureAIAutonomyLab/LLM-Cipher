import torch
import torch.nn.functional as F
import torch.nn as nn


class EuclideanContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(EuclideanContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_similarity = F.cosine_similarity(output1, output2)

        loss_contrastive = torch.mean((1-label) * 0.5 * (1 - cosine_similarity) +
                                      label * 0.5 * torch.clamp(cosine_similarity + self.margin, min=0.0))

        return loss_contrastive


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, out_1, out_2, *args):
        for vec1, vec2 in zip(out_1, out_2):
            # print("Vector from out_1:")
            # print(vec1)
            # print("\nVector from out_2:")
            # print(vec2)

            cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
            # print("\nCosine Similarity:", cosine_sim.item()) 
            if cosine_sim <= 0.9:
                print(cosine_sim)
                input("Press Enter to view the next pair...")

        # The rest of the original forward method goes here
        out = torch.cat([out_1, out_2], dim=0)
        n_samples = len(out)

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)

        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        epsilon = 1e-10
        loss = -torch.log(pos / (neg + epsilon)).mean()
        return loss

