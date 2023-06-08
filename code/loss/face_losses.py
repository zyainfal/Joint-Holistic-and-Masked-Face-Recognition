import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class MagFace(nn.Module):
    """Implementation for "MagFace: A Universal Representation for Face Recognition and Quality Assessment"
    """
    def __init__(self, margin_am=0.0, scale=64, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8, lamda=35):#35
        super(MagFace, self).__init__()
        self.margin_am = margin_am
        self.scale = scale        
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.lamda = lamda

    def calc_margin(self, x):
        margin = (self.u_margin-self.l_margin) / (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def forward(self, logits, ada_margin, labels):
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        cos_theta = logits.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        min_cos_theta = torch.cos(math.pi - ada_margin)        
        cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        #index.scatter_(1, labels.data.view(-1, 1), 1)
        index.data[torch.arange(0, logits.size(0)), labels.data.view(-1)] = 1.0
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output

class CurricularFace(nn.Module):
    """Implementation for "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition".
    """
    def __init__(self, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.register_buffer('t', torch.zeros(1))

    def forward(self, logits, labels):
        cos_theta = logits.clamp(-1, 1)  # for numerical stability
        
        idx = labels.clone().squeeze()
        
        target_logit = cos_theta[torch.arange(0, logits.size(0)), idx].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        #cos_theta.scatter_(1, labels.long(), final_target_logit)
        cos_theta[torch.arange(0, logits.size(0)), idx] = final_target_logit.view(-1)
        output = cos_theta * self.s
        return output

class CosFace(nn.Module):
    def __init__(self, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        cosine = logits.clamp(-1, 1)
        index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, labels[index], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret

class ArcFace(nn.Module):
    """ Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.clamp(-1, 1)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

class NpairLoss(torch.nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, s=64., m=0.5):
        super(NpairLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, anchor, positive, target, arclogit=False, mean_logit=True):
        #batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        if mean_logit:
            embedding_mean = (anchor + positive) / 2
            logit_mean = torch.matmul(embedding_mean, torch.transpose(embedding_mean, 0, 1))
            logit = target.detach() * (logit - logit_mean) + logit_mean
        # print(logit)
        logit = self.arclogits(logit) if arclogit else (logit * self.s)
        loss_ce = cross_entropy(logit, target)
        #l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce #+ self.l2_reg*l2_loss*0.25
        return loss

    def arclogits(self, logits):
        nB = logits.size(0)
        cos_theta = logits.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, idx_] = cos_theta_m[idx_, idx_]
        return output * self.s  # scale up in order to make softmax work, first introduced in normface

class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s=64, 
                 m1=1,
                 m2=0,
                 m3=0.4,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        
        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise        

        return logits
"""

class MagLinear(torch.nn.Module):
    def __init__(self, scale=64.0, easy_margin=True):
        super(MagLinear, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin

    def forward(self, x, w, m, l_a, u_a):
 
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(l_a, u_a)
        ada_margin = m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(w, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm)
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        return [cos_theta, cos_theta_m], x_norm


class MagLoss(torch.nn.Module):
    def __init__(self, l_a, u_a, l_margin, u_margin, scale=64.0):
        super(MagLoss, self).__init__()
        self.l_a = l_a
        self.u_a = u_a
        self.scale = scale
        self.cut_off = np.cos(np.pi/2-l_margin)
        self.large_value = 1 << 10

    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def forward(self, input, target, x_norm):
        loss_g = self.calc_loss_G(x_norm)

        cos_theta, cos_theta_m = input
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')
        return loss.mean(), loss_g, one_hot

class ArcLoss(torch.nn.Module):
    def __init__(self, in_features, out_features, margin=0.5, scale=64.0, smooth=0.0):
        super(ArcLoss, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.mm = math.sin(math.pi - margin) * margin#self.sin_m * margin #
        self.threshold = math.cos(math.pi - margin)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=smooth)

    def forward(self, x, target, is_arc=True):
        if is_arc:
            # norm the weight
            weight_norm = F.normalize(self.weight, dim=0)
            cos_theta = torch.mm(F.normalize(x), weight_norm)
            cos_theta = cos_theta.clamp(-1, 1)
            sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(1e-6, 1))
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
            
            cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
            # multiply the scale in advance
            cos_theta_m = self.scale * cos_theta_m
            cos_theta = self.scale * cos_theta
            
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, target.view(-1, 1), 1.0)
            output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        else:
            output = torch.mm(x, self.weight)
        loss = self.criterion(output, target)
        return loss#.mean()
"""