import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomDistLoss(nn.Module):
    def __init__(self, device, config, num_classes=1):
        super(CustomDistLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.alpha = config['alpha']  # used to reduce cross entropy loss
        self.beta = config['beta']  # used to reduce regression loss
        # weight between teacher and training loss
        self.mew = config['mew']
        self.nu = config['nu']
        self.margin = config['margin']

    # calculates the binary cross entropy loss between x and y
    # given that x and y have dimensions [batch_size, 1, 200, 175]
    def cross_entropy(self, x, y):
        return F.binary_cross_entropy(input=x, target=y, reduction='mean')

    def teacher_reg_loss(self, loc_preds, loc_teacher_preds, loc_targets):

        student_squared_diff = torch.square(loc_preds - loc_targets)
        teacher_squared_diff = torch.square(loc_teacher_preds - loc_targets)

        # to determine whether to keep the student squared difference
        check_margin = torch.gt(student_squared_diff +
                                self.margin, teacher_squared_diff)

        reg_error = check_margin * student_squared_diff

        return reg_error

    def forward(self, preds, teacher_preds, targets):

        batch_size = targets.size(0)

        cls_targets, loc_targets = targets.split([1, 6], dim=1)

        if preds.size(1) == 7:
            cls_preds, loc_preds = preds.split([1, 6], dim=1)
            cls_teacher_preds, loc_teacher_preds = teacher_preds.split([
                1, 6], dim=1)
        elif preds.size(1) == 15:
            cls_preds, loc_preds, _ = preds.split([1, 6, 8], dim=1)
            cls_teacher_preds, loc_teacher_preds = teacher_preds.split([
                                                                       1, 6], dim=1)

        # calculating cross entropy with respect to the training data
        cls_loss_training = self.cross_entropy(
            cls_preds, cls_targets)
        # calculating cross entropy with respect to the teacher data
        cls_loss_teacher = self.cross_entropy(
            cls_preds, cls_teacher_preds)

        # adding the two cross entropies together - factor alpha to lower cross entropy loss
        cls_loss = (self.mew * cls_loss_training +
                    (1 - self.mew) * cls_loss_teacher) * self.alpha

        # only evaluating regression on points where the targets are non-zero (for classification)
        # this is because regression can be very wrong and contradictory to the ground truth
        pos_pixels_targets = cls_targets.sum()

        # calculating regression loss with respect to target values
        # multiply by beta, regression reduction factor
        loc_loss_training = F.smooth_l1_loss(
            cls_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels_targets * self.beta

        # calculating regression loss with respect to teacher values
        loc_loss_teacher = self.teacher_reg_loss(
            cls_targets * loc_preds, cls_targets * loc_teacher_preds, loc_targets)

        # taking mean over elements summed over (all elements where the classification targets are 1)
        # multiply by beta, regression reduction factor
        loc_loss_teacher = torch.sum(
            loc_loss_teacher) / pos_pixels_targets * self.beta

        loc_loss = loc_loss_training + loc_loss_teacher * self.nu

        loss = cls_loss + loc_loss
        cls = cls_loss_training.item() * self.mew * self.alpha
        loc = loc_loss_training.item() * (1 - self.mew) * self.alpha
        cls_teacher = cls_loss_teacher.item()
        loc_teacher = loc_loss_teacher.item() * self.nu

        return loss, cls, loc, cls_teacher, loc_teacher
