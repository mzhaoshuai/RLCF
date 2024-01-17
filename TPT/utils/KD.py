# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_distill_loss(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    """
    vanilla KD, KLDiv between teacher and student
    """
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean").mean()
    loss_kd = loss_kd * T_stu * T_stu

    return loss_kd


def kd_distill_loss_v2(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    """
    vanilla KD, KLDiv between teacher and student, only the gradient related part
    """
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    # kl_div = -p log q
    loss_kd = - torch.sum(pred_teacher * log_pred_student, dim=1).mean()
    loss_kd = loss_kd * T_stu * T_stu

    return loss_kd
