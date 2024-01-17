# coding=utf-8
"""
Reference

[1] https://github.com/forjiuzhou/Spherical-Knowledge-Distillation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def atkd_distill_loss(logits_student, logits_teacher,
						temperature=4.0, multiplier=2.0, eps=1e-5, version=1):
	"""Adaptive Temperature KD
	Args: use dfault params T=4.0 and multiplier=2
	"""

	if version == 1:
		# normalize
		with torch.no_grad():
			s_mu, s_std = torch.std_mean(logits_student, dim=-1, unbiased=False, keepdim=True)
			t_mu, t_std = torch.std_mean(logits_teacher, dim=-1, unbiased=False, keepdim=True)
		norm_s_logit = (logits_student - s_mu) / (s_std + eps) * multiplier
		norm_t_logit = (logits_teacher - t_mu) / (t_std + eps) * multiplier

		# KLDiv between normalized logits
		loss_kd = F.kl_div(F.log_softmax(norm_s_logit, dim=-1),
								F.softmax(norm_t_logit, dim=-1),
								reduction='none') * s_std * s_std
		loss_kd = loss_kd.sum(-1).mean()

	elif version == 2:
		C = logits_student.shape[-1]
		logits_s = F.layer_norm(logits_student, (C,), weight=None, bias=None, eps=eps) * multiplier
		logits_t = F.layer_norm(logits_teacher, (C,), weight=None, bias=None, eps=eps) * multiplier

		# KLDiv between normalized logits
		loss_kd = F.kl_div(F.log_softmax(logits_s / temperature, dim=-1),
								F.softmax(logits_t / temperature, dim=-1),
								reduction='batchmean') * temperature * temperature

	return loss_kd
