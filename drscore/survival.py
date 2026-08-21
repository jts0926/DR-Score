from __future__ import annotations

import torch


def cox_negative_partial_log_likelihood(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    model: torch.nn.Module | None = None,
    l2_coefficient: float = 0.0,
) -> torch.Tensor:
    """Breslow Cox loss with the same minibatch risk-set definition as training."""
    risk = risk.reshape(-1)
    time = time.reshape(-1).to(risk.device)
    event = event.reshape(-1).to(risk.device).float()
    if event.sum() == 0:
        loss = risk.sum() * 0.0
    else:
        risk_set = time.unsqueeze(0) >= time.unsqueeze(1)
        log_risk = torch.logsumexp(
            risk.unsqueeze(0).masked_fill(~risk_set, -torch.inf), dim=1
        )
        loss = -((risk - log_risk) * event).sum() / event.sum()

    if model is not None and l2_coefficient > 0:
        penalty = sum(
            torch.linalg.vector_norm(parameter)
            for name, parameter in model.named_parameters()
            if "weight" in name
        )
        loss = loss + float(l2_coefficient) * penalty
    return loss
