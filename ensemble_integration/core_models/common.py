"""Shared length validation, masking and packed BiGRU, independent of legacy models."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def validate_sequence(x, lengths, valid_frame_mask):
    if x.ndim < 3 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Expected a nonempty [B,T,...] sequence")
    b, t = x.shape[:2]
    if lengths.shape != (b,) or lengths.dtype != torch.int64:
        raise ValueError("Expected int64 lengths [B]")
    if valid_frame_mask.shape != (b, t) or valid_frame_mask.dtype != torch.bool or valid_frame_mask.device != x.device:
        raise ValueError("Expected bool valid-frame mask [B,T] on the input device")
    if bool(torch.any(lengths < 1)) or bool(torch.any(lengths > t)):
        raise ValueError("Zero-length and overlong sequences are invalid")
    expected = torch.arange(t, device=x.device)[None, :] < lengths.to(x.device)[:, None]
    if not torch.equal(valid_frame_mask, expected):
        raise ValueError("Mask must describe exactly the right-padded true lengths")
    if not bool(torch.isfinite(x[valid_frame_mask]).all()):
        raise ValueError("Real frames/features contain nonfinite values")


def masked_mean(x, valid_frame_mask):
    valid = valid_frame_mask.unsqueeze(-1)
    return torch.where(valid, x, 0).sum(dim=1) / valid.sum(dim=1).to(x.dtype)


class PackedBiGRU(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(512, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, features, lengths, valid_frame_mask):
        validate_sequence(features, lengths, valid_frame_mask)
        packed = pack_padded_sequence(features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        output, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        return output
