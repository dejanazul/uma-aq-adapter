import logging
import math

import torch
from fairchem.core.models.uma.nn.mole import MOLE, MOLEDGL

from adapt_freeze import freeze_to_task


class _TaskLoRA:
    def forward(self, x):
        out = super().forward(x)
        g = self.global_mole_tensors
        rows = torch.repeat_interleave(g.lora_gate, g.mole_sizes.to(g.lora_gate.device))
        rows = rows[g.ac_start_idx : g.ac_start_idx + x.shape[0]]
        lora = (x @ self.lora_A.t()) @ self.lora_B.t() * self.lora_scale
        return out + lora * rows.view(-1, *([1] * (x.ndim - 1))).to(lora.dtype)


class TaskLoRAMOLE(_TaskLoRA, MOLE):
    pass


class TaskLoRAMOLEDGL(_TaskLoRA, MOLEDGL):
    pass


def add_task_lora(model, task, rank=8, alpha=8.0):
    backbone = model.backbone
    n = 0
    for module in backbone.modules():
        if type(module) in (MOLE, MOLEDGL):
            module.__class__ = TaskLoRAMOLE if type(module) is MOLE else TaskLoRAMOLEDGL
            dev = module.weights.device
            module.lora_A = torch.nn.Parameter(torch.empty(rank, module.in_features, device=dev))
            torch.nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            module.lora_B = torch.nn.Parameter(torch.zeros(module.out_features, rank, device=dev))
            module.lora_scale = alpha / rank
            n += module.lora_A.numel() + module.lora_B.numel()

    def set_gate(mod, args, kwargs):
        data = args[0] if args else kwargs['data']
        names = data['dataset']
        mod.global_mole_tensors.lora_gate = torch.tensor(
            [d == task for d in names], dtype=torch.float32, device=data['pos'].device
        )

    backbone.register_forward_pre_hook(set_gate, with_kwargs=True)
    logging.warning(f'task-gated LoRA on {task}: rank {rank}, alpha {alpha}, {n} adapter parameters')
    return model


def initialize_task_lora_model(checkpoint_location, overrides=None, heads=None, task='aqcat25', rank=8, alpha=8.0):
    from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model

    model = freeze_to_task(initialize_finetuning_model(checkpoint_location, overrides, heads), task)
    return add_task_lora(model, task, rank, alpha)


if __name__ == '__main__':
    from fairchem.core.models.uma.nn.mole import MOLEGlobals

    torch.manual_seed(0)
    g = MOLEGlobals(expert_mixing_coefficients=torch.softmax(torch.randn(3, 4), 1), mole_sizes=torch.tensor([2, 3, 2]))
    layer = MOLE(num_experts=4, in_features=5, out_features=6, global_mole_tensors=g, bias=True)

    class BB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer, self.global_mole_tensors = layer, g

        def forward(self, data):
            return self.layer(data['x'])

    x = torch.randn(7, 5)
    ref = layer(x).detach()
    toy = torch.nn.Module()
    toy.backbone = bb = BB()
    add_task_lora(toy, 'task', rank=2, alpha=2.0)
    data = {'x': x, 'dataset': ['task', 'other', 'task'], 'pos': torch.zeros(1)}
    assert torch.equal(bb(data), ref), 'B = 0 must give the original output'
    with torch.no_grad():
        layer.lora_B.normal_()
    out = bb(data)
    assert torch.equal(out[2:5], ref[2:5]), 'rows of the non-task system changed'
    assert not torch.allclose(out[:2], ref[:2]) and not torch.allclose(out[5:], ref[5:]), 'task rows did not change'
    out.sum().backward()
    assert layer.lora_A.grad is not None and layer.lora_B.grad is not None
    print('ok: adapters change only task rows; non-task rows bit-identical')
