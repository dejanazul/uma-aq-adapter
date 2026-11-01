import logging

import torch


def freeze_to_task(model, task):
    for p in model.parameters():
        p.requires_grad_(False)
    model.backbone.dataset_embedding.dataset_emb_dict[task].weight.requires_grad_(True)
    head = model.output_heads['energyandforcehead']
    idx = head.dataset_name_to_exp[task]

    def keep_slot(g, i=idx):
        out = torch.zeros_like(g)
        out[i] = g[i]
        return out

    for name, p in head.named_parameters():
        if name.split('.')[-1] == 'weights':
            p.requires_grad_(True)
            p.register_hook(keep_slot)
    n = sum(
        p[idx].numel() if n_.startswith('output_heads') and n_.split('.')[-1] == 'weights' else p.numel()
        for n_, p in model.named_parameters()
        if p.requires_grad
    )
    logging.warning(f'task-only fine-tuning on {task} (head slot {idx}): {n} trainable parameters')
    return model


def initialize_task_only_model(checkpoint_location, overrides=None, heads=None, task='aqcat25'):
    from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model

    return freeze_to_task(initialize_finetuning_model(checkpoint_location, overrides, heads), task)


if __name__ == '__main__':

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.randn(3, 2, 2))
            self.bias = torch.nn.Parameter(torch.randn(2))
            self.dataset_name_to_exp = {'a': 0, 'task': 1, 'b': 2}

        def forward(self, x):
            return (self.weights.sum(0) @ x).sum() + self.bias.sum()

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.dataset_embedding = torch.nn.Module()
            self.backbone.dataset_embedding.dataset_emb_dict = torch.nn.ModuleDict(
                {k: torch.nn.Embedding(1, 2) for k in ('a', 'task', 'b')}
            )
            self.backbone.shared = torch.nn.Linear(2, 2)
            self.output_heads = torch.nn.ModuleDict({'energyandforcehead': Head()})

    m = freeze_to_task(Toy(), 'task')
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    opt = torch.optim.AdamW(m.parameters(), lr=0.1, weight_decay=0)
    for _ in range(3):
        e = m.backbone.dataset_embedding.dataset_emb_dict
        x = m.backbone.shared(e['task'].weight[0] + e['a'].weight[0])
        loss = m.output_heads['energyandforcehead'](x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = m.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed == {
        'backbone.dataset_embedding.dataset_emb_dict.task.weight',
        'output_heads.energyandforcehead.weights',
    }, changed
    w0, w1 = before['output_heads.energyandforcehead.weights'], after['output_heads.energyandforcehead.weights']
    assert torch.equal(w0[0], w1[0]) and torch.equal(w0[2], w1[2]) and not torch.equal(w0[1], w1[1])
    print('ok: only the task embedding and head slot changed')
