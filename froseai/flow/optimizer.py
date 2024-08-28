import torch
from torch.optim.optimizer import required
from ..opt_frame import FroseAiOptFrame


class FedAvg(FroseAiOptFrame):
    def __init__(self, parameters, client_id: int, config_path: str,
                 lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, train_data_num=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, initial_lr=lr)
        super(FedAvg, self).__init__(parameters, defaults, client_id, config_path)

        self._train_data_num = train_data_num

    def __setstate__(self, state):
        super(FedAvg, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss

    def snd_params(self):
        return {"sample_num": self._train_data_num}

    def rcv_params(self, others):
        pass
