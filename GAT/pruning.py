import math
import numpy as np
import torch
from utils import get_W, compute_edge_score_from_edge_index


class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.accumulation_n
        else:
            self.dense_grad = None
        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class FastScheduler:
    """
    FastGLT with DIFFERENT sparsity targets for weights vs edges + GRADUAL schedule.
      - weight_sparsity, edge_sparsity: target fractions of zeros (e.g., 0.9 => 90% zeros)
      - init_*_sparsity: initial fractions of zeros at t=0
      - linear ramp from initial to target over [0, T_end]
      - per-step cosine 'swap' fraction; plus extra prune to close gap to current goal
      - optional warmup: small extra prune in early intervals via warmup_k
    """
    def __init__(
        self,
        model,
        optimizer,
        T_end=400,
        delta=100,
        alpha=0.3,
        accumulation_n=1,
        sparsity_distribution='uniform',
        static_topo=False,
        ignore_linear_layers=False,
        ignore_parameters=False,
        state_dict=None,
        pretrain=False,
        # targets (fractions of zeros)
        weight_sparsity=0.90,
        edge_sparsity=0.10,
        # initial sparsities (fractions of zeros)
        init_weight_sparsity=0.0,
        init_edge_sparsity=0.0,
        # warmup
        beta=1.0,
        warmup_steps=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.W, self._linear_layers_mask, self._parameters_mask = get_W(model, return_linear_layers_mask=True)
        _create_step_wrapper(self, optimizer)

        self.delta_T = int(delta)
        self.alpha = float(alpha)
        self.T_end = int(T_end)
        self.accumulation_n = int(accumulation_n)
        self.static_topo = static_topo
        self.ignore_linear_layers = ignore_linear_layers
        self.ignore_parameters = ignore_parameters
        self.sparsity_distribution = sparsity_distribution

        self.N = [torch.numel(w) for w in self.W]

        # Per-tensor target and current sparsity
        self.S_target = []
        self.S_curr = []
        for is_linear, is_para in zip(self._linear_layers_mask, self._parameters_mask):
            if is_linear and self.ignore_linear_layers:
                self.S_target.append(0.0); self.S_curr.append(0.0)
            elif is_para and self.ignore_parameters:
                self.S_target.append(0.0); self.S_curr.append(0.0)
            else:
                if is_para:
                    self.S_target.append(float(edge_sparsity))
                    self.S_curr.append(float(init_edge_sparsity))
                else:
                    self.S_target.append(float(weight_sparsity))
                    self.S_curr.append(float(init_weight_sparsity))

        # Warmup
        self.warmup_steps = int(warmup_steps)
        self.warmup_k = math.pow(beta, -1.0 / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        #print("self.warmup_k", self.warmup_k)
        if not pretrain:
                self._init_sparsity()
        else:
                self.fine_tuning_sparsity()
        self.step = 0
        self.FastGLT_steps = 0
        #self.backward_masks = None

        # Initialize to initial sparsity
        if state_dict is not None:
            self.load_state_dict(state_dict)
            if self.backward_masks is None:
                self._init_sparsity(self.S_curr)
            self.apply_mask_to_weights()
        else:
            self._init_sparsity(self.S_curr) if not pretrain else self.fine_tuning_sparsity()

        # Hooks
        self.backward_hook_objects = []
        for i, (w, S_t) in enumerate(zip(self.W, self.S_target)):
            if S_t <= 0:
                self.backward_hook_objects.append(None)
                continue
            if getattr(w, '_has_FastGLT_backward_hook', False):
                raise Exception('This model already has been registered to a FastScheduler.')
            hook = IndexMaskHook(i, self)
            self.backward_hook_objects.append(hook)
            w.register_hook(hook)
            setattr(w, '_has_FastGLT_backward_hook', True)

        if pretrain:
            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients()

        assert 0 < self.accumulation_n < self.delta_T
        assert self.sparsity_distribution in ('uniform', )

    def __str__(self):
        s = 'FastScheduler(\n'
        s += f'layers={len(self.N)},\n'
        total_params = total_nonzero = 0
        N_str, S_str = '[', '['
        for N, mask, W, is_linear in zip(self.N, self.backward_masks, self.W, self._linear_layers_mask):
            actual_zero = 0 if mask is None else torch.sum(W[mask == 0] == 0).item()
            N_str += f'{N-actual_zero}/{N}, '
            S_str += f'{(N-actual_zero)/N*100:.2f}%, '
            total_params += N
            total_nonzero += N - actual_zero
        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'
        s += f'nonzero_params={N_str},\n'
        s += f'nonzero_percentages={S_str},\n'
        s += f'total_nonzero_params={total_nonzero}/{total_params} ({total_nonzero/total_params*100:.2f}%),\n'
        s += f'step={self.step}, num_FastGLT_steps={self.FastGLT_steps},\n'
        s += f'ignoring_linear_layers={self.ignore_linear_layers}, ignoring_parameters={self.ignore_parameters},\n'
        s += f'warmup_steps={self.warmup_steps}\n'
        return s + ')'

    # ---------- init helpers ----------

    @torch.no_grad()
    def _init_sparsity(self, S_init_list):
        self.backward_masks = []
        for w, n, S_init in zip(self.W, self.N, S_init_list):
            if S_init <= 0:
                mask = torch.ones_like(w, dtype=torch.bool, device=w.device)
                self.backward_masks.append(mask)
                continue
            s = int(round(S_init * n))  # zeros
            perm = torch.randperm(n, device=w.device)[:s]
            flat = torch.ones(n, device=w.device)
            flat[perm] = 0
            mask = flat.reshape(w.shape).bool()
            w *= mask
            self.backward_masks.append(mask)

    @torch.no_grad()
    def fine_tuning_sparsity(self):
        self.backward_masks = []
        for w, n, S_init in zip(self.W, self.N, self.S_curr):
            if S_init <= 0:
                mask = torch.ones_like(w, dtype=torch.bool, device=w.device)
                self.backward_masks.append(mask); continue
            s = int(round(S_init * n))
            n_keep = n - s
            score = torch.abs(w).view(-1)
            _, order = torch.topk(score, k=n)
            new_vals = torch.where(
                torch.arange(n, device=w.device) < n_keep,
                torch.ones_like(order),
                torch.zeros_like(order),
            )
            flat_mask = new_vals.scatter(0, order, new_vals)
            mask = flat_mask.reshape(w.shape).bool()
            w *= mask
            self.backward_masks.append(mask)

    # ---------- scheduling ----------

    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if (self.step % self.delta_T) == 0 and self.step <= self.T_end:
            self._FastGLT_step()
            self.FastGLT_steps += 1
            return False
        if self.step > self.T_end:
            if self.step == self.T_end + 1:
                print("reset")
                self.reset_momentum()
                self.apply_mask_to_weights()
                self.apply_mask_to_gradients()
            return False
        return True

    def check_if_backward_hook_should_accumulate_grad(self):
        if self.step >= self.T_end:
            return False
        steps_til = self.delta_T - (self.step % self.delta_T)
        return steps_til <= self.accumulation_n

    def cosine_annealing(self):
        #return self.alpha / 2.0 * (1.0 + np.cos((self.step * np.pi) / max(1, self.T_end)))
        print("*********", self.alpha * (1 - self.step / self.T_end) ** 1)
        return self.alpha * (1 - self.step / self.T_end) ** 1

    def _scheduled_goal_sparsity(self, l):
        """Linear ramp from S_curr[0] to S_target over [0, T_end]."""
        S_tgt = self.S_target[l]
        if self.T_end <= 0 or self.step >= self.T_end:
            return S_tgt
        frac = max(0.0, min(1.0, self.step / float(self.T_end)))
        S0 = self.S_curr[l]
        return S0 * (1 - frac) + S_tgt * frac

    # ---------- main GLT step ----------

    @torch.no_grad()
    def _FastGLT_step(self):
        drop_fraction = self.cosine_annealing()
        #print("drop_fraction",drop_fraction)

        for l, (w, is_linear, is_para) in enumerate(zip(self.W, self._linear_layers_mask, self._parameters_mask)):
            if self.S_target[l] <= 0 and self.S_curr[l] <= 0:
                continue

            current_mask = self.backward_masks[l]
            n_total = self.N[l]
            n_ones = int(torch.sum(current_mask).item())
            #print("n_ones",n_ones)

            # swap amount from cosine schedule
            n_swap = int(max(0, min(n_ones, int(n_ones * drop_fraction))))

            # how many ones we *should* have by now
            S_goal = self._scheduled_goal_sparsity(l)            # fraction zeros
            n_goal_ones = int(round((1.0 - S_goal) * n_total))

            # net prune needed to reach the goal this interval
            net_prune_needed = max(0, n_ones - n_goal_ones)

            # base prune/grow
            n_prune = n_swap + net_prune_needed
            n_grow = n_swap

            # warmup: add a tiny extra prune early + decay S_curr multiplicatively
            if self.FastGLT_steps < self.warmup_steps:
                extra = int((self.warmup_k - 1.0) * self.S_target[l] * n_total)
                #extra = ((self.warmup_k - 1.0) * self.S_target[l] * n_total)
                #print("extra:", extra)
                n_prune = min(n_ones, n_prune + max(0, extra))
                self.S_curr[l] = min(1.0, self.S_curr[l] * self.warmup_k)

            n_prune = min(n_prune, n_ones)
            n_keep = max(0, n_ones - n_prune)

            if n_prune == 0 and n_grow == 0:
                continue

            # ----- drop: keep largest by |w| -----
            score_drop = torch.abs(w).view(-1)
            _, order = torch.topk(score_drop, k=n_total)  # descending
            new_vals = torch.where(
                torch.arange(n_total, device=w.device) < n_keep,
                torch.ones_like(order),
                torch.zeros_like(order),
            )
            mask_keep_flat = new_vals.scatter(0, order, new_vals)  # 1 keep, 0 drop

            # ----- grow: choose among zeros -----
            score_grad = (torch.abs(self.backward_hook_objects[l].dense_grad)
                          if (self.backward_hook_objects[l] is not None and
                              self.backward_hook_objects[l].dense_grad is not None)
                          else torch.zeros_like(w))

            if is_para:
                edge_score = torch.abs(
                    compute_edge_score_from_edge_index(
                        self.model.graph.edges(),
                        self.model.num_nodes,
                        current_mask.squeeze(),
                        device=self.model.device
                    )
                ).reshape(-1, 1)
                score_grow = edge_score + score_grad
            else:
                score_grow = score_grad

            flat_grow = score_grow.view(-1)
            # prevent kept-ones from being grown
            lifted = torch.where(
                mask_keep_flat.bool() == 1,
                torch.ones_like(flat_grow) * (torch.min(flat_grow) - 1),
                flat_grow
            )

            _, order = torch.topk(lifted, k=n_total)
            new_vals = torch.where(
                torch.arange(n_total, device=w.device) < n_grow,
                torch.ones_like(order),
                torch.zeros_like(order),
            )
            mask_grow_flat = new_vals.scatter(0, order, new_vals)

            # apply updates
            mask_grow = mask_grow_flat.reshape(current_mask.shape)
            # init new connections
            if is_para:
                val = torch.mean(w[current_mask]) if torch.any(current_mask) else 0.0
                grow_tensor = torch.ones_like(w) * val
            else:
                grow_tensor = torch.zeros_like(w)

            new_connections = ((mask_grow == 1) & (current_mask == 0))
            w.data = torch.where(new_connections.to(w.device), grow_tensor, w)

            # combine masks
            new_mask = (mask_keep_flat + mask_grow_flat).reshape(current_mask.shape).bool()
            current_mask.data = new_mask
            self.backward_masks[l] = current_mask

            # update achieved S_curr for reporting
            self.S_curr[l] = 1.0 - (float(torch.sum(current_mask).item()) / float(n_total))

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients()

    # ---------- mask/grad utils ----------

    @torch.no_grad()
    def reset_momentum(self):
        for w, mask in zip(self.W, self.backward_masks):
            if mask is None: continue
            st = self.optimizer.state[w]
            if 'momentum_buffer' in st:
                st['momentum_buffer'] *= mask

    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask in zip(self.W, self.backward_masks):
            if mask is None: continue
            w *= mask

    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask in zip(self.W, self.backward_masks):
            if mask is None: continue
            if w.grad is not None:
                w.grad *= mask

    @torch.no_grad()
    def getS(self):
        total_weight = total_weight_zero = 0
        total_paras = total_paras_zero = 0
        for N, mask, W, is_linear, is_para in zip(self.N, self.backward_masks, self.W,
                                                   self._linear_layers_mask, self._parameters_mask):
            if mask is None:
                continue
            actual_zero = torch.sum(W[mask == 0] == 0).item()
            if is_para:
                total_paras += N
                total_paras_zero += actual_zero
            if is_linear:
                total_weight += N
                total_weight_zero += actual_zero
        spar_weight = (total_weight_zero / total_weight) if total_weight > 0 else 0.0
        spar_adj = (total_paras_zero / total_paras) if total_paras > 0 else 0.0
        return spar_weight * 100.0, spar_adj * 100.0
