"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from logger import logkv_mean, log

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
        gradient_clip_val=None,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.gradient_clip_val = gradient_clip_val

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt):
        # 1) 拷贝梯度到 master_params
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)

        # 2) 梯度裁剪
        if self.gradient_clip_val is not None:
            th.nn.utils.clip_grad_norm_(self.master_params, self.gradient_clip_val)

        # 3) 计算梯度范数，检查溢出
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)

            # >>> 在这里插入一次对模型关键参数的打印，辅助定位是哪儿溢出了 <<<
            self._debug_print_model_params("OverflowBeforeOpt")

            return False

        # 4) 反缩放梯度
        for p in self.master_params:
            p.grad.mul_(1.0 / (2 ** self.lg_loss_scale))

        # 5) optimizer.step() 更新参数
        opt.step()
        zero_master_grads(self.master_params)

        # 6) 把更新后的 master_params 拷贝回模型
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)

        # 7) 动态升高 loss_scale
        self.lg_loss_scale += self.fp16_scale_growth

 
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logkv_mean("grad_norm", grad_norm)
        logkv_mean("param_norm", param_norm)

        # 梯度裁剪
        if self.gradient_clip_val is not None:
            th.nn.utils.clip_grad_norm_(self.master_params, self.gradient_clip_val)

        opt.step()

        # 同理你也可以在这里 debug-print
        # self._debug_print_model_params("AfterOptFP32")

        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


    def master_params_to_state_dict(self, master_params):
        """
        用于保存权重时，把 master_params 拷回 state_dict。
        """
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        """
        从 state_dict 恢复 master_params。
        """
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    """
    如果梯度范数 (或其他数值) 是 NaN/Inf，就返回 True。
    """
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
