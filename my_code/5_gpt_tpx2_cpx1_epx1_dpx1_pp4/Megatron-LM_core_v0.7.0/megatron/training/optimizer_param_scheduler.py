# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate decay and weight decay incr functions."""

import math

from .utils import print_rank_0

class OptimizerParamScheduler(object):
    """Anneals learning rate and weight decay"""

    def __init__(self, optimizer, init_lr, max_lr, min_lr,
                 lr_warmup_steps, lr_decay_steps, lr_decay_style,
                 start_wd, end_wd, wd_incr_steps, wd_incr_style,
                 use_checkpoint_opt_param_scheduler=True,
                 override_opt_param_scheduler=False):

        # Class values.
        self.optimizer = optimizer                                             # trace_info : t_15907

        self.init_lr = init_lr                                                 # trace_info : t_15908
        self.max_lr = float(max_lr)                                            # trace_info : t_15909
        self.min_lr = min_lr                                                   # trace_info : t_15910
        assert self.min_lr >= 0.0                                              # trace_info : t_15911
        assert self.max_lr >= self.min_lr                                      # trace_info : t_15912
        assert self.init_lr <= self.max_lr                                     # trace_info : t_15913

        self.lr_warmup_steps = lr_warmup_steps                                 # trace_info : t_15914
        self.num_steps = 0                                                     # trace_info : t_15915
        self.lr_decay_steps = lr_decay_steps                                   # trace_info : t_15916
        assert self.lr_decay_steps > 0                                         # trace_info : t_15917
        assert self.lr_warmup_steps < self.lr_decay_steps                      # trace_info : t_15918

        self.lr_decay_style = lr_decay_style                                   # trace_info : t_15919

        self.start_wd = start_wd                                               # trace_info : t_15920
        self.end_wd = end_wd                                                   # trace_info : t_15921
        assert self.start_wd >= 0.0                                            # trace_info : t_15922
        assert self.end_wd >= self.start_wd                                    # trace_info : t_15923
        self.wd_incr_steps = wd_incr_steps                                     # trace_info : t_15924
        self.wd_incr_style = wd_incr_style                                     # trace_info : t_15925

        self.override_opt_param_scheduler = override_opt_param_scheduler       # trace_info : t_15926
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler# trace_info : t_15927
        if self.override_opt_param_scheduler:                                  # trace_info : t_15928
            assert not self.use_checkpoint_opt_param_scheduler, 'both override and '\
                'use-checkpoint are set.'

        # Set the learning rate
        self.step(0)                                                           # trace_info : t_15929
        print_rank_0('> learning rate decay style: {}'.format(self.lr_decay_style))# trace_info : t_15968


    def get_wd(self):
        """ Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:                                # trace_info : t_15932, t_21078, t_24806, t_28534
            return self.end_wd

        if self.wd_incr_style == 'constant':                                   # trace_info : t_15933, t_21079, t_24807, t_28535
            assert self.start_wd == self.end_wd                                # trace_info : t_15934, t_21080, t_24808, t_28536
            return self.end_wd                                                 # trace_info : t_15935, t_21081, t_24809, t_28537

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception('{} weight decay increment style is not supported.'.format(
                self.wd_incr_style))

        return self.start_wd + coeff * delta_wd


    def get_lr(self, param_group):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        max_lr = param_group.get('max_lr', self.max_lr)                        # trace_info : t_15939, t_15954, t_21085, t_21100, t_24813, ...
        min_lr = param_group.get('min_lr', self.min_lr)                        # trace_info : t_15940, t_15955, t_21086, t_21101, t_24814, ...

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:# trace_info : t_15941, t_15956, t_21087, t_21102, t_24815, ...
            return (                                                           # trace_info : t_15949, t_15964, t_21095, t_21110, t_24823, ...
                self.init_lr                                                   # trace_info : t_15942, t_15948, t_15957, t_15963, t_21088, ...
                + (
                    (max_lr - self.init_lr)                                    # trace_info : t_15943, t_15945, t_15947, t_15958, t_15960, ...
                    * float(self.num_steps)                                    # trace_info : t_15944, t_15959, t_21090, t_21105, t_24818, ...
                    / float(self.lr_warmup_steps)                              # trace_info : t_15946, t_15961, t_21092, t_21107, t_24820, ...
                )
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps ** 0.5 / (num_steps ** 0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.lr_decay_style))

        return min_lr + coeff * delta_lr


    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_steps += increment                                            # trace_info : t_15930, t_21076, t_24804, t_28532
        new_wd = self.get_wd()                                                 # trace_info : t_15931, t_21077, t_24805, t_28533
        for param_group in self.optimizer.param_groups:                        # trace_info : t_15936, t_15952, t_15967, t_21082, t_21098, ...
            new_lr = self.get_lr(param_group)                                  # trace_info : t_15938, t_15953, t_21084, t_21099, t_24812, ...
            param_group['lr'] = new_lr * param_group.get('lr_mult', 1.0)       # trace_info : t_15950, t_15965, t_21096, t_21111, t_24824, ...
            param_group['weight_decay'] = new_wd * param_group.get('wd_mult', 1.0)# trace_info : t_15951, t_15966, t_21097, t_21112, t_24825, ...


    def state_dict(self):
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, \
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint' \
                f'value {sd_value} for {name} do not match'
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value


    def load_state_dict(self, sd):

        if 'start_lr' in sd:
            max_lr_ = sd['start_lr']
        else:
            max_lr_ = sd['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')

        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')

        if 'warmup_iter' in sd:
            lr_warmup_steps_ = sd['warmup_iter']
        elif 'warmup_steps' in sd:
            lr_warmup_steps_ = sd['warmup_steps']
        else:
            lr_warmup_steps_ = sd['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(self.lr_warmup_steps,
                                                lr_warmup_steps_,
                                                'warmup iterations')

        if 'end_iter' in sd:
            lr_decay_steps_ = sd['end_iter']
        elif 'decay_steps' in sd:
            lr_decay_steps_  = sd['decay_steps']
        else:
            lr_decay_steps_ = sd['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(self.lr_decay_steps, lr_decay_steps_,
                                               'total number of iterations')

        if 'decay_style' in sd:
            lr_decay_style_ = sd['decay_style']
        else:
            lr_decay_style_ = sd['lr_decay_style']
        self.lr_decay_style = self._check_and_set(self.lr_decay_style,
                                               lr_decay_style_,
                                               'learning rate decay style')

        if 'num_iters' in sd:
            num_steps = sd['num_iters']
        else:
            num_steps = sd['num_steps']
        self.step(increment=num_steps)


        if 'start_wd' in sd:
            self.start_wd = self._check_and_set(self.start_wd,
                                                sd['start_wd'],
                                                "start weight decay")
            self.end_wd = self._check_and_set(self.end_wd,
                                                sd['end_wd'],
                                                "end weight decay")
            self.wd_incr_steps = self._check_and_set(self.wd_incr_steps,
                                                sd['wd_incr_steps'],
                                                "total number of weight decay iterations")
            self.wd_incr_style = self._check_and_set(self.wd_incr_style,
                                                sd['wd_incr_style'],
                                                "weight decay incr style")