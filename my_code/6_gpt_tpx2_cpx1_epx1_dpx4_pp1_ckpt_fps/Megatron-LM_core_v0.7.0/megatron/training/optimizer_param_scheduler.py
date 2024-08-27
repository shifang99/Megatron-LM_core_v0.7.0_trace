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
        self.optimizer = optimizer                                             # trace_info : t_15870

        self.init_lr = init_lr                                                 # trace_info : t_15871
        self.max_lr = float(max_lr)                                            # trace_info : t_15872
        self.min_lr = min_lr                                                   # trace_info : t_15873
        assert self.min_lr >= 0.0                                              # trace_info : t_15874
        assert self.max_lr >= self.min_lr                                      # trace_info : t_15875
        assert self.init_lr <= self.max_lr                                     # trace_info : t_15876

        self.lr_warmup_steps = lr_warmup_steps                                 # trace_info : t_15877
        self.num_steps = 0                                                     # trace_info : t_15878
        self.lr_decay_steps = lr_decay_steps                                   # trace_info : t_15879
        assert self.lr_decay_steps > 0                                         # trace_info : t_15880
        assert self.lr_warmup_steps < self.lr_decay_steps                      # trace_info : t_15881

        self.lr_decay_style = lr_decay_style                                   # trace_info : t_15882

        self.start_wd = start_wd                                               # trace_info : t_15883
        self.end_wd = end_wd                                                   # trace_info : t_15884
        assert self.start_wd >= 0.0                                            # trace_info : t_15885
        assert self.end_wd >= self.start_wd                                    # trace_info : t_15886
        self.wd_incr_steps = wd_incr_steps                                     # trace_info : t_15887
        self.wd_incr_style = wd_incr_style                                     # trace_info : t_15888

        self.override_opt_param_scheduler = override_opt_param_scheduler       # trace_info : t_15889
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler# trace_info : t_15890
        if self.override_opt_param_scheduler:                                  # trace_info : t_15891
            assert not self.use_checkpoint_opt_param_scheduler, 'both override and '\
                'use-checkpoint are set.'

        # Set the learning rate
        self.step(0)                                                           # trace_info : t_15892
        print_rank_0('> learning rate decay style: {}'.format(self.lr_decay_style))# trace_info : t_15931


    def get_wd(self):
        """ Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:                                # trace_info : t_15895, t_21034, t_24671, t_92278
            return self.end_wd

        if self.wd_incr_style == 'constant':                                   # trace_info : t_15896, t_21035, t_24672, t_92279
            assert self.start_wd == self.end_wd                                # trace_info : t_15897, t_21036, t_24673, t_92280
            return self.end_wd                                                 # trace_info : t_15898, t_21037, t_24674, t_92281

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

        max_lr = param_group.get('max_lr', self.max_lr)                        # trace_info : t_15902, t_15917, t_21041, t_21056, t_24678, ...
        min_lr = param_group.get('min_lr', self.min_lr)                        # trace_info : t_15903, t_15918, t_21042, t_21057, t_24679, ...

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:# trace_info : t_15904, t_15919, t_21043, t_21058, t_24680, ...
            return (                                                           # trace_info : t_15912, t_15927, t_21051, t_21066, t_24688, ...
                self.init_lr                                                   # trace_info : t_15905, t_15911, t_15920, t_15926, t_21044, ...
                + (
                    (max_lr - self.init_lr)                                    # trace_info : t_15906, t_15908, t_15910, t_15921, t_15923, ...
                    * float(self.num_steps)                                    # trace_info : t_15907, t_15922, t_21046, t_21061, t_24683, ...
                    / float(self.lr_warmup_steps)                              # trace_info : t_15909, t_15924, t_21048, t_21063, t_24685, ...
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
        self.num_steps += increment                                            # trace_info : t_15893, t_21032, t_24669, t_92276
        new_wd = self.get_wd()                                                 # trace_info : t_15894, t_21033, t_24670, t_92277
        for param_group in self.optimizer.param_groups:                        # trace_info : t_15899, t_15915, t_15930, t_21038, t_21054, ...
            new_lr = self.get_lr(param_group)                                  # trace_info : t_15901, t_15916, t_21040, t_21055, t_24677, ...
            param_group['lr'] = new_lr * param_group.get('lr_mult', 1.0)       # trace_info : t_15913, t_15928, t_21052, t_21067, t_24689, ...
            param_group['weight_decay'] = new_wd * param_group.get('wd_mult', 1.0)# trace_info : t_15914, t_15929, t_21053, t_21068, t_24690, ...


    def state_dict(self):
        state_dict = {                                                         # trace_info : t_31407, t_99000
            'max_lr': self.max_lr,                                             # trace_info : t_31397, t_98990
            'lr_warmup_steps': self.lr_warmup_steps,                           # trace_info : t_31398, t_98991
            'num_steps': self.num_steps,                                       # trace_info : t_31399, t_98992
            'lr_decay_style': self.lr_decay_style,                             # trace_info : t_31400, t_98993
            'lr_decay_steps': self.lr_decay_steps,                             # trace_info : t_31401, t_98994
            'min_lr': self.min_lr,                                             # trace_info : t_31402, t_98995
            'start_wd': self.start_wd,                                         # trace_info : t_31403, t_98996
            'end_wd': self.end_wd,                                             # trace_info : t_31404, t_98997
            'wd_incr_style': self.wd_incr_style,                               # trace_info : t_31405, t_98998
            'wd_incr_steps': self.wd_incr_steps                                # trace_info : t_31406, t_98999
        }
        return state_dict                                                      # trace_info : t_31408, t_99001


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