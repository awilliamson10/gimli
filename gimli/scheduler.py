import math


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def warmup(learning_rate, step, warmup_steps):
    return learning_rate * step / warmup_steps


# learning rate decay scheduler (cosine with warmup)
def lr_scheduler(optimizer, learning_rate, warmup_steps, decay_steps, min_lr):
    def _lr_adjuster(step):
        # 1. linear warmup
        if step < warmup_steps:
            lr = warmup(learning_rate, step, warmup_steps)
        # 2. if step > decay_steps, return min learning rate
        elif step > decay_steps:
            return min_lr
        # 3. in between, use cosine decay to min learning rate
        else:
            decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = min_lr + coeff * (learning_rate - min_lr)
        assign_learning_rate(optimizer, lr)

    return _lr_adjuster
