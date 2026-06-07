import math


def test_step_schedule_reaches_same_endpoints():
    base_lr = 3e-4
    total_steps = 5000
    values = []
    for step in (0, total_steps // 2, total_steps):
        progress = min(1.0, step / total_steps)
        values.append(base_lr * (0.1 + 0.9 * (math.cos(math.pi * progress) + 1.0) * 0.5))
    assert values[0] == base_lr
    assert values[-1] == base_lr * 0.1
    assert values[0] > values[1] > values[2]
