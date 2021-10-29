def evaluate_metric_quadratic(x):
    if x < 0.5:
        return 0
    else:
        return 4*(x-0.5)**2


def evaluate_metric_linear(x):
    if x < 0.5:
        return 0
    else:
        return 2*(x-0.5)