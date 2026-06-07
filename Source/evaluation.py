import math
import numpy as np

def calculate_eoq_metrics(real_demand, Q_star, ROP, h=0.001, S=3.05, p=5.0, L=2, max_cap=500):
    inv = Q_star
    pipeline = [0] * L
    total_cost = 0
    inv_history = []
    for d in real_demand:
        received = pipeline.pop(0) if L > 0 else 0
        inv = min(inv + received, max_cap)

        order = 0
        if inv + sum(pipeline) <= ROP:
            order = Q_star
            total_cost += S

        if L > 0:
            pipeline.append(order)
        else:
            inv = min(inv + order, max_cap)

        shortage = max(0, d - inv)
        inv = max(0, inv - d)
        total_cost += inv * h + shortage * p
        inv_history.append(inv)
    return total_cost, inv_history
