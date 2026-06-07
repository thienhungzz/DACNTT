import numpy as np
import matplotlib.pyplot as plt

def reduce_mem_usage(df):
    """Giảm thiểu dung lượng bộ nhớ của DataFrame."""
    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_name_2']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('No_Event').astype('category')

    num_cols = {'sales': np.float32, 'sell_price': np.float32,
                'wday': np.int8, 'month': np.int8, 'year': np.int16,
                'snap_CA': np.int8, 'snap_TX': np.int8, 'snap_WI': np.int8}
    for c, dtype in num_cols.items():
        if c in df.columns:
            df[c] = df[c].astype(dtype)
    return df

def plot_inventory_comparison(rl_inv, eoq_inv, actual_demand, title, max_days=365):
    """Hàm hỗ trợ vẽ biểu đồ so sánh tồn kho."""
    plt.figure(figsize=(15, 6))
    plot_len = min(len(rl_inv), max_days)
    plt.plot(rl_inv[:plot_len], label='RL Inventory', color='green', alpha=0.8, linewidth=2)
    plt.plot(eoq_inv[:plot_len], label='EOQ Inventory', color='orange', linestyle='--')
    plt.plot(actual_demand[:plot_len], label='Actual Demand', color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Units')
    plt.legend()
    plt.tight_layout()
    plt.show()
