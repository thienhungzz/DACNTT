import pandas as pd
import numpy as np
import gc
import os

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

def preprocess_data(data_path="../data/"):
    """Thực hiện toàn bộ quá trình tiền xử lý dữ liệu."""
    print("Đang tải dữ liệu...")
    df_calendar = pd.read_csv(f"{data_path}calendar.csv")
    df_stv = pd.read_csv(f"{data_path}sales_train_validation.csv")
    df_sp = pd.read_csv(f"{data_path}sell_prices.csv")

    print("Đang biến đổi dữ liệu (Melt)...")
    d_cols = [c for c in df_stv.columns if 'd_' in c]
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df_melted = pd.melt(df_stv, id_vars=id_cols, value_vars=d_cols, var_name='d', value_name='sales')
    del df_stv
    gc.collect()

    print("Đang gộp dữ liệu (Merge)...")
    df_merged = pd.merge(df_melted, df_calendar, on='d', how='left')
    del df_melted, df_calendar
    gc.collect()

    df_final = pd.merge(df_merged, df_sp, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    del df_merged, df_sp
    gc.collect()

    print("Đang làm sạch dữ liệu...")
    df_final = df_final.dropna(subset=['sell_price'])
    df_final = df_final.reset_index(drop=True)

    print("Đang tính toán các chi phí (Cost)...")
    df_m5 = df_final.copy()
    del df_final
    gc.collect()
    
    margin_data = {
        'year': [2011, 2011, 2011, 2012, 2012, 2012, 2013, 2013, 2013,
                 2014, 2014, 2014, 2015, 2015, 2015, 2016, 2016, 2016],
        'cat_id': ['FOODS', 'HOUSEHOLD', 'HOBBIES', 'FOODS', 'HOUSEHOLD', 'HOBBIES',
                   'FOODS', 'HOUSEHOLD', 'HOBBIES', 'FOODS', 'HOUSEHOLD', 'HOBBIES',
                   'FOODS', 'HOUSEHOLD', 'HOBBIES', 'FOODS', 'HOUSEHOLD', 'HOBBIES'],
        'gross_margin': [0.1875, 0.3015, 0.3750, 0.1920, 0.3105, 0.3820, 0.1945, 0.2980, 0.3910,
                         0.2012, 0.3025, 0.3865, 0.2103, 0.2987, 0.3845, 0.2131, 0.3064, 0.3912]
    }
    df_margins = pd.DataFrame(margin_data)

    if 'year' not in df_m5.columns:
        df_m5['date'] = pd.to_datetime(df_m5['date'])
        df_m5['year'] = df_m5['date'].dt.year
    else:
        df_m5['year'] = df_m5['year'].astype(int)

    df_m5 = pd.merge(df_m5, df_margins, on=['year', 'cat_id'], how='left')
    df_m5['gross_margin'] = df_m5['gross_margin'].fillna(0.25)
    df_m5['unit_cost_C'] = df_m5['sell_price'] * (1 - df_m5['gross_margin'])
    
    annual_holding_rate = 0.26
    df_m5['annual_holding_cost_H'] = df_m5['unit_cost_C'] * annual_holding_rate
    df_m5['daily_holding_cost_h'] = df_m5['annual_holding_cost_H'] / 365

    target_freq = {'FOODS': 365, 'HOUSEHOLD': 120, 'HOBBIES': 52}
    total_days = df_m5['date'].nunique()
    total_years = total_days / 365.25

    item_stats = df_m5.groupby(['item_id', 'cat_id']).agg(
        total_sales=('sales', 'sum'),
        mean_H=('annual_holding_cost_H', 'mean')
    ).reset_index()
    item_stats['annual_D'] = item_stats['total_sales'] / total_years
    item_stats = item_stats[item_stats['annual_D'] > 0].copy()
    item_stats['K'] = np.sqrt((item_stats['annual_D'] * item_stats['mean_H']) / 2)

    cat_stats = item_stats.groupby('cat_id').agg(
        K_total=('K', 'sum'),
        sku_count=('item_id', 'count')
    ).reset_index()
    cat_stats['target_freq'] = cat_stats['cat_id'].map(target_freq)
    cat_stats['N_target'] = cat_stats['sku_count'] * cat_stats['target_freq']
    cat_stats['setup_cost_S'] = np.round((cat_stats['K_total'] / cat_stats['N_target']) ** 2, 2)

    s_mapping = cat_stats.set_index('cat_id')['setup_cost_S'].to_dict()
    df_m5['setup_cost_S'] = df_m5['cat_id'].map(s_mapping)

    print("Đang tạo đặc trưng ML và tối ưu bộ nhớ...")
    ml_full_df = df_m5[['date', 'item_id', 'store_id', 'sales', 'wday', 'month', 'year',
                        'event_name_1', 'event_name_2', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price']].copy()
    ml_full_df = reduce_mem_usage(ml_full_df)

    ml_full_df = ml_full_df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)

    grouped = ml_full_df.groupby(['store_id', 'item_id'], observed=True)['sales']
    for i in [1, 7, 14, 28]:
        ml_full_df[f'lag_{i}'] = grouped.shift(i).astype(np.float32)

    ml_full_df['rolling_mean_7'] = grouped.shift(1).rolling(window=7, min_periods=1).mean().astype(np.float32)
    ml_full_df['rolling_mean_28'] = grouped.shift(1).rolling(window=28, min_periods=1).mean().astype(np.float32)

    ml_full_df.dropna(inplace=True)
    ml_full_df.reset_index(drop=True, inplace=True)
    gc.collect()

    print("Hoàn tất tiền xử lý!")
    return ml_full_df, df_m5

if __name__ == "__main__":
    # Test hàm
    ml_df, m5_df = preprocess_data(data_path="../data/")
    # Lưu ra file nếu cần thiết
    # ml_df.to_parquet("../data/ml_full_data.parquet")
