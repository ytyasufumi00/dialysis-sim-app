import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 計算ロジッククラス (変更なし)
# ==========================================
class DrugSimulation:
    def __init__(self, drug_params, weight):
        self.weight = weight
        self.V1 = drug_params['V1_per_kg'] * weight
        self.V2 = drug_params['V2_per_kg'] * weight
        
        # 組織間移行速度
        self.Q_inter = drug_params['Q_inter_L_min']
        self.k12 = self.Q_inter / self.V1
        self.k21 = self.Q_inter / self.V2
        
        # 自己消失
        self.k_el = 0.693 / (drug_params['T_half_hours'] * 60)

    def calculate_hd_clearance(self, Qb, Qd, KoA, sc=1.0):
        if Qb == 0: return 0
        ratio = Qb / Qd
        Z = (KoA / Qb) * (1 - ratio)
        
        if abs(1 - ratio) < 0.001:
            clearance = Qb * (KoA / (KoA + Qb))
        else:
            exp_z = np.exp(Z)
            clearance = Qb * (exp_z - 1) / (exp_z - ratio)
        return clearance * sc

# ==========================================
# 2. Streamlit アプリケーション部分
# ==========================================

st.set_page_config(page_title="透析除去シミュレーター", layout="wide")

st.title("💊 薬物過量投与 透析除去シミュレーター")
st.markdown("2コンパートメントモデルによる、透析中および透析後のリバウンドシミュレーション")

# --- サイドバー：条件設定 ---
st.sidebar.header("1. 患者・透析条件")

weight = st.sidebar.number_input("患者体重 (kg)", value=60.0, step=1.0)
qb = st.sidebar.slider("血流量 Qb (mL/min)", 100, 400, 200, step=10)
qd = st.sidebar.slider("透析液流量 Qd (mL/min)", 300, 800, 500, step=50)
hd_duration = st.sidebar.slider("透析時間 (時間)", 1, 8, 4) * 60 # 分換算
hd_start = st.sidebar.number_input("服用から透析開始まで (分)", value=60, step=10)

st.sidebar.header("2. 薬剤選択・設定")
drug_choice = st.sidebar.selectbox("対象薬剤", ["Caffeine", "Acyclovir", "Custom"])

# デフォルトパラメータ
default_params = {
    'Caffeine': {'V1': 0.2, 'V2': 0.4, 'Q': 0.5, 'T1/2': 10.0, 'KoA': 700},
    'Acyclovir': {'V1': 0.15, 'V2': 0.55, 'Q': 0.2, 'T1/2': 20.0, 'KoA': 600},
    'Custom': {'V1': 0.2, 'V2': 0.4, 'Q': 0.3, 'T1/2': 12.0, 'KoA': 500}
}
p = default_params[drug_choice]

# パラメータ調整（Custom選択時以外も微調整可能に）
with st.sidebar.expander("薬剤パラメータ詳細設定", expanded=(drug_choice=="Custom")):
    overdose_amount = st.number_input("摂取量 (mg)", value=3000 if drug_choice=="Acyclovir" else 5000)
    v1_pk = st.slider("V1 (L/kg) 中心室", 0.05, 1.0, p['V1'], 0.05)
    v2_pk = st.slider("V2 (L/kg) 末梢室", 0.05, 2.0, p['V2'], 0.05)
    q_inter = st.slider("組織間移行クリアランス (L/min)", 0.01, 2.0, p['Q'], 0.01)
    t_half = st.number_input("消失半減期 (時間)", value=p['T1/2'])
    koa = st.number_input("KoA (mL/min)", value=p['KoA'])

# 辞書に再格納
current_params = {
    'V1_per_kg': v1_pk, 'V2_per_kg': v2_pk, 
    'Q_inter_L_min': q_inter, 'T_half_hours': t_half, 'KoA': koa
}

# --- シミュレーション実行 ---
if st.button("シミュレーション実行", type="primary"):
    
    sim = DrugSimulation(current_params, weight)
    
    # 時間軸作成 (開始〜透析終了後5時間)
    total_time = hd_start + hd_duration + 300
    time_steps = np.arange(0, total_time, 1)
    
    conc_v1 = np.zeros(len(time_steps))
    conc_v2 = np.zeros(len(time_steps))
    
    # 初期量計算 (平衡状態を仮定)
    total_V_L = sim.V1 + sim.V2
    A1 = overdose_amount * (sim.V1 / total_V_L)
    A2 = overdose_amount * (sim.V2 / total_V_L)
    
    # HDクリアランス計算
    cl_hd_val_L = sim.calculate_hd_clearance(qb, qd, koa) / 1000.0
    
    # ループ計算
    for i, t in enumerate(time_steps):
        conc_v1[i] = A1 / sim.V1
        conc_v2[i] = A2 / sim.V2
        
        is_hd_active = (t >= hd_start) and (t < hd_start + hd_duration)
        current_cl = cl_hd_val_L if is_hd_active else 0.0
        
        trans = (sim.k21 * A2) - (sim.k12 * A1)
        elim = sim.k_el * A1
        rem_hd = (A1 / sim.V1) * current_cl
        
        A1 = A1 + trans - elim - rem_hd
        A2 = A2 - trans
        
        if A1 < 0: A1 = 0
        if A2 < 0: A2 = 0

    # --- 結果描画 ---
    st.subheader(f"シミュレーション結果: {drug_choice}")
    
    # カラム分け（グラフと数値）
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_steps/60, conc_v1, label='Blood (V1)', color='tab:red', linewidth=2.5)
        ax.plot(time_steps/60, conc_v2, label='Tissue (V2)', color='tab:blue', linestyle='--')
        
        # 透析区間のハイライト
        ax.axvspan(hd_start/60, (hd_start + hd_duration)/60, color='gray', alpha=0.2, label='HD Session')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µg/mL)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        # 主要な数値の表示
        start_idx = hd_start
        end_idx = hd_start + hd_duration
        rebound_idx = min(end_idx + 60, len(time_steps)-1) # 終了1時間後
        
        c_start = conc_v1[start_idx]
        c_end = conc_v1[end_idx]
        c_rebound = conc_v1[rebound_idx]
        
        st.metric("透析前濃度", f"{c_start:.1f} µg/mL")
        st.metric("透析終了時濃度", f"{c_end:.1f} µg/mL", delta=f"-{(c_start-c_end):.1f}")
        st.metric("終了1時間後 (リバウンド)", f"{c_rebound:.1f} µg/mL", delta=f"+{(c_rebound-c_end):.1f}", delta_color="inverse")
        
        removal_rate = (1 - (c_end / c_start)) * 100
        st.info(f"濃度低下率: {removal_rate:.1f}%")

else:
    st.info("👈 サイドバーの設定を確認して「シミュレーション実行」を押してください。")
