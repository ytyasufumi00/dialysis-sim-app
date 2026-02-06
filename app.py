import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 計算ロジッククラス (修正版)
# ==========================================
class DrugSimulation:
    def __init__(self, drug_params, weight):
        self.weight = weight
        self.V1 = drug_params['V1_per_kg'] * weight
        self.V2 = drug_params['V2_per_kg'] * weight
        
        # 組織間移行速度定数
        self.Q_inter = drug_params['Q_inter_L_min']
        self.k12 = self.Q_inter / self.V1
        self.k21 = self.Q_inter / self.V2
        
        # --- 【修正点】消失速度定数 k_el の計算 ---
        # 2コンパートメントモデルにおいて、入力された「半減期(T1/2)」が
        # 「全身クリアランスとしての見かけの半減期」を指すと解釈し、
        # V1だけでなくV_totalを考慮して微小速度定数 k_el (k10) を算出します。
        # CL_total = (ln(2) * V_total) / T_half
        # k_el = CL_total / V1
        
        total_V = self.V1 + self.V2
        t_half_min = drug_params['T_half_hours'] * 60
        
        if t_half_min > 0:
            self.k_el = (0.693 * total_V) / (t_half_min * self.V1)
        else:
            self.k_el = 0

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

def run_scenario(sim, time_steps, A1_init, A2_init, hd_config=None):
    conc_v1 = np.zeros(len(time_steps))
    conc_v2 = np.zeros(len(time_steps))
    
    A1 = A1_init
    A2 = A2_init
    
    # HDクリアランス値 (L/min)
    hd_cl_val = hd_config['cl_val'] if hd_config else 0.0
    hd_start = hd_config['start'] if hd_config else -1
    hd_end = hd_config['start'] + hd_config['duration'] if hd_config else -1
    
    for i, t in enumerate(time_steps):
        conc_v1[i] = A1 / sim.V1
        conc_v2[i] = A2 / sim.V2
        
        # 透析実施判定
        current_cl = 0.0
        if hd_config and (t >= hd_start) and (t < hd_end):
            current_cl = hd_cl_val
        
        # 差分方程式
        trans_2to1 = sim.k21 * A2
        trans_1to2 = sim.k12 * A1
        trans_net = trans_2to1 - trans_1to2
        
        elim = sim.k_el * A1
        rem_hd = (A1 / sim.V1) * current_cl # 1分間
        
        A1 = A1 + trans_net - elim - rem_hd
        A2 = A2 - trans_net
        
        if A1 < 0: A1 = 0
        if A2 < 0: A2 = 0
        
    return conc_v1, conc_v2

# ==========================================
# 2. Streamlit UI
# ==========================================

st.set_page_config(page_title="透析除去シミュレーター", layout="wide")

st.title("💊 薬物過量投与 透析除去シミュレーター")

# --- サイドバー設定 ---
st.sidebar.header("1. 患者・透析条件")
weight = st.sidebar.number_input("患者体重 (kg)", value=60.0, step=1.0)
qb = st.sidebar.slider("血流量 Qb (mL/min)", 100, 400, 200, step=10)
qd = st.sidebar.slider("透析液流量 Qd (mL/min)", 300, 800, 500, step=50)
hd_duration = st.sidebar.slider("透析時間 (時間)", 1, 12, 4) * 60
hd_start = st.sidebar.number_input("服用から透析開始まで (分)", value=60, step=10)

st.sidebar.header("2. 薬剤選択・設定")
drug_choice = st.sidebar.selectbox("対象薬剤", ["Caffeine", "Acyclovir", "Custom"])

default_params = {
    'Caffeine': {'V1': 0.2, 'V2': 0.4, 'Q': 0.5, 'T1/2': 10.0, 'KoA': 700},
    'Acyclovir': {'V1': 0.15, 'V2': 0.55, 'Q': 0.2, 'T1/2': 20.0, 'KoA': 600},
    'Custom': {'V1': 0.2, 'V2': 0.4, 'Q': 0.3, 'T1/2': 12.0, 'KoA': 500}
}
p = default_params[drug_choice]

with st.sidebar.expander("薬剤パラメータ詳細設定", expanded=True): # 常に表示推奨
    overdose_amount = st.number_input("摂取量 (mg)", value=12000 if drug_choice=="Acyclovir" else 5000)
    v1_pk = st.slider("V1 (L/kg) 中心室", 0.05, 2.0, p['V1'], 0.05)
    v2_pk = st.slider("V2 (L/kg) 末梢室", 0.05, 5.0, p['V2'], 0.05)
    q_inter = st.slider("組織間移行クリアランス (L/min)", 0.01, 2.0, p['Q'], 0.01, help="値が小さいほどリバウンドが強くなります")
    t_half = st.number_input("消失半減期 (時間)", value=p['T1/2'], help="全身からの排泄半減期")
    koa = st.number_input("KoA (mL/min)", value=p['KoA'])

current_params = {
    'V1_per_kg': v1_pk, 'V2_per_kg': v2_pk, 
    'Q_inter_L_min': q_inter, 'T_half_hours': t_half, 'KoA': koa
}

# --- 実行 ---
if st.button("シミュレーション実行", type="primary"):
    
    sim = DrugSimulation(current_params, weight)
    
    # 時間軸: 24時間固定
    total_time = 24 * 60 
    time_steps = np.arange(0, total_time, 1)
    
    # 初期量 (平衡状態を仮定)
    total_V_L = sim.V1 + sim.V2
    A1_init = overdose_amount * (sim.V1 / total_V_L)
    A2_init = overdose_amount * (sim.V2 / total_V_L)
    
    # HDクリアランス
    cl_hd_val_L = sim.calculate_hd_clearance(qb, qd, koa) / 1000.0
    
    # Scenario A: With HD
    hd_config = {'start': hd_start, 'duration': hd_duration, 'cl_val': cl_hd_val_L}
    c1_hd, c2_hd = run_scenario(sim, time_steps, A1_init, A2_init, hd_config)
    
    # Scenario B: No HD
    c1_none, c2_none = run_scenario(sim, time_steps, A1_init, A2_init, None)

    # --- 描画 ---
    st.subheader(f"Simulation Result: {drug_choice} (24h)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot No HD lines (Reference)
        ax.plot(time_steps/60, c1_none, label='Blood (No HD)', color='gray', linestyle=':', linewidth=2, alpha=0.6)
        # Tissue (No HD)も追加して、Tissue (With HD)と比較できるようにする
        ax.plot(time_steps/60, c2_none, label='Tissue (No HD)', color='lightblue', linestyle=':', linewidth=1.5, alpha=0.5)
        
        # Plot HD lines
        ax.plot(time_steps/60, c1_hd, label='Blood (With HD)', color='tab:red', linewidth=2.5)
        ax.plot(time_steps/60, c2_hd, label='Tissue (With HD)', color='tab:blue', linestyle='--', linewidth=2)
        
        # HD区間
        ax.axvspan(hd_start/60, (hd_start + hd_duration)/60, color='red', alpha=0.1, label='HD Session')
        
        ax.set_title("Concentration vs Time") # 文字化け回避のため英語
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µg/mL)')
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        idx_24h = -1
        
        st.markdown("### at 24 hours")
        st.metric("Blood (With HD)", f"{c1_hd[idx_24h]:.1f} µg/mL")
        st.metric("Blood (No HD)", f"{c1_none[idx_24h]:.1f} µg/mL")
        
        reduction = (1 - c1_hd[idx_24h] / c1_none[idx_24h]) * 100
        st.success(f"Reduction: {reduction:.1f}%")
        
        st.markdown("---")
        # リバウンド評価
        end_idx = hd_start + hd_duration
        if end_idx < len(time_steps):
            post_1h_idx = min(end_idx + 60, len(time_steps)-1)
            reb_diff = c1_hd[post_1h_idx] - c1_hd[end_idx]
            
            st.write("### Post-HD Rebound")
            if reb_diff > 0:
                st.warning(f"Rebound (+1h): +{reb_diff:.1f} µg/mL")
            else:
                st.info("No significant rebound")

else:
    st.info("Please set parameters and click button.")
