import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 計算ロジッククラス
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
        
        # 自己消失速度定数
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

# シミュレーション実行用関数 (透析あり/なし共通)
def run_scenario(sim, time_steps, A1_init, A2_init, hd_config=None):
    """
    hd_config: None なら透析なし。
               {'start': 分, 'duration': 分, 'cl_val': L/min} なら透析あり。
    """
    conc_v1 = np.zeros(len(time_steps))
    conc_v2 = np.zeros(len(time_steps))
    
    A1 = A1_init
    A2 = A2_init
    
    for i, t in enumerate(time_steps):
        conc_v1[i] = A1 / sim.V1
        conc_v2[i] = A2 / sim.V2
        
        # 透析クリアランスの決定
        current_cl = 0.0
        if hd_config:
            if (t >= hd_config['start']) and (t < hd_config['start'] + hd_config['duration']):
                current_cl = hd_config['cl_val']
        
        # 差分方程式
        trans = (sim.k21 * A2) - (sim.k12 * A1)
        elim = sim.k_el * A1
        rem_hd = (A1 / sim.V1) * current_cl # 1分間
        
        A1 = A1 + trans - elim - rem_hd
        A2 = A2 - trans
        
        if A1 < 0: A1 = 0
        if A2 < 0: A2 = 0
        
    return conc_v1, conc_v2

# ==========================================
# 2. Streamlit アプリケーション部分
# ==========================================

st.set_page_config(page_title="透析除去シミュレーター", layout="wide")

st.title("💊 薬物過量投与 透析除去シミュレーター")
st.markdown("透析介入あり vs 自然経過の比較シミュレーション (24時間)")

# --- サイドバー：条件設定 ---
st.sidebar.header("1. 患者・透析条件")

weight = st.sidebar.number_input("患者体重 (kg)", value=60.0, step=1.0)
qb = st.sidebar.slider("血流量 Qb (mL/min)", 100, 400, 200, step=10)
qd = st.sidebar.slider("透析液流量 Qd (mL/min)", 300, 800, 500, step=50)
hd_duration = st.sidebar.slider("透析時間 (時間)", 1, 12, 4) * 60 # 分換算
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

with st.sidebar.expander("薬剤パラメータ詳細設定", expanded=(drug_choice=="Custom")):
    overdose_amount = st.number_input("摂取量 (mg)", value=3000 if drug_choice=="Acyclovir" else 5000)
    v1_pk = st.slider("V1 (L/kg) 中心室", 0.05, 1.0, p['V1'], 0.05)
    v2_pk = st.slider("V2 (L/kg) 末梢室", 0.05, 2.0, p['V2'], 0.05)
    q_inter = st.slider("組織間移行クリアランス (L/min)", 0.01, 2.0, p['Q'], 0.01)
    t_half = st.number_input("消失半減期 (時間)", value=p['T1/2'])
    koa = st.number_input("KoA (mL/min)", value=p['KoA'])

current_params = {
    'V1_per_kg': v1_pk, 'V2_per_kg': v2_pk, 
    'Q_inter_L_min': q_inter, 'T_half_hours': t_half, 'KoA': koa
}

# --- シミュレーション実行 ---
if st.button("シミュレーション実行", type="primary"):
    
    sim = DrugSimulation(current_params, weight)
    
    # 時間軸: 24時間固定 (1440分)
    total_time = 24 * 60 
    time_steps = np.arange(0, total_time, 1)
    
    # 初期量計算 (平衡状態を仮定)
    total_V_L = sim.V1 + sim.V2
    A1_init = overdose_amount * (sim.V1 / total_V_L)
    A2_init = overdose_amount * (sim.V2 / total_V_L)
    
    # HDクリアランス計算 (L/min)
    cl_hd_val_L = sim.calculate_hd_clearance(qb, qd, koa) / 1000.0
    
    # --- シナリオA: 透析あり (With HD) ---
    hd_config = {
        'start': hd_start,
        'duration': hd_duration,
        'cl_val': cl_hd_val_L
    }
    c1_hd, c2_hd = run_scenario(sim, time_steps, A1_init, A2_init, hd_config)
    
    # --- シナリオB: 透析なし (No HD / Natural Course) ---
    c1_none, c2_none = run_scenario(sim, time_steps, A1_init, A2_init, hd_config=None)

    # --- 結果描画 ---
    st.subheader(f"シミュレーション結果: {drug_choice} (24時間推移)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 透析なし (点線・薄い色)
        ax.plot(time_steps/60, c1_none, label='Blood (No HD)', color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # 透析あり (実線・濃い色)
        ax.plot(time_steps/60, c1_hd, label='Blood (With HD)', color='tab:red', linewidth=2.5)
        # 組織濃度（透析ありの時のみ表示すると見やすい）
        ax.plot(time_steps/60, c2_hd, label='Tissue (With HD)', color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 透析区間のハイライト
        ax.axvspan(hd_start/60, (hd_start + hd_duration)/60, color='red', alpha=0.1, label='HD Session')
        
        ax.set_title("血中濃度推移の比較")
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µg/mL)')
        ax.set_xlim(0, 24) # X軸を24時間に固定
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        # 数値比較
        end_hd_time = hd_start + hd_duration
        idx_end = min(end_hd_time, len(time_steps)-1)
        idx_24h = -1 # 最後の点(24h)
        
        st.markdown("### 24時間後の濃度")
        val_hd_24 = c1_hd[idx_24h]
        val_none_24 = c1_none[idx_24h]
        
        st.metric("透析あり (24h)", f"{val_hd_24:.1f} µg/mL")
        st.metric("透析なし (24h)", f"{val_none_24:.1f} µg/mL")
        
        if val_none_24 > 0:
            reduction = (1 - val_hd_24 / val_none_24) * 100
            st.success(f"透析による減少効果: {reduction:.1f}%")
            
        st.markdown("---")
        st.markdown("### 透析終了直後")
        st.write(f"血中濃度: **{c1_hd[idx_end]:.1f}** µg/mL")
        
        # リバウンドチェック
        idx_1h_post = min(end_hd_time + 60, len(time_steps)-1)
        rebound_val = c1_hd[idx_1h_post]
        if rebound_val > c1_hd[idx_end]:
            diff = rebound_val - c1_hd[idx_end]
            st.warning(f"⚠ 1時間後に +{diff:.1f} µg/mL のリバウンド予測")

else:
    st.info("👈 サイドバーの設定を確認して「シミュレーション実行」を押してください。")
