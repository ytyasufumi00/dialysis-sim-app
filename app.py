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
        
        # 組織間移行速度定数 (L/min -> rate constant)
        self.Q_inter = drug_params['Q_inter_L_min']
        self.k12 = self.Q_inter / self.V1
        self.k21 = self.Q_inter / self.V2
        
        # 消失速度定数 k_el の計算
        # 入力された半減期(T1/2)を、全身(V_total)からのクリアランスとみなして算出
        # T1/2 = 0.693 * Vd / CL  => CL = 0.693 * Vd / T1/2
        # k_el = CL / V1
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
    
    # HD設定の展開
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
        rem_hd = (A1 / sim.V1) * current_cl # 1分間あたりの除去
        
        A1 = A1 + trans_net - elim - rem_hd
        A2 = A2 - trans_net
        
        if A1 < 0: A1 = 0
        if A2 < 0: A2 = 0
        
    return conc_v1, conc_v2

# ==========================================
# 2. UI & 解説表示用関数
# ==========================================

def draw_explanation():
    st.markdown("---")
    st.header("📚 パラメータ解説と臨床的意義")
    
    tab1, tab2 = st.tabs(["各薬剤の半減期目安 (正常 vs 腎不全)", "シミュレーションの考え方"])
    
    with tab1:
        st.markdown("### ⏱️ 半減期 ($T_{1/2}$) の入力目安")
        st.markdown("患者の病態（腎機能廃絶や過量服薬）に合わせて、適切な半減期を選択して入力してください。")
        
        # データ定義
        data = [
            {"薬": "アシクロビル", "正常": "2.5 時間", "腎不全/中毒": "**20 時間**", "備考": "腎排泄型。腎不全で著明に延長。"},
            {"薬": "リチウム", "正常": "18~24 時間", "腎不全/中毒": "**40~50+ 時間**", "備考": "腎排泄型。透析後のリバウンドが大。"},
            {"薬": "メタノール", "正常": "2~3 時間", "腎不全/中毒": "**30~50+ 時間**", "備考": "代謝拮抗薬(ホメピゾール等)使用時は著明に延長。"},
            {"薬": "カフェイン", "正常": "3~6 時間", "腎不全/中毒": "**10~100 時間**", "備考": "肝代謝。過量服薬による代謝飽和で延長。"},
            {"薬": "バルプロ酸", "正常": "10~16 時間", "腎不全/中毒": "**~30 時間**", "備考": "肝代謝。中毒域で蛋白結合が外れ、透析効率UP。"},
            {"薬": "カルバマゼピン", "正常": "10~20 時間", "腎不全/中毒": "**20~40 時間**", "備考": "肝代謝。徐放剤による吸収遅延・リバウンドに注意。"},
        ]
        st.table(data)
        st.caption("※ 文献により数値には幅があります。中毒時は個体差が大きいため、最悪のケース（長い半減期）を想定するのが安全です。")

    with tab2:
        st.markdown("""
        ### 📊 グラフの見方
        * **Blood (With HD):** 透析を行った場合の血中濃度（赤実線）。
        * **Tissue (With HD):** 透析を行った場合の組織内濃度（青破線）。これが高いままだと、透析後にリバウンドします。
        * **Blood (No HD):** 透析をしなかった場合の自然経過（灰色点線）。透析の効果判定に使用します。
        * **Tissue (No HD):** 透析をしなかった場合の組織内濃度（水色点線）。**【今回追加】**
        
        ### 💡 KoA (総括物質移動係数)
        * **800~1000:** メタノール、リチウム（小分子・水溶性）→ 非常に抜けやすい
        * **500~700:** カフェイン、アシクロビル、バルプロ酸（遊離型）→ 抜けやすい
        * **< 400:** 蛋白結合率が高い薬物など → 抜けにくい
        """)

# ==========================================
# 3. メインアプリケーション
# ==========================================

st.set_page_config(page_title="透析除去シミュレーター", layout="wide")
st.title("💊 薬物過量投与 透析除去シミュレーター")

# --- サイドバー設定 ---
st.sidebar.header("1. 患者・透析条件")
weight = st.sidebar.number_input("患者体重 (kg)", value=60.0, step=1.0)
qb = st.sidebar.slider("血流量 Qb (mL/min)", 100, 400, 200, step=10)
qd = st.sidebar.slider("透析液流量 Qd (mL/min)", 300, 800, 500, step=50)
hd_duration = st.sidebar.slider("透析時間 (時間)", 1, 12, 4) * 60
hd_start = st.sidebar.number_input("服用から透析開始まで (分)", value=120, step=30)

st.sidebar.header("2. 薬剤選択・設定")
drug_list = ["Caffeine", "Acyclovir", "Carbamazepine", "Valproic Acid", "Methanol", "Lithium", "Custom"]
drug_choice = st.sidebar.selectbox("対象薬剤", drug_list)

# --- 薬剤パラメータ辞書 ---
# 中毒時・腎不全時を想定したデフォルト値
default_params = {
    'Caffeine': {'V1': 0.2, 'V2': 0.4, 'Q': 0.5, 'T1/2': 15.0, 'KoA': 700, 'dose': 6000},
    'Acyclovir': {'V1': 0.15, 'V2': 0.55, 'Q': 0.2, 'T1/2': 20.0, 'KoA': 600, 'dose': 5000},
    'Carbamazepine': {'V1': 0.3, 'V2': 0.8, 'Q': 0.25, 'T1/2': 24.0, 'KoA': 450, 'dose': 6000},
    'Valproic Acid': {'V1': 0.15, 'V2': 0.25, 'Q': 0.3, 'T1/2': 20.0, 'KoA': 650, 'dose': 20000},
    'Methanol': {'V1': 0.4, 'V2': 0.2, 'Q': 0.8, 'T1/2': 40.0, 'KoA': 900, 'dose': 30000}, 
    'Lithium': {'V1': 0.3, 'V2': 0.6, 'Q': 0.15, 'T1/2': 40.0, 'KoA': 850, 'dose': 4000},
    'Custom': {'V1': 0.2, 'V2': 0.4, 'Q': 0.3, 'T1/2': 12.0, 'KoA': 500, 'dose': 5000}
}

p = default_params[drug_choice]

with st.sidebar.expander("薬剤パラメータ詳細設定", expanded=True):
    overdose_amount = st.number_input("摂取量 (mg)", value=p['dose'], step=100)
    
    st.caption(f"▼ {drug_choice} 設定値")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        v1_pk = st.slider("V1 (L/kg) 中心室", 0.05, 2.0, p['V1'], 0.01)
    with col_v2:
        v2_pk = st.slider("V2 (L/kg) 末梢室", 0.05, 5.0, p['V2'], 0.01)
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        t_half = st.number_input("半減期 (時間)", value=float(p['T1/2']), help="下の表を参考に設定してください")
    with col_k2:
        koa = st.number_input("KoA (mL/min)", value=int(p['KoA']))
        
    q_inter = st.slider("組織間移行クリアランス Q (L/min)", 0.01, 2.0, p['Q'], 0.01)

current_params = {
    'V1_per_kg': v1_pk, 'V2_per_kg': v2_pk, 
    'Q_inter_L_min': q_inter, 'T_half_hours': t_half, 'KoA': koa
}

# --- 実行ボタン ---
if st.button("シミュレーション実行", type="primary"):
    
    sim = DrugSimulation(current_params, weight)
    
    total_time = 24 * 60 
    time_steps = np.arange(0, total_time, 1)
    
    # 初期分布計算
    total_V_L = sim.V1 + sim.V2
    A1_init = overdose_amount * (sim.V1 / total_V_L)
    A2_init = overdose_amount * (sim.V2 / total_V_L)
    
    cl_hd_val_L = sim.calculate_hd_clearance(qb, qd, koa) / 1000.0
    
    # 計算
    hd_config = {'start': hd_start, 'duration': hd_duration, 'cl_val': cl_hd_val_L}
    c1_hd, c2_hd = run_scenario(sim, time_steps, A1_init, A2_init, hd_config)
    c1_none, c2_none = run_scenario(sim, time_steps, A1_init, A2_init, None)

    # --- グラフ描画 ---
    st.subheader(f"Simulation Result: {drug_choice} (24h)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # --- No HD (Reference) ---
        # Blood (No HD)
        ax.plot(time_steps/60, c1_none, label='Blood (No HD)', color='gray', linestyle=':', linewidth=2, alpha=0.6)
        # Tissue (No HD) - 復活！
        ax.plot(time_steps/60, c2_none, label='Tissue (No HD)', color='lightblue', linestyle=':', linewidth=1.5, alpha=0.6)
        
        # --- With HD ---
        # Tissue (With HD)
        ax.plot(time_steps/60, c2_hd, label='Tissue (With HD)', color='tab:blue', linestyle='--', linewidth=2, alpha=0.8)
        # Blood (With HD) - 最前面に
        ax.plot(time_steps/60, c1_hd, label='Blood (With HD)', color='tab:red', linewidth=2.5)
        
        # HD Area
        ax.axvspan(hd_start/60, (hd_start + hd_duration)/60, color='red', alpha=0.1, label='HD Session')
        
        ax.set_title("Concentration vs Time")
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µg/mL or mg/L)')
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        st.pyplot(fig)
        
    with col2:
        idx_24h = -1
        st.markdown("### at 24 hours")
        st.metric("Blood (With HD)", f"{c1_hd[idx_24h]:.1f}")
        st.metric("Blood (No HD)", f"{c1_none[idx_24h]:.1f}")
        
        if c1_none[idx_24h] > 0:
            reduction = (1 - c1_hd[idx_24h] / c1_none[idx_24h]) * 100
            st.success(f"Reduction: {reduction:.1f}%")
            
        st.markdown("---")
        # リバウンド表示
        end_idx = hd_start + hd_duration
        if end_idx < len(time_steps):
            post_1h_idx = min(end_idx + 60, len(time_steps)-1)
            reb_diff = c1_hd[post_1h_idx] - c1_hd[end_idx]
            
            st.write("### Post-HD Rebound")
            if reb_diff > 1.0: 
                st.warning(f"Rebound (+1h): +{reb_diff:.1f}")
            else:
                st.info("No significant rebound")
    
    # --- 解説セクション ---
    draw_explanation()

else:
    st.info("👈 サイドバーで条件を設定し、「シミュレーション実行」を押してください。")
    draw_explanation()
