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
    st.header("📚 パラメータ解説と入力の目安")
    
    tab1, tab2, tab3 = st.tabs(["基本パラメータ (V1, V2, KoA)", "腎機能と半減期", "薬剤別入力ガイド"])
    
    with tab1:
        st.markdown("""
        ### 1. 分布容積 ($V_1, V_2$) の考え方
        体内の薬物の居場所を「2つの部屋」に例えて計算しています。
        * **$V_1$ (中心室):** 血液や、血流が非常に多い臓器（心臓、腎臓、肝臓など）。透析で直接浄化できるのはここだけです。
        * **$V_2$ (末梢室):** 筋肉、脂肪、皮膚など。ここにある薬物は、一度血液($V_1$)に戻ってこないと透析で除去できません。
        
        > **ポイント:** $V_2$が大きい薬物ほど、組織に大量に蓄積しており、透析後に組織から血液への「染み出し（リバウンド）」が強く起こります。

        ### 2. KoA (総括物質移動係数)
        ダイアライザ（透析膜）の性能を表す指標です。
        * **意味:** 分子が小さいほど通りやすく（値が大きい）、分子が大きいほど通りにくい（値が小さい）。
        * **目安:**
            * 尿素などの小分子: KoA > 1000
            * **カフェイン/アシクロビル (MW 200前後): KoA 600~800** (非常に抜けやすい)
            * バンコマイシンなど中分子: KoA 300~500
        """)

    with tab2:
        st.markdown("""
        ### 腎機能と消失半減期 ($T_{1/2}$)
        このシミュレーターにおける「半減期」は、**「その患者さんの全身状態における半減期」**を入力してください。
        
        $$ T_{1/2} = \frac{0.693 \times V_{d}}{CL_{total}} $$
        
        * **腎排泄型薬剤（アシクロビルなど）:** 腎機能が悪いとクリアランス($CL_{total}$)が激減するため、半減期は**著しく延長**します。
        * **肝代謝型薬剤（カフェインなど）:**
            腎不全単独では半減期はあまり変わりませんが、**過量投与による代謝飽和**（肝臓の処理能力オーバー）により、半減期が延長します。
        """)

    with tab3:
        st.info("以下の表を参考に、患者の状態に合わせてパラメータを調整してください。")
        
        st.markdown("#### 💊 アシクロビル (Acyclovir)")
        st.markdown("""
        * **特徴:** ほとんどが腎臓から排泄されるため、腎機能に依存します。
        * **入力の目安:**
        """)
        st.table({
            "患者の状態": ["腎機能正常", "維持透析 / 無尿", "急性腎障害 (AKI)"],
            "半減期 ($T_{1/2}$) 目安": ["2.5 ~ 3 時間", "20 時間", "10 ~ 20 時間"],
            "備考": ["速やかに排泄される", "ほとんど排泄されない", "重症度に応じて設定"]
        })

        st.markdown("#### ☕ カフェイン (Caffeine)")
        st.markdown("""
        * **特徴:** 肝臓で代謝されます。通常は速いですが、**過量服薬時は代謝酵素が飽和し、分解スピードが落ちます（非線形薬物動態）。**
        * **入力の目安:**
        """)
        st.table({
            "患者の状態": ["治療域 (コーヒー数杯)", "中毒域 (過量服薬)", "重篤な中毒"],
            "半減期 ($T_{1/2}$) 目安": ["3 ~ 5 時間", "10 ~ 15 時間", "20 ~ 100 時間"],
            "備考": ["通常の代謝速度", "代謝が遅れ始める", "代謝が極端に遅延"]
        })
        st.warning("※ カフェイン中毒の場合、腎機能が正常でも「半減期」は長めに（10時間以上）設定するのが現実に近くなります。")

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
hd_start = st.sidebar.number_input("服用から透析開始まで (分)", value=60, step=10, help="服用時刻を0分とした時の透析開始時刻")

st.sidebar.header("2. 薬剤選択・設定")
drug_choice = st.sidebar.selectbox("対象薬剤", ["Caffeine", "Acyclovir", "Custom"])

# デフォルト値
default_params = {
    'Caffeine': {'V1': 0.2, 'V2': 0.4, 'Q': 0.5, 'T1/2': 15.0, 'KoA': 700}, # 中毒を想定してT1/2長め
    'Acyclovir': {'V1': 0.15, 'V2': 0.55, 'Q': 0.2, 'T1/2': 20.0, 'KoA': 600}, # 腎不全を想定
    'Custom': {'V1': 0.2, 'V2': 0.4, 'Q': 0.3, 'T1/2': 12.0, 'KoA': 500}
}
p = default_params[drug_choice]

with st.sidebar.expander("薬剤パラメータ詳細設定", expanded=True):
    overdose_amount = st.number_input("摂取量 (mg)", value=10000 if drug_choice=="Acyclovir" else 6000)
    
    st.caption(f"▼ {drug_choice} の推奨設定")
    v1_pk = st.slider("V1 (L/kg) 中心室", 0.05, 2.0, p['V1'], 0.01)
    v2_pk = st.slider("V2 (L/kg) 末梢室", 0.05, 5.0, p['V2'], 0.01)
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        t_half = st.number_input("消失半減期 (時間)", value=p['T1/2'], help="患者の腎・肝機能に応じた値を入力")
    with col_k2:
        koa = st.number_input("KoA (mL/min)", value=p['KoA'], help="透析膜の性能")
        
    q_inter = st.slider("組織間移行クリアランス (L/min)", 0.01, 2.0, p['Q'], 0.01, help="値が小さいほどリバウンドが顕著")

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
        
        # No HD
        ax.plot(time_steps/60, c1_none, label='Blood (No HD)', color='gray', linestyle=':', linewidth=2, alpha=0.6)
        ax.plot(time_steps/60, c2_none, label='Tissue (No HD)', color='lightblue', linestyle=':', linewidth=1.5, alpha=0.5)
        
        # With HD
        ax.plot(time_steps/60, c1_hd, label='Blood (With HD)', color='tab:red', linewidth=2.5)
        ax.plot(time_steps/60, c2_hd, label='Tissue (With HD)', color='tab:blue', linestyle='--', linewidth=2)
        
        # HD Area
        ax.axvspan(hd_start/60, (hd_start + hd_duration)/60, color='red', alpha=0.1, label='HD Session')
        
        ax.set_title("Concentration vs Time")
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
            if reb_diff > 0.5: # わずかな誤差は無視
                st.warning(f"Rebound (+1h): +{reb_diff:.1f} µg/mL")
            else:
                st.info("No significant rebound")
    
    # --- 解説セクションの呼び出し ---
    draw_explanation()

else:
    st.info("👈 サイドバーで条件を設定し、「シミュレーション実行」を押してください。")
    # 実行前にも解説が見られるように表示
    draw_explanation()
