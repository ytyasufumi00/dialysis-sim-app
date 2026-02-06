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
    st.header("📚 パラメータ解説と臨床的意義")
    
    tab1, tab2, tab3 = st.tabs(["基礎知識 (Vd, KoA)", "各薬剤の特徴 (Key!)", "腎機能・半減期ガイド"])
    
    with tab1:
        st.markdown("""
        ### 1. 分布容積 ($V_1, V_2$)
        * **$V_1$ (中心室):** 血管内など、透析で直接浄化できる領域。
        * **$V_2$ (末梢室):** 組織・細胞内。ここにある薬物は移動($Q$)してこないと除去できません。
        
        ### 2. KoA (総括物質移動係数)
        透析膜の性能指標（拡散しやすさ）。
        * **800~1000 (超高効率):** メタノール、リチウム、尿素（小分子・非結合）
        * **500~700 (高効率):** カフェイン、アシクロビル、バルプロ酸(遊離型)
        * **< 400 (低効率):** 蛋白結合率が高い薬物（通常時のフェニトイン等）、巨大分子
        """)

    with tab2:
        st.info("薬剤ごとの挙動の違いに注目してください。")
        
        st.markdown("""
        #### 🔵 リチウム (Lithium)
        * **特徴:** 細胞内に蓄積するため、血中($V_1$)から抜けても、細胞($V_2$)からゆっくり湧き出してきます。
        * **挙動:** 透析終了後の**リバウンド（再上昇）が顕著**です。シミュレーションで終了1時間後の値を要確認。

        #### 🟡 バルプロ酸 (Valproic Acid)
        * **特徴:** 通常は蛋白結合率が高い(90%)ため透析で抜けにくいですが、**中毒域では結合が飽和し、遊離型が急増するため透析が著効します**。
        * **設定:** KoAを高め(600~)に設定しています。

        #### 🔴 メタノール (Methanol)
        * **特徴:** 極めて分子が小さく水溶性。透析で劇的に抜けます。
        * **注意:** 治療（ホメピゾール等）で代謝をブロックしている場合、半減期は**30〜50時間以上**になります。透析なしでは体から抜けません。

        #### 🟠 カルバマゼピン (Carbamazepine)
        * **特徴:** 活性代謝物の存在や、腸管からの再吸収もあり、リバウンドが有名です。蛋白結合率があるため、KoAはやや低めですが、長時間透析で総除去量を稼ぎます。
        """)

    with tab3:
        st.table({
            "薬剤": ["Caffeine", "Acyclovir", "Carbamazepine", "Valproic Acid", "Methanol", "Lithium"],
            "半減期目安 (Overdose時)": ["10~100h (代謝飽和)", "20h (腎不全)", "15~30h", "15~30h", "30~50h (代謝遮断時)", "24~36h (腎不全)"],
            "透析効効率": ["高い", "高い", "中程度", "高濃度で高い", "極めて高い", "高い(リバウンド大)"]
        })

# ==========================================
# 3. メインアプリケーション
# ==========================================

st.set_page_config(page_title="透析除去シミュレーター", layout="wide")
st.title("💊 薬物過量投与 透析除去シミュレーター (拡張版)")

# --- サイドバー設定 ---
st.sidebar.header("1. 患者・透析条件")
weight = st.sidebar.number_input("患者体重 (kg)", value=60.0, step=1.0)
qb = st.sidebar.slider("血流量 Qb (mL/min)", 100, 400, 200, step=10)
qd = st.sidebar.slider("透析液流量 Qd (mL/min)", 300, 800, 500, step=50)
hd_duration = st.sidebar.slider("透析時間 (時間)", 1, 12, 4) * 60
hd_start = st.sidebar.number_input("服用から透析開始まで (分)", value=120, step=30, help="分布がある程度完了した時点を想定")

st.sidebar.header("2. 薬剤選択・設定")
drug_list = ["Caffeine", "Acyclovir", "Carbamazepine", "Valproic Acid", "Methanol", "Lithium", "Custom"]
drug_choice = st.sidebar.selectbox("対象薬剤", drug_list)

# --- 薬剤パラメータ辞書 (Overdose / Renal Failure Scenario) ---
default_params = {
    # カフェイン: 代謝飽和でT1/2延長、除去良好
    'Caffeine': {'V1': 0.2, 'V2': 0.4, 'Q': 0.5, 'T1/2': 15.0, 'KoA': 700, 'dose': 6000},
    
    # アシクロビル: 腎不全でT1/2著明延長、除去良好
    'Acyclovir': {'V1': 0.15, 'V2': 0.55, 'Q': 0.2, 'T1/2': 20.0, 'KoA': 600, 'dose': 10000},
    
    # カルバマゼピン: Vd中等度、リバウンドあり、蛋白結合あるがOverdoseで遊離増
    'Carbamazepine': {'V1': 0.3, 'V2': 0.8, 'Q': 0.25, 'T1/2': 24.0, 'KoA': 450, 'dose': 8000},
    
    # バルプロ酸: 中毒域では蛋白結合飽和→Vd増・除去率増
    'Valproic Acid': {'V1': 0.15, 'V2': 0.25, 'Q': 0.3, 'T1/2': 18.0, 'KoA': 650, 'dose': 30000},
    
    # メタノール: Vdは体液量に近い。KoA最強。代謝ブロックでT1/2超延長
    'Methanol': {'V1': 0.4, 'V2': 0.2, 'Q': 0.8, 'T1/2': 40.0, 'KoA': 900, 'dose': 40000}, 
    
    # リチウム: 細胞内分布(V2)からの戻りが遅い(Q小)→リバウンド最強
    'Lithium': {'V1': 0.3, 'V2': 0.6, 'Q': 0.15, 'T1/2': 30.0, 'KoA': 850, 'dose': 5000},
    
    # カスタム
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
        t_half = st.number_input("半減期 (時間)", value=float(p['T1/2']), help="中毒時・腎不全時の値を想定")
    with col_k2:
        koa = st.number_input("KoA (mL/min)", value=int(p['KoA']), help="膜面積1.5~2.0m2想定")
        
    q_inter = st.slider("組織間移行クリアランス Q (L/min)", 0.01, 2.0, p['Q'], 0.01, help="小さいほどリバウンド大")

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
        
        # With HD
        ax.plot(time_steps/60, c1_hd, label='Blood (With HD)', color='tab:red', linewidth=2.5)
        ax.plot(time_steps/60, c2_hd, label='Tissue (With HD)', color='tab:blue', linestyle='--', linewidth=2)
        
        # HD Area
        ax.axvspan(hd_start/60, (hd_start + hd_duration)/60, color='red', alpha=0.1, label='HD Session')
        
        ax.set_title("Concentration vs Time")
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (µg/mL or mg/L)')
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
        ax.legend()
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
