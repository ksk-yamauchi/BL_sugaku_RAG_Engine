import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# ⚙️ ページ設定と状態管理
# ==========================================
st.set_page_config(page_title="スタサプRAG FT ダッシュボード", page_icon="🏫", layout="wide")

if "step2_active" not in st.session_state:
    st.session_state.step2_active = False
if "delivery_mode" not in st.session_state:
    st.session_state.delivery_mode = None  
if "selected_action" not in st.session_state:
    st.session_state.selected_action = "分数が含まれる平方完成"
if "target_override" not in st.session_state:
    st.session_state.target_override = None  
if "pack_items" not in st.session_state:
    st.session_state.pack_items = {
        "item1": {"type": "動画", "title": "[数学Ⅰ] 分数が含まれる平方完成の解法 (02:15〜)", "visible": True},
        "item2": {"type": "動画", "title": "[数学Ⅰ] 練習問題: 分数が含まれる平方完成の解説 (05:30〜)", "visible": True},
        "item3": {"type": "動画 (前提)", "title": "[中学復習] 分数式の加法と減法", "visible": True},
    }
# 💡 【修正】画面が切り替わっても絶対に消えない専用の変数でタブを記憶
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "🗺️ 習熟度可視化 (クラス全体)"

# ==========================================
# 📊 ダミーデータの生成
# ==========================================
@st.cache_data
def load_dummy_data():
    data = [
        ["数学Ⅰ", "数と式", "式の計算", "展開の公式", "3つの文字を含む式の展開", 65, "知識・技能"],
        ["数学Ⅰ", "数と式", "式の計算", "因数分解", "基本的なたすき掛け", 75, "知識・技能"],
        ["数学Ⅰ", "数と式", "式の計算", "因数分解", "文字が含まれるたすき掛け", 25, "思考力・判断力等"],
        ["数学Ⅰ", "数と式", "実数", "絶対値", "絶対値記号を2つ含む方程式", 40, "思考力・判断力等"],
        ["数学Ⅰ", "二次関数", "二次関数のグラフ", "平方完成", "基本的な平方完成", 80, "知識・技能"],
        ["数学Ⅰ", "二次関数", "二次関数のグラフ", "平方完成", "分数が含まれる平方完成", 15, "知識・技能"],
        ["数学Ⅰ", "二次関数", "最大・最小", "二次関数の最大・最小", "定義域に制限がある最大・最小", 50, "思考力・判断力等"],
        ["数学Ⅰ", "二次関数", "最大・最小", "二次関数の最大・最小", "軸が動く場合の最大・最小", 30, "思考力・判断力等"],
        ["数学A", "場合の数と確率", "場合の数", "順列", "円順列", 70, "知識・技能"],
        ["数学A", "場合の数と確率", "場合の数", "組合せ", "組分け問題", 35, "思考力・判断力等"],
        ["数学A", "場合の数と確率", "確率", "条件付き確率", "原因の確率", 20, "思考力・判断力等"],
    ]
    df = pd.DataFrame(data, columns=["科目", "大項目", "中項目", "親概念", "学習アクション", "習熟度", "観点"])
    df["ウェイト"] = 1 
    return df

df_full = load_dummy_data()

# ==========================================
# 🎛️ サイドバー (フィルタリング)
# ==========================================
st.sidebar.title("🏫 フィルター設定")

if not st.session_state.step2_active:
    scope = st.sidebar.radio("👥 分析対象", ["学年全体", "クラス"])
    target_name = "学年全体"
    if scope == "クラス":
        target_name = st.sidebar.selectbox("クラスを選択", ["1年A組", "1年B組", "1年C組"])

    st.sidebar.markdown("---")
    st.sidebar.selectbox("📚 教科", ["数学"])
    selected_subjects = st.sidebar.multiselect(
        "📖 科目 (複数選択可)", 
        ["数学Ⅰ", "数学A"],
        default=["数学Ⅰ", "数学A"]
    )
    df_filtered = df_full[df_full["科目"].isin(selected_subjects)] if selected_subjects else df_full
else:
    st.sidebar.info("現在、宿題配信設定（STEP2）の編集中です。")
    target_name = "1年A組" if "scope" not in locals() else target_name

# ==========================================
# 🚀 画面遷移ロジック
# ==========================================
if st.session_state.step2_active:
    # -----------------------------------------------------------
    # 【画面B】 STEP2: 配信設定 画面
    # -----------------------------------------------------------
    st.title("宿題配信")
    st.markdown("### STEP2. 配信設定")
    
    st.markdown("#### 選択内容を確認")
    with st.container(border=True):
        if st.session_state.delivery_mode == "pack":
            st.info("💡 **遡り学習パック**: AIが現在の躓きと、根本原因となる前提知識の動画をセットで抽出しました。不要なものは「×」で外せます。")
            active_items = 0
            for key, item in st.session_state.pack_items.items():
                if item["visible"]:
                    active_items += 1
                    col1, col2, col3 = st.columns([1, 8, 1])
                    with col1:
                        st.markdown(f"**{item['type']}**")
                    with col2:
                        st.markdown(item["title"])
                    with col3:
                        if st.button("❌", key=f"del_{key}", help="この教材をパックから外す"):
                            st.session_state.pack_items[key]["visible"] = False
                            st.rerun()
            if active_items == 0:
                st.warning("配信する教材がありません。")
        elif st.session_state.delivery_mode == "video":
            st.markdown("**動画** | [数学Ⅰ] 分数が含まれる平方完成の解法 (02:15〜)")
        else:
            st.markdown("**問題** | [確認問題] 問3, 問4: 平方完成ドリル")

    st.markdown("#### 配信設定")
    with st.container(border=True):
        st.markdown("**配信先** <span style='color:red; font-size:0.8em;'>必須</span>", unsafe_allow_html=True)
        
        if st.session_state.target_override:
            st.success(f"✅ **{st.session_state.target_override}** （※面談ダッシュボードからの個別配信）")
        else:
            st.success(f"✅ **{target_name}** （※AI自動絞り込み: 当該アクションの習熟度が50%以下の対象者）")
        
        st.markdown("**宿題のタイプ** <span style='color:red; font-size:0.8em;'>必須</span>", unsafe_allow_html=True)
        if st.session_state.delivery_mode in ["pack", "video"]:
            st.radio("タイプ", ["動画＋確認テスト", "確認テストのみ"], index=0, label_visibility="collapsed")
        else:
            st.radio("タイプ", ["動画＋確認テスト", "確認テストのみ"], index=1, label_visibility="collapsed")
            
        st.markdown("**配信日時** <span style='color:red; font-size:0.8em;'>必須</span>", unsafe_allow_html=True)
        st.radio("日時", ["いますぐ配信", "日時指定"], index=0, label_visibility="collapsed")

    col_back, col_space, col_submit = st.columns([2, 5, 2])
    with col_back:
        if st.button("＜ 戻る", use_container_width=True):
            st.session_state.step2_active = False
            st.session_state.target_override = None 
            for k in st.session_state.pack_items:
                st.session_state.pack_items[k]["visible"] = True
            st.rerun()
    with col_submit:
        if st.button("宿題を配信", type="primary", use_container_width=True):
            st.success("🎉 宿題が配信されました！（デモ終了）")
            st.balloons()

else:
    # -----------------------------------------------------------
    # 【画面A】 ダッシュボード 画面
    # -----------------------------------------------------------
    st.title("🎯 スタサプRAG 先生向けダッシュボード")
    
    tabs_list = [
        "🗺️ 習熟度可視化 (クラス全体)", 
        "👤 面談ダッシュボード (連携デモ)", 
        "📅 年間計画(シラバス)自動生成", 
        "📸 プリント・答案画像解析"
    ]
    
    # 💡 セッションに保存されたタブ位置を初期値としてセット
    try:
        default_index = tabs_list.index(st.session_state.current_tab)
    except ValueError:
        default_index = 0

    selected_tab = st.radio(
        "メニュー",
        tabs_list,
        index=default_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    
    # 💡 選ばれたタブを即座にセッションへ保存（これで次回戻ってきても安全）
    st.session_state.current_tab = selected_tab
    
    st.markdown("---")
    
    if selected_tab == "🗺️ 習熟度可視化 (クラス全体)":
        prefix = "学年全体の" if target_name == "学年全体" else f"{target_name}の"
        st.markdown(f"**{prefix}現在の習熟度（GNN-KT推論スコア）**を可視化しています。赤色のブロックから根本的な躓き原因を特定してください。")

        if df_filtered.empty:
            st.warning("表示するデータがありません。")
        else:
            fig = px.treemap(
                df_filtered,
                path=[px.Constant("全選択科目 横断"), "科目", "大項目", "中項目", "親概念", "学習アクション"],
                values="ウェイト",
                color="習熟度",
                color_continuous_scale="RdYlBu", 
                range_color=[0, 100],
                hover_data={"習熟度": ":.1f%", "観点": True}, 
            )
            fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=450)
            fig.data[0].textinfo = 'label'
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 学習アクション・エクスプローラー (詳細分析と任意配信)", expanded=False):
            st.info("💡 【プロトタイプ仕様】 本番環境では上のマップのブロックをクリックすると連動してこのパネルが開き、対象が切り替わります。今回は下のリストから選択して体験してください。")
            
            st.session_state.selected_action = st.selectbox(
                "🔍 詳細を表示する学習アクション (Level 4) を選択:", 
                df_filtered["学習アクション"].tolist(),
                index=df_filtered["学習アクション"].tolist().index("分数が含まれる平方完成") if "分数が含まれる平方完成" in df_filtered["学習アクション"].tolist() else 0
            )
            
            selected_row = df_filtered[df_filtered["学習アクション"] == st.session_state.selected_action].iloc[0]
            current_score = selected_row["習熟度"]
            current_comp = selected_row["観点"]
            
            st.markdown(f"### 📍 {st.session_state.selected_action}")
            st.markdown(f"**現在の習熟度**: `{current_score}%` ｜ **学習観点**: `{current_comp}`")
            st.markdown("---")
            st.markdown(f"**この概念に対するアクション ({target_name}の対象者へ配信):**")
            
            col_v, col_p, col_pack = st.columns(3)
            with col_v:
                if st.button("📘 動画を配信", key="main_v", use_container_width=True):
                    st.session_state.delivery_mode = "video"
                    st.session_state.step2_active = True
                    st.rerun()
            with col_p:
                if st.button("📗 確認問題・類題のみを配信", key="main_p", use_container_width=True):
                    st.session_state.delivery_mode = "problem"
                    st.session_state.step2_active = True
                    st.rerun()
            with col_pack:
                if st.button("⏪ 前提知識を含むパック配信", key="main_pack", use_container_width=True):
                    st.session_state.delivery_mode = "pack"
                    st.session_state.step2_active = True
                    st.rerun()

        st.markdown("---")

        st.subheader(f"🚨 分析レポート")
        with st.container(border=True):
            st.error(f"**⚠️ {prefix}[見逃せない]弱点を検出しました (習熟度 15%)**")
            st.markdown("現在の躓き: **[数学Ⅰ] 二次関数 ➔ 平方完成 ➔ 『分数が含まれる平方完成』**")
            st.markdown("💡 **AI推論**: 係数に分数が含まれる段階で**計算間違いが疑われます。** 前提知識である **中学数学の『分数式の計算』** の定着不足が根本原因の可能性が高いです。")
            
            if st.button("⚡ この弱点に対する「遡り学習パック」を一括配信設定する", type="primary"):
                st.session_state.delivery_mode = "pack"
                st.session_state.step2_active = True
                st.rerun()

    elif selected_tab == "👤 面談ダッシュボード (連携デモ)":
        st.info("💡 **【ユースケース・デモ】** ここは面談担当の教員が、生徒の個別データを閲覧している画面を想定しています。専門外の教科であっても、AIの傾向分析から直接最適な課題を配信できます。")
        
        st.subheader("👤 山田太郎さん のデータ")
        
        with st.container(border=True):
            st.info("📘 **山田太郎さん の習熟度から見た傾向**\n\n💡 **AI推論 (GNN-KT)**: 数Ⅰの『二次関数』の習熟度が低いようですが、オントロジーの繋がりから、同時に数Aの『整数』の計算力も低下している兆候が検知されました。科目を横断した根本的なフォローを推奨します。")
            
            st.markdown("#### 🎯 AI推奨アクション (数学Ⅰ)")
            st.markdown("山田太郎さんの根本的な躓き原因である **『分数が含まれる平方完成』** の克服に向けた個別課題を配信します。")
            
            col_iv, col_ip, col_ipack = st.columns(3)
            with col_iv:
                if st.button("📘 動画を配信", key="iv_v", use_container_width=True):
                    st.session_state.delivery_mode = "video"
                    st.session_state.target_override = "山田太郎さん"
                    st.session_state.step2_active = True
                    st.rerun()
            with col_ip:
                if st.button("📗 確認問題・類題のみを配信", key="iv_p", use_container_width=True):
                    st.session_state.delivery_mode = "problem"
                    st.session_state.target_override = "山田太郎さん"
                    st.session_state.step2_active = True
                    st.rerun()
            with col_ipack:
                if st.button("⏪ 前提知識を含むパック配信", key="iv_pack", type="primary", use_container_width=True):
                    st.session_state.delivery_mode = "pack"
                    st.session_state.target_override = "山田太郎さん"
                    st.session_state.step2_active = True
                    st.rerun()
                    
    elif selected_tab == "📅 年間計画(シラバス)自動生成":
        st.info("🚧 【将来拡張】学校の年間指導計画（シラバス）PDFをアップロードすると、AIが自動でオントロジーとマッチングし、毎週の推奨配信リストを作成する機能がここに実装されます。")
    
    elif selected_tab == "📸 プリント・答案画像解析":
        st.info("🚧 【将来拡張】先生が自作したプリントや、生徒の答案画像をアップロードすると、AIが画像を解析して類似問題や解説動画を逆引き検索する機能がここに実装されます。")