import json
import os
import re
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# =========================================================
# ⚙️ 設定・初期化
# =========================================================
st.set_page_config(
    page_title="スタサプRAG AIチューター", page_icon="🎓", layout="wide"
)

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ .env ファイルに GEMINI_API_KEY が設定されていません。")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.6-flash"

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_DB_PATH = os.path.join(PARENT_DIR, "global_vector_db_cache.json")

TOLERANCE = 0.05
PENALTY_WEIGHT = 0.015


# =========================================================
# 🧠 ベクトル化・類似度計算 ＆ 画像解析 (JSON出力対応版)
# =========================================================
@st.cache_resource
def load_db():
    if not os.path.exists(GLOBAL_DB_PATH):
        return None
    with open(GLOBAL_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text, model_name):
    clean_model = (
        model_name
        if model_name.startswith("models/")
        else f"models/{model_name}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:embedContent?key={API_KEY}"
    payload = {"model": clean_model, "content": {"parts": [{"text": text}]}}
    try:
        res = requests.post(
            url, json=payload, proxies={"http": None, "https": None}, timeout=15
        )
        res.raise_for_status()
        return res.json()["embedding"]["values"]
    except Exception as e:
        st.error(f"❌ ベクトルAPI通信エラー: {e}")
        return None


def cosine_similarity(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return (
        0.0 if norm1 == 0 or norm2 == 0 else float(np.dot(v1, v2) / (norm1 * norm2))
    )


def clean_math_for_label(text):
    if not text: 
        return ""
    t = text.replace("$", "")
    t = t.replace(r"\pm", "±").replace(r"\times", "×").replace(r"\div", "÷")
    t = t.replace("^2", "²").replace("^3", "³")
    t = t.replace(r"\sqrt", "√")
    return t

# 🌟 ラベルの二重表示を防ぐヘルパー関数
def clean_q_label(q_num_str):
    if not q_num_str:
        return ""
    return str(q_num_str).replace("確認問題 ", "").replace("確認問題", "").replace("大問 ", "").replace("大問", "").strip()


def analyze_image_with_gemini_json(image_bytes):
    prompt = """
    あなたは優秀な数学教師です。アップロードされた画像を解析し、以下の厳密なJSON形式で出力してください。
    
    【抽出ルール】
    ・画像内に「次の式を展開せよ。」や「次の方程式を解け。」のような、複数の小問に共通する「リード文（指示文）」が含まれている場合、それを単独の項目にせず、それぞれの小問の冒頭に必ず結合してください。
    ・（例）「次の式を展開せよ。(1) $x+y$ (2) $x-y$」という画像の場合、抽出結果は ["次の式を展開せよ。(1) $x+y$", "次の式を展開せよ。(2) $x-y$"] とします。
    ・数式部分はLaTeX形式とし、必ず $ 記号で囲んでください（例: $a^2 + b^2$）。
    ・バックスラッシュはエスケープして正しいJSON文字列にしてください。
    
    {
      "image_type": "PROBLEM", 
      "extracted_items": [
        "次の式を展開せよ。(1) $a^2 + b^2$", 
        "次の式を展開せよ。(2) $2ab$"
      ]
    }
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes, mime_type="image/jpeg"
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.1
            ),
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"画像解析エラー: {e}")
        return {"image_type": "PROBLEM", "extracted_items": []}


def generate_required_concepts(problem_text):
    prompt = f"""
    あなたは優秀な高校数学教師です。以下の生徒が直面している問題（または質問）を解くために必要な「高校数学の概念や公式」を箇条書きで簡潔に提示・解説してください。
    数式はLaTeX形式とし、必ず $ または $$ 記号で囲んで生徒が理解しやすいように要点をまとめてください。
    
    【問題・質問】
    {problem_text}
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return response.text
    except Exception:
        return "⚠️ 概念情報の取得に失敗しました。"


def sanitize_query_for_search(raw_query):
    text = raw_query
    text = re.sub(r'\$.*?\$', ' ', text)
    text = re.sub(r'[\(（][a-zA-Z0-9あ-んア-ン]{1,2}[\)）]', ' ', text)
    text = re.sub(r'[\[［【][a-zA-Z0-9あ-んア-ン]{1,2}[\]］】]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else raw_query


# =========================================================
# 🔍 検索ロジック ＆ GraphRAG ネットワーク探索
# =========================================================
def execute_search_for_ui(search_query, db, is_drilldown=False):
    embed_model = db.get("embed_model", "models/text-embedding-004")
    concept_nodes = db.get("global_concept_nodes", {})
    question_nodes = db.get("global_question_nodes", {})

    sanitized_query = sanitize_query_for_search(search_query)

    query_vector = get_embedding(sanitized_query, embed_model)
    if not query_vector:
        return None

    raw_concept_results = []
    for cid, node in concept_nodes.items():
        sim = cosine_similarity(query_vector, node["concept_vector"])
        raw_concept_results.append({"id": cid, "base_score": sim, "score": sim, "node": node})

    question_results = []
    for qid, node in question_nodes.items():
        sim = cosine_similarity(query_vector, node["question_vector"])
        question_results.append({"id": qid, "base_score": sim, "score": sim, "node": node})

    concept_keywords = [
        "指導", "教え方", "とは", "意味", "概念", "基礎", "仕組み", "について",
        "学びたい", "知りたい", "わからない", "教えて",
    ]
    if any(kw in sanitized_query for kw in concept_keywords):
        for item in raw_concept_results:
            item["score"] += 0.05

    problem_keywords = [
        "問題", "解き方", "問", "例題", "演習", "確認問題", "解法", "計算", "を展開",
    ]
    if any(kw in sanitized_query for kw in problem_keywords):
        for item in question_results:
            item["score"] += 0.05

    stop_words = [
        "について", "指導したい", "教えて", "解き方", "解法", "とは", "意味", 
        "概念", "基礎", "仕組み", "学びたい", "知りたい", "わからない", 
        "問題", "ありますか", "探して", "の", "指導", "ドリル", "テスト"
    ]
    
    clean_query = sanitized_query
    for w in stop_words:
        clean_query = clean_query.replace(w, " ")
    
    keywords = [kw.strip() for kw in clean_query.split() if kw.strip()]
    if not keywords:
        keywords = [sanitized_query.strip()]
    
    for item in raw_concept_results:
        node = item["node"]
        target_text = f"{node.get('concept_name', '')} {node.get('parent_concept', '')} {node.get('summary', '')}"
        if any(len(kw) >= 2 and kw in target_text for kw in keywords):
            item["score"] += 0.30

    for item in question_results:
        node = item["node"]
        target_text = f"{node.get('question_text', '')} {node.get('matched_concept', '')}"
        if any(len(kw) >= 2 and kw in target_text for kw in keywords):
            item["score"] += 0.30

    raw_concept_results.sort(key=lambda x: x["score"], reverse=True)
    question_results.sort(key=lambda x: x["score"], reverse=True)

    top_concept_score = raw_concept_results[0]["score"] if raw_concept_results else 0.0
    top_question_score = question_results[0]["score"] if question_results else 0.0

    # 🌟 ルート判定のハイブリッド化
    is_concept_intent = top_concept_score >= top_question_score
    
    explicit_problem_kws = ["問題", "演習", "ドリル", "テスト", "解き方", "解法"]
    if any(kw in search_query for kw in explicit_problem_kws):
        is_concept_intent = False
    elif any(kw in search_query for kw in ["とは", "意味", "教えて", "概念", "仕組み"]):
        is_concept_intent = True

    result_data = {
        "intent": "concept" if is_concept_intent else "question",
        "top_match": None,
        "top_matches": [],  # 🌟 1%以内の僅差スコアの複数候補
        "runner_ups": [],
        "linked_questions": [],
        "connected_questions": [],
        "required_concepts_text": "",
        "is_drilldown": is_drilldown,
        "graph_prerequisites": [], 
        "graph_siblings": [],      
        "graph_next_steps": [],
        "graph_linked_tasks": [],       # 新設: 問題ルート用のタスクノード配列
        "graph_linked_knowledges": [],  # 新設: 問題ルート用の知識ノード配列
        "sanitized_query": sanitized_query if sanitized_query != search_query else None
    }

    target_list = raw_concept_results if is_concept_intent else question_results
    if not target_list:
        return result_data

    top_raw = target_list[0]
    top_time_idx = top_raw["node"].get("global_timeline_index", 0)

    processed_results = []
    for item in target_list:
        base_sim = item["base_score"]
        boost_amount = item["score"] - item["base_score"]
        node = item["node"]
        this_time_idx = node.get("global_timeline_index", 0)
        final_score = item["score"]
        penalty_reason = "なし (既習・復習範囲)"

        if (
            top_time_idx is not None
            and this_time_idx is not None
            and this_time_idx > top_time_idx
        ):
            if node.get("lecture_name") == top_raw["node"].get("lecture_name"):
                penalty_reason = "同じ講義内の先のステップ"
            else:
                penalty = (this_time_idx - top_time_idx) * PENALTY_WEIGHT
                final_score -= penalty
                penalty_reason = f"⚠️ 未習範囲減点 (-{penalty*100:.1f}%)"

        processed_results.append(
            {
                "final_score": final_score,
                "base_score": base_sim,
                "boost_amount": boost_amount,
                "node": node,
                "penalty_reason": penalty_reason,
            }
        )

    processed_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # 🌟 僅差スコア（1%以内）の複数取得ロジック
    top_score = processed_results[0]["final_score"]
    top_matches = []
    runner_ups = []
    for r in processed_results:
        diff = top_score - r["final_score"]
        if diff <= 0.01:
            top_matches.append(r)
        elif diff <= TOLERANCE:
            runner_ups.append(r)
            
    result_data["top_match"] = top_matches[0]
    result_data["top_matches"] = top_matches
    result_data["runner_ups"] = runner_ups

    top_node = processed_results[0]["node"]
    
    target_c_name = top_node.get("concept_name") if is_concept_intent else top_node.get("matched_concept")
    
    # 概念ルート用の Graph RAG オントロジー構築（既存）
    if target_c_name:
        target_c_node = None
        for cid, c_node in concept_nodes.items():
            if c_node["concept_name"] == target_c_name:
                target_c_node = c_node
                break
                
        if target_c_node:
            prereqs_raw = target_c_node.get("prerequisite_concepts", [])
            for p_item in prereqs_raw:
                if isinstance(p_item, dict):
                    p_name = p_item.get("concept_name", "")
                    p_type = p_item.get("dependency_type", "mandatory")
                    p_reason = p_item.get("reasoning", "")
                else:
                    p_name = str(p_item)
                    p_type = "mandatory"
                    p_reason = ""
                
                if not p_name:
                    continue

                for cid, c_node in concept_nodes.items():
                    if c_node["concept_name"] == p_name or c_node.get("parent_concept") == p_name:
                        already_added = any(
                            x["node"]["global_c_id"] == c_node["global_c_id"] 
                            for x in result_data["graph_prerequisites"]
                        )
                        if not already_added:
                            result_data["graph_prerequisites"].append({
                                "node": c_node,
                                "dependency_type": p_type,
                                "reasoning": p_reason,
                                "prereq_name": p_name
                            })
                            
            parent_name = target_c_node.get("parent_concept")
            if parent_name and parent_name != "未分類":
                for cid, c_node in concept_nodes.items():
                    if c_node.get("parent_concept") == parent_name and c_node["concept_name"] != target_c_name:
                        if c_node not in result_data["graph_siblings"]:
                            result_data["graph_siblings"].append(c_node)
            
            for cid, c_node in concept_nodes.items():
                p_list = c_node.get("prerequisite_concepts_list", [])
                if not p_list:
                    p_raw = c_node.get("prerequisite_concepts", [])
                    p_list = [p["concept_name"] if isinstance(p, dict) else p for p in p_raw]
                    
                if target_c_name in p_list:
                    if c_node not in result_data["graph_next_steps"]:
                        result_data["graph_next_steps"].append(c_node)

    graph_concept_names = set()
    for item in result_data["graph_prerequisites"]:
        p_n = item["node"].get("concept_name")
        if p_n: graph_concept_names.add(p_n)
        
    for node in result_data["graph_siblings"] + result_data["graph_next_steps"]:
        if "concept_name" in node:
            graph_concept_names.add(node["concept_name"])
            
    filtered_runner_ups = []
    for r in result_data["runner_ups"]:
        r_node = r["node"]
        if "concept_name" in r_node:
            if r_node["concept_name"] not in graph_concept_names:
                filtered_runner_ups.append(r)
        else:
            filtered_runner_ups.append(r)
            
    result_data["runner_ups"] = filtered_runner_ups

    # 🌟 相談1＆2: 配列構造に対応した問題・横展開の引き当てロジック
    if is_concept_intent:
        # タブ対応のため、top_matches の各要素に対して個別にリンクを取得
        for tm in top_matches:
            c_name = tm["node"].get("concept_name")
            tm["linked_questions"] = [
                q for q in question_nodes.values() 
                if c_name in q.get("linked_task_names", []) or c_name in q.get("linked_knowledge_names", [])
            ]
        # (後方互換用)
        c_name_legacy = top_node["concept_name"]
        result_data["linked_questions"] = [
            q for q in question_nodes.values() 
            if c_name_legacy in q.get("linked_task_names", []) or c_name_legacy in q.get("linked_knowledge_names", [])
        ]
    else:
        # 問題ルート:
        top_qid = top_node["global_q_id"]
        t_names = set(top_node.get("linked_task_names", []))
        k_names = set(top_node.get("linked_knowledge_names", []))
        
        connected_questions = []
        for qid, q in question_nodes.items():
            if qid == top_qid: continue
            
            q_t_names = set(q.get("linked_task_names", []))
            q_k_names = set(q.get("linked_knowledge_names", []))
            
            # 案A（タスク優先）: 同じタスク（技能）を測るものを横展開（類題）とする
            if t_names and q_t_names.intersection(t_names):
                connected_questions.append(q)
            # フォールバック: もしタスクがない問題なら、知識が共通するものを横展開とする
            elif not t_names and k_names and q_k_names.intersection(k_names):
                connected_questions.append(q)
                
        result_data["connected_questions"] = connected_questions

        # 問題ルートの Graph RAG 表示用ノードの取得
        for cid, cnode in concept_nodes.items():
            c_name_val = cnode.get("concept_name")
            if c_name_val in t_names:
                result_data["graph_linked_tasks"].append(cnode)
            elif c_name_val in k_names:
                result_data["graph_linked_knowledges"].append(cnode)

        needs_concept = not is_drilldown
        if needs_concept:
            content_search_kws = ["の問題", "例題", "類題", "ありますか", "探して", "ドリル", "テスト"]
            solve_kws = ["解き方", "解法", "教えて", "解説", "わからない"]
            if any(kw in search_query for kw in content_search_kws) and not any(kw in search_query for kw in solve_kws):
                needs_concept = False

        if needs_concept:
            result_data["required_concepts_text"] = generate_required_concepts(search_query)

    return result_data


# 🌟 【UI共通化】動画アイテム（主従対応版）を描画するヘルパー関数
def render_video_item(v, unique_key, is_modeling=False):
    with st.container(border=True):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            a_type = v.get("alignment_type", "")
            
            # 🌟 バッジの出し分け（モデリング時はシンプルに【解説】）
            if a_type in ["concept_introduction", "task_walkthrough", "direct_explanation"]:
                badge = "💡 【解説】" if is_modeling else "💡 【メイン解説】"
            elif a_type == "prerequisite_review":
                badge = "⏪ 【前提・復習】"
            else:
                badge = "🎬 【解説動画】"

            st.markdown(f"{badge} **{v.get('video_file')}** (`{v.get('start_time')}`〜)")
            
            # 🌟 第二アクション（大問解説）では「なぜこの動画？」の理由表記をカット
            if not is_modeling and v.get('reasoning'):
                st.markdown(f"**🤔 なぜこの動画？:** `{v.get('reasoning')}`")
            if v.get('explanation_summary'):
                st.markdown(f"**💬 講師の解説（概要）:** {v.get('explanation_summary')}")
                
        with col_btn:
            if st.button("📑 チャプター", key=f"cat_btn_{unique_key}", use_container_width=True):
                st.session_state.history.append({
                    "result": st.session_state.current_result,
                    "query": st.session_state.display_query,
                    "image_choices": st.session_state.pending_image_choices,
                })
                st.session_state.selected_video = v.get("video_file")
                st.rerun()

# =========================================================
# 🎨 画面描画 (UI)
# =========================================================
def main():
    st.title("🚀 スタサプRAG ナレッジエンジン")

    db = load_db()
    if not db:
        st.error(
            "❌ データベースが見つかりません。先にバックエンドでDBを構築してください。"
        )
        return

    # 🌟 サイドバーにバージョン情報を表示
    engine_ver = db.get("metadata", {}).get("engine_version", "バージョン情報なし")
    st.sidebar.markdown(f"**⚙️ エンジンバージョン:**\n`{engine_ver}`")
    st.sidebar.markdown(f"**📱 UI バージョン:**\n`AIチューター UI Ver 4.6.1`")

    # 🌟 State初期化
    for key in ["history", "current_result", "display_query", "pending_image_choices", "last_clicked_node", "selected_video"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "history" else None

    # ==========================================
    # 🎬 タイムスタンプ（チャプター）表示 UI
    # ==========================================
    if st.session_state.get("selected_video"):
        v_file = st.session_state.selected_video
        catalog = db.get("global_video_catalog", {})
        v_data = catalog.get(v_file)
        
        if st.button("🔙 検索結果に戻る", use_container_width=True):
            st.session_state.selected_video = None
            st.rerun()
            
        st.markdown(f"## 📺 動画プレイヤー: `{v_file}`")
        if v_data:
            st.caption(f"🎓 講義名: {v_data.get('lecture_name', '')} | 🏷️ 授業タイプ: {v_data.get('role', '')}")
            
            # ダミーの動画プレイヤー枠 (純粋なURL文字列で指定)
            st.video("[https://www.w3schools.com/html/mov_bbb.mp4](https://www.w3schools.com/html/mov_bbb.mp4)") 
            
            st.markdown("### 📑 タイムライン・チャプター (解説要約つき)")
            st.info("💡 講師の解説（概要）を事前に確認して、見たいチャプターから再生できます。")
            
            for idx, seg in enumerate(v_data.get("segments", [])):
                start = seg.get('start_time', '00:00')
                end = seg.get('end_time', '00:00')
                topic = seg.get('topic', '無題')
                
                with st.expander(f"⏱️ {start} 〜 {end} | 📌 {topic}", expanded=(idx==0)):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**💬 講師の解説（概要）:**\n> {seg.get('explanation_summary', 'データなし')}")
                    with col2:
                        if st.button("▶️ ここから再生", key=f"play_{v_file}_{idx}", use_container_width=True):
                            st.toast(f"{start} から再生を開始しました！（モック機能）")
        else:
            st.warning("⚠️ この動画の詳細なチャプターカタログデータがデータベースに見つかりません。")
            
        return  # プレイヤーを描画したら他のUIは出さずに終了


    # ==========================================
    # 📸 画像アップロード複数候補 UI
    # ==========================================
    if st.session_state.pending_image_choices:
        st.info(
            "📸 画像から複数の項目が検出されました。学習したい問題を1つ選択してください。"
        )
        img_data = st.session_state.pending_image_choices
        choices = img_data.get("extracted_items", [])
        img_type = img_data.get("image_type", "PROBLEM")

        with st.container(border=True):
            cols = st.columns(min(len(choices), 3))
            for i, item in enumerate(choices):
                with cols[i % 3]:
                    st.markdown(f"**候補 {i+1}:**")
                    st.markdown(item)
                    if st.button(
                        "🔍 この問題を検索",
                        key=f"choice_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.history.append({
                            "result": st.session_state.current_result,
                            "query": st.session_state.display_query,
                            "image_choices": st.session_state.pending_image_choices,
                        })
                        boost = " の解き方" if img_type == "PROBLEM" else " について詳しく知りたい"
                        new_query = f"{item}{boost}"
                        st.session_state.display_query = new_query
                        with st.spinner("検索中..."):
                            res = execute_search_for_ui(new_query, db, is_drilldown=False)
                            if res:
                                st.session_state.current_result = res
                            st.session_state.pending_image_choices = None
                            st.session_state.last_clicked_node = None
                        st.rerun()

            st.divider()
            if st.button("🏠 キャンセルしてトップに戻る", use_container_width=True):
                st.session_state.pending_image_choices = None
                st.session_state.history = []
                st.session_state.last_clicked_node = None
                st.rerun()


    # ==========================================
    # 🔍 検索結果 UI
    # ==========================================
    elif st.session_state.current_result:
        res = st.session_state.current_result
        
        st.info(f"🔍 現在の学習テーマ: **{st.session_state.display_query}**")
        
        if res.get("sanitized_query"):
            st.caption(f"✨ **自動ノイズ除去フィルター適用:** `{res['sanitized_query']}`")
            
        col_back, col_home = st.columns(2)

        with col_back:
            if st.button("🔙 前の画面に戻る", use_container_width=True):
                if len(st.session_state.history) > 0:
                    prev_state = st.session_state.history.pop()
                    st.session_state.current_result = prev_state["result"]
                    st.session_state.display_query = prev_state["query"]
                    st.session_state.pending_image_choices = prev_state.get("image_choices")
                    st.session_state.last_clicked_node = None
                else:
                    st.session_state.current_result = None
                    st.session_state.pending_image_choices = None
                st.rerun()

        with col_home:
            if st.button("🏠 新しい検索を始める (トップに戻る)", use_container_width=True):
                st.session_state.history = []
                st.session_state.current_result = None
                st.session_state.display_query = ""
                st.session_state.pending_image_choices = None
                st.session_state.last_clicked_node = None
                st.rerun()

        st.divider()

        if res and res.get("top_matches"):
            intent_label = (
                "📘 概念インプット優先ルート"
                if res["intent"] == "concept"
                else "📗 問題・解法ステップ優先ルート"
            )

            st.subheader(f"🎯 第一候補: {intent_label}")

            with st.container(border=True):
                # ===============================================
                # 📘 概念インプット優先ルートの描画 (タブ化対応)
                # ===============================================
                if res["intent"] == "concept":
                    top_matches = res.get("top_matches", [res["top_match"]])
                    if len(top_matches) > 1:
                        tabs = st.tabs([m["node"].get("concept_name", "概念") for m in top_matches])
                    else:
                        tabs = [st.container()]
                        
                    for tab_idx, (tm, tab) in enumerate(zip(top_matches, tabs)):
                        with tab:
                            node = tm["node"]
                            title = node.get("concept_name")
                            
                            if res.get("is_drilldown"):
                                st.markdown(f"### {title}")
                            else:
                                display_score = min(tm['final_score'], 1.0)
                                base = tm['base_score']
                                boost = tm['boost_amount']
                                pen_str = f" | {tm['penalty_reason']}" if tm['penalty_reason'] != "なし (既習・復習範囲)" else ""
                                
                                st.markdown(f"### {title} (総合適合度: {display_score*100:.1f}%)")
                                st.caption(
                                    f"📊 **【スコア内訳】** ベース類似度: {base*100:.1f}% "
                                    f"| 加点ブースト: +{boost*100:.1f}%{pen_str}"
                                )
                                
                            st.caption(f"🎓 講義名: {node.get('lecture_name', '未設定')}")

                            st.info(f"**💡 概念要約:** {node.get('summary', '要約なし')}")
                            p_concept = node.get("parent_concept", "未分類")
                            
                            comp_str = node.get("pillar", "未設定")
                            
                            st.markdown(f"**🔼 親概念 (Level 3):** `{p_concept}` | **🏷️ 観点:** `{comp_str}`")
                            
                            with st.expander(
                                f"🏛️ 指導要領: {node.get('mext_hierarchy', '未割り当て')} "
                                f"(コード: {node.get('mext_code', 'N/A')})"
                            ):
                                st.markdown(f"**【公式テキスト】** {node.get('mext_official_text', '情報なし')}")
                                st.markdown(f"**【解説要約】** {node.get('mext_explanation', '情報なし')}")

                            st.divider()

                            st.markdown("#### 🎬 第一アクション (概念インプット講義)")
                            catalog = db.get("global_video_catalog", {})
                            input_videos = []
                            
                            c_videos = node.get("main_videos", []) + node.get("review_videos", [])
                            
                            for v in c_videos:
                                v_file = v.get("video_file")
                                v_role = catalog.get(v_file, {}).get("role", "")
                                # role が exercise_walkthrough （大問・例題解説）でない場合のみ表示
                                if v_role != "exercise_walkthrough":
                                    input_videos.append(v)

                            for idx, v in enumerate(input_videos):
                                render_video_item(v, f"action1_{node.get('global_c_id')}_{tab_idx}_{idx}", is_modeling=False)
                            if not input_videos:
                                st.write("該当なし")

                            st.markdown("#### 📘 第二アクション (モデリング: 大問・例題解説)")
                            exercises = [q for q in tm.get("linked_questions", []) if q.get("type") == "exercise"]
                            if exercises:
                                for i, ex in enumerate(exercises):
                                    with st.container(border=True):
                                        st.caption(f"🎓 講義名: {ex.get('lecture_name', '未設定')}")
                                        st.markdown(f"**📌 📘 大問 {clean_q_label(ex.get('local_q_num', ''))}**")
                                        st.markdown(ex.get("question_text", ""))
                                        
                                        ex_videos = ex.get("aligned_videos", [])
                                        for idx, v in enumerate(ex_videos):
                                            render_video_item(v, f"action2_ex_{ex.get('global_q_id')}_{tab_idx}_{idx}", is_modeling=True)
                                        
                                        if st.button("🔍 詳しく見る", key=f"action2_btn_{ex.get('global_q_id')}_{tab_idx}_{i}", use_container_width=True):
                                            st.session_state.history.append({
                                                "result": st.session_state.current_result,
                                                "query": st.session_state.display_query,
                                                "image_choices": st.session_state.pending_image_choices,
                                            })
                                            new_query = f"{ex.get('question_text', '')} の解き方"
                                            st.session_state.display_query = new_query
                                            st.session_state.last_clicked_node = None
                                            with st.spinner("問題ルートへ切り替え中..."):
                                                res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                                if res_next:
                                                    st.session_state.current_result = res_next
                                            st.rerun()
                            else:
                                st.write("該当なし")

                            st.markdown("#### 📗 第三アクション (アセスメント: 確認問題演習)")
                            questions = [q for q in tm.get("linked_questions", []) if q.get("type") == "question"]
                            if questions:
                                cols = st.columns(3)
                                for i, q in enumerate(questions):
                                    with cols[i % 3]:
                                        with st.container(border=True):
                                            st.caption(f"🎓 講義名: {q.get('lecture_name', '未設定')}")
                                            st.markdown(f"**📌 📗 確認問題 {clean_q_label(q.get('local_q_num', ''))}**")
                                            st.markdown(q.get("question_text", ""))

                                            if st.button(
                                                "🔍 解き方を見る",
                                                key=f"action3_{q.get('global_q_id')}_{tab_idx}_{i}",
                                                use_container_width=True,
                                            ):
                                                st.session_state.history.append({
                                                    "result": st.session_state.current_result,
                                                    "query": st.session_state.display_query,
                                                    "image_choices": st.session_state.pending_image_choices,
                                                })
                                                new_query = f"{q.get('question_text', '')} の解き方"
                                                st.session_state.display_query = new_query
                                                st.session_state.last_clicked_node = None
                                                with st.spinner("問題ルートへ切り替え中..."):
                                                    res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                                    if res_next:
                                                        st.session_state.current_result = res_next
                                                st.rerun()
                            else:
                                st.write("該当なし")

                # ===============================================
                # 📗 問題・解法ステップ優先ルートの描画 (本バージョンでは既存通り)
                # ===============================================
                else:
                    top = res["top_match"]
                    node = top["node"]
                    title = f"{node.get('lecture_name', '')} {clean_q_label(node.get('local_q_num', ''))}"
                    
                    if res.get("is_drilldown"):
                        st.markdown(f"### {title}")
                    else:
                        display_score = min(top['final_score'], 1.0)
                        base = top['base_score']
                        boost = top['boost_amount']
                        pen_str = f" | {top['penalty_reason']}" if top['penalty_reason'] != "なし (既習・復習範囲)" else ""
                        
                        st.markdown(f"### {title} (総合適合度: {display_score*100:.1f}%)")
                        st.caption(
                            f"📊 **【スコア内訳】** ベース類似度: {base*100:.1f}% "
                            f"| 加点ブースト: +{boost*100:.1f}%{pen_str}"
                        )
                        
                    st.caption(f"🎓 講義名: {node.get('lecture_name', '未設定')}")

                    st.markdown("#### 🧠 ゼロアクション (前提知識の確認)")
                    if res.get("required_concepts_text"):
                        st.info(res["required_concepts_text"])
                    else:
                        st.write("下部のGraph RAGから前提知識を確認してください。")

                    st.markdown(f"#### 📝 本命の問題 ({node.get('lecture_name', '')} {node.get('local_q_num', '')})")
                    st.info(node.get("question_text", ""))
                    st.markdown(f"**🧠 関連概念:** {node.get('matched_concept', '')}")

                    st.divider()

                    st.markdown("#### 🎬 第一アクション (問題の直接解説・モデリング)")
                    edges = node.get("aligned_videos", [])
                    for idx, e in enumerate(edges):
                        render_video_item(e, f"action1_q_{node.get('global_q_id')}_{idx}", is_modeling=True)
                    if not edges:
                        st.write("該当なし")

                    st.markdown("#### 🧬 第二アクション (同概念の横展開演習)")
                    if res.get("connected_questions"):
                        cols = st.columns(3)
                        for i, q in enumerate(res["connected_questions"]):
                            with cols[i % 3]:
                                with st.container(border=True):
                                    q_label = "📘 大問" if q.get("type") == "exercise" else "📗 確認問題"
                                    st.caption(f"🎓 講義名: {q.get('lecture_name', '未設定')}")
                                    st.markdown(f"**📌 {q_label} {clean_q_label(q.get('local_q_num', ''))}**")
                                    st.markdown(q.get("question_text", ""))

                                    if st.button(
                                        "🔍 これも解く",
                                        key=f"hub_{q.get('global_q_id', i)}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.history.append({
                                            "result": st.session_state.current_result,
                                            "query": st.session_state.display_query,
                                            "image_choices": st.session_state.pending_image_choices,
                                        })
                                        new_query = f"{q.get('question_text', '')} の解き方"
                                        st.session_state.display_query = new_query
                                        st.session_state.last_clicked_node = None
                                        with st.spinner("切り替え中..."):
                                            res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                            if res_next:
                                                st.session_state.current_result = res_next
                                        st.rerun()
                    else:
                        st.write("該当なし")

            # 🌟 次点の表示
            if res.get("runner_ups"):
                st.subheader("🥈 次点 (関連概念・類題)")
                cols = st.columns(3)
                for i, r in enumerate(res["runner_ups"]):
                    r_node = r["node"]
                    with cols[i % 3]:
                        with st.container(border=True):
                            if res["intent"] == "concept":
                                st.markdown(f"**🧠 {r_node.get('concept_name', '')}**")
                                st.caption(f"🎓 講義名: {r_node.get('lecture_name', '')}")
                                p_c = r_node.get("parent_concept", "")
                                if p_c:
                                    st.caption(f"🔼 親概念: {p_c}")
                                st.markdown(r_node.get("summary", ""))
                            else:
                                q_label = "📘 大問" if r_node.get("type") == "exercise" else "📗 確認問題"
                                st.caption(f"🎓 講義名: {r_node.get('lecture_name', '未設定')}")
                                st.markdown(f"**📌 {q_label} {clean_q_label(r_node.get('local_q_num', ''))}**")
                                st.markdown(f"**🧠 概念:** {r_node.get('matched_concept', '')}")
                                st.markdown(r_node.get("question_text", ""))

                            st.markdown("<hr style='margin: 0.5em 0;'>", unsafe_allow_html=True)
                            
                            display_r_score = min(r['final_score'], 1.0)
                            r_base = r['base_score']
                            r_boost = r['boost_amount']
                            r_pen_str = f" | {r['penalty_reason']}" if r['penalty_reason'] != "なし (既習・復習範囲)" else ""
                            
                            st.caption(f"📊 総合: {display_r_score*100:.1f}%")
                            st.caption(f"(ベース {r_base*100:.1f}% + ブースト {r_boost*100:.1f}%{r_pen_str})")

                            btn_key = f"btn_runner_{r_node.get('global_c_id', r_node.get('global_q_id', i))}"
                            if st.button("🔍 詳しく見る", key=btn_key, use_container_width=True):
                                st.session_state.history.append({
                                    "result": st.session_state.current_result,
                                    "query": st.session_state.display_query,
                                    "image_choices": st.session_state.pending_image_choices,
                                })
                                if res["intent"] == "concept":
                                    new_query = f"{r_node.get('concept_name', '')} について詳しく知りたい"
                                else:
                                    new_query = f"{r_node.get('question_text', '')} の解き方"
                                st.session_state.display_query = new_query
                                st.session_state.last_clicked_node = None
                                with st.spinner("切り替え中..."):
                                    res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                    if res_next:
                                        st.session_state.current_result = res_next
                                st.rerun()

            st.divider()

            # 🌟 Graph RAG ネットワーク (知識群の視覚的表現)
            if res.get("graph_prerequisites") or res.get("graph_siblings") or res.get("graph_next_steps") or res.get("graph_linked_tasks") or res.get("graph_linked_knowledges"):
                st.subheader("🧭 Graph RAG: オントロジー探索 (学習の繋がり)")
                
                # 概念ルートの場合はトップマッチの1位をGraphの中心として描画
                top_node_ref = res["top_match"]["node"]
                c_name_target = top_node_ref.get("concept_name") if res["intent"] == "concept" else top_node_ref.get("matched_concept", "")
                p_name_target = top_node_ref.get("parent_concept", "未分類")
                
                if c_name_target:
                    graph_nodes = []
                    graph_edges = []
                    node_ids = set()

                    def add_graph_node(nid, label, node_type, tooltip=None, is_current=False):
                        if nid not in node_ids and nid:
                            display_tooltip = tooltip if tooltip else f"👆 クリックして「{label}」を検索"
                            
                            clean_label = clean_math_for_label(label)
                            max_len = 15
                            if clean_label.startswith("親: "):
                                raw_name = clean_label.replace("親: ", "")
                                display_label = "親: " + (raw_name[:12] + "..." if len(raw_name) > 12 else raw_name)
                            else:
                                display_label = clean_label[:max_len] + "..." if len(clean_label) > max_len else clean_label

                            # 🌟 知識群（グループ）を「形」と「色」で表現
                            if is_current:
                                color, shape = "#FFECB3", "star" # 現在地: 星(明るい黄色)
                            elif node_type == "foundation_knowledge":
                                color, shape = "#BBDEFB", "box" # 基礎知識: 四角(薄い青)
                            elif node_type == "perspective_condition":
                                color, shape = "#E1BEE7", "hexagon" # 視点: 六角形(薄い紫)
                            elif node_type == "derived_knowledge":
                                color, shape = "#C8E6C9", "box" # 再構成知識: 四角(薄い緑)
                            elif node_type == "tasks":
                                color, shape = "#B0BEC5", "box" # タスク: 四角(グレー/紺系)
                            else:
                                color, shape = "#E0E0E0", "ellipse" # その他: 楕円(灰)

                            graph_nodes.append(Node(id=nid, label=display_label, size=30, color=color, shape=shape, title=display_tooltip))
                            node_ids.add(nid)

                    target_node_obj = next((c for c in db.get("global_concept_nodes", {}).values() if c.get("concept_name") == c_name_target), {})
                    add_graph_node(c_name_target, c_name_target, target_node_obj.get("type", "unknown"), tooltip="📍 現在地", is_current=True) 
                    
                    if p_name_target and p_name_target != "未分類":
                        add_graph_node(p_name_target, f"親: {p_name_target}", "unknown")
                        graph_edges.append(Edge(source=p_name_target, target=c_name_target, dashes=True, arrows=""))

                    for pre_info in res.get("graph_prerequisites", []):
                        pre_node = pre_info["node"]
                        pre_name = pre_node.get("concept_name")
                        dep_type = pre_info.get("dependency_type", "mandatory")
                        reasoning = pre_info.get("reasoning", "")
                        
                        if pre_name:
                            is_mandatory = (dep_type == "mandatory")
                            badge_str = "🔵 [必須前提]" if is_mandatory else "🟡 [補足前提]"
                            tt_text = f"{badge_str} {pre_name}\n💡 理由: {reasoning}" if reasoning else f"{badge_str} {pre_name}"
                            
                            add_graph_node(pre_name, pre_name, pre_node.get("type", "unknown"), tooltip=tt_text)
                            graph_edges.append(Edge(source=pre_name, target=c_name_target, label=dep_type, dashes=not is_mandatory))

                    for sib in res.get("graph_siblings", []):
                        sib_name = sib.get("concept_name")
                        if sib_name:
                            add_graph_node(sib_name, sib_name, sib.get("type", "unknown"))
                            if p_name_target and p_name_target != "未分類":
                                graph_edges.append(Edge(source=p_name_target, target=sib_name, dashes=True, arrows=""))
                                
                    for nxt in res.get("graph_next_steps", []):
                        nxt_name = nxt.get("concept_name")
                        if nxt_name:
                            add_graph_node(nxt_name, nxt_name, nxt.get("type", "unknown")) 
                            graph_edges.append(Edge(source=c_name_target, target=nxt_name, label="requires", dashes=True))

                    config = Config(
                        width="100%",
                        height=400,
                        directed=True,
                        physics=False,
                        hierarchical={"enabled": True, "direction": "UD", "sortMethod": "directed"},
                    )

                    with st.expander("🗺️ 学習スキルツリーを開く (クリックで探索可能)", expanded=True):
                        st.info("💡 **ヒント**: 気になるノードにカーソルを合わせるか、クリックするとその概念の世界へワープして探索を続けられます！")
                        st.caption("🟦 基礎知識 | 🟪 視点・条件(六角形) | 🟩 再構成知識 | ⬛ タスク(技能) | ⭐️ 現在地")
                        clicked_node_id = agraph(nodes=graph_nodes, edges=graph_edges, config=config)
                        
                        target_to_check = c_name_target if res["intent"] == "concept" else top_node_ref.get("global_q_id")
                        if clicked_node_id and clicked_node_id != target_to_check:
                            if clicked_node_id != st.session_state.last_clicked_node:
                                st.session_state.last_clicked_node = clicked_node_id
                                st.session_state.history.append({
                                    "result": st.session_state.current_result,
                                    "query": st.session_state.display_query,
                                    "image_choices": st.session_state.pending_image_choices,
                                })
                                clean_query_name = clicked_node_id.replace("親: ", "")
                                new_query = f"{clean_query_name} について詳しく知りたい"
                                st.session_state.display_query = new_query
                                with st.spinner(f"「{clean_query_name}」へワープ中..."):
                                    res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                    if res_next:
                                        st.session_state.current_result = res_next
                                st.rerun()

                col_p, col_s, col_n = st.columns(3)
                with col_p:
                    st.markdown("#### ⏪ 遡り学習 (前提)")
                    if res.get("graph_prerequisites"):
                        for pre_info in res["graph_prerequisites"]:
                            p_node = pre_info["node"]
                            dep_type = pre_info.get("dependency_type", "mandatory")
                            reasoning = pre_info.get("reasoning", "")
                            badge_str = "🔵 [必須]" if dep_type == "mandatory" else "🟡 [補足]"
                            
                            with st.expander(f"{badge_str} {p_node.get('concept_name', '')}"):
                                st.caption(f"🔼 親ハブ: {p_node.get('parent_concept', '')}")
                                if reasoning:
                                    st.info(f"💡 **前提となる理由:** {reasoning}")
                                st.write(p_node.get("summary", ""))
                                for idx, v in enumerate(p_node.get("main_videos", []) + p_node.get("review_videos", [])):
                                    render_video_item(v, f"pre_{p_node.get('global_c_id')}_{idx}")
                                if st.button("🔍 学ぶ", key=f"g_pre_{p_node.get('global_c_id')}", use_container_width=True):
                                    st.session_state.history.append({"result": st.session_state.current_result, "query": st.session_state.display_query})
                                    st.session_state.display_query = f"{p_node.get('concept_name', '')} について詳しく知りたい"
                                    st.session_state.last_clicked_node = None
                                    st.session_state.current_result = execute_search_for_ui(st.session_state.display_query, db, is_drilldown=True)
                                    st.rerun()
                    else:
                        st.write("該当なし")

                with col_s:
                    st.markdown("#### ⏩ 横展開 (兄弟)")
                    if res.get("graph_siblings"):
                        for s_node in res["graph_siblings"]:
                            with st.expander(f"🧠 {s_node.get('concept_name', '')}"):
                                st.caption(f"🔼 親ハブ: {s_node.get('parent_concept', '')}")
                                st.write(s_node.get("summary", ""))
                                for idx, v in enumerate(s_node.get("main_videos", []) + s_node.get("review_videos", [])):
                                    render_video_item(v, f"sib_{s_node.get('global_c_id')}_{idx}")
                                
                                if st.button("🔍 学ぶ", key=f"g_sib_{s_node.get('global_c_id')}", use_container_width=True):
                                    st.session_state.history.append({"result": st.session_state.current_result, "query": st.session_state.display_query})
                                    st.session_state.display_query = f"{s_node.get('concept_name', '')} について詳しく知りたい"
                                    st.session_state.last_clicked_node = None
                                    st.session_state.current_result = execute_search_for_ui(st.session_state.display_query, db, is_drilldown=True)
                                    st.rerun()
                    else:
                        st.write("該当なし")
                        
                with col_n:
                    st.markdown("#### ⏭️ 応用先 (Next)")
                    if res.get("graph_next_steps"):
                        for n_node in res["graph_next_steps"]:
                            with st.expander(f"🧠 {n_node.get('concept_name', '')}"):
                                st.caption(f"🔼 親ハブ: {n_node.get('parent_concept', '')}")
                                st.write(n_node.get("summary", ""))
                                for idx, v in enumerate(n_node.get("main_videos", []) + n_node.get("review_videos", [])):
                                    render_video_item(v, f"nxt_{n_node.get('global_c_id')}_{idx}")
                                
                                if st.button("🔍 学ぶ", key=f"g_nxt_{n_node.get('global_c_id')}", use_container_width=True):
                                    st.session_state.history.append({"result": st.session_state.current_result, "query": st.session_state.display_query})
                                    st.session_state.display_query = f"{n_node.get('concept_name', '')} について詳しく知りたい"
                                    st.session_state.last_clicked_node = None
                                    st.session_state.current_result = execute_search_for_ui(st.session_state.display_query, db, is_drilldown=True)
                                    st.rerun()
                    else:
                        st.write("該当なし")

    # ==========================================
    # トップ検索画面
    # ==========================================
    else:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 テキストで検索")
                query = st.text_area(
                    "わからない概念や問題を教えてください",
                    value=st.session_state.display_query,
                    height=100,
                )
            with col2:
                st.subheader("📸 画像で検索 (ノート・プリント)")
                uploaded_image = st.file_uploader(
                    "画像をアップロード", type=["jpg", "jpeg", "png"]
                )

            search_button = st.button(
                "検索を実行 🔍", type="primary", use_container_width=True
            )

        query_applied = False
        if query != st.session_state.display_query:
            st.session_state.display_query = query
            if query.strip() != "":
                query_applied = True

        if search_button or query_applied:
            if uploaded_image:
                with st.spinner("🧠 AIが画像を解析中..."):
                    image_bytes = uploaded_image.getvalue()
                    img_data = analyze_image_with_gemini_json(image_bytes)

                    if len(img_data.get("extracted_items", [])) > 1:
                        st.session_state.pending_image_choices = img_data
                        st.session_state.history = []
                        st.rerun()
                    else:
                        item = (
                            img_data["extracted_items"][0]
                            if img_data.get("extracted_items")
                            else ""
                        )
                        if not item:
                            st.warning(
                                "⚠️ 画像から問題やテキストを読み取れませんでした。"
                            )
                            st.stop()
                        boost = (
                            " の解き方"
                            if img_data.get("image_type") == "PROBLEM"
                            else " について詳しく知りたい"
                        )
                        new_query = f"{item}{boost}"
                        st.session_state.display_query = new_query
                        with st.spinner("検索中..."):
                            res = execute_search_for_ui(new_query, db, is_drilldown=False)
                            if res:
                                st.session_state.current_result = res
                        st.session_state.history = []
                        st.session_state.last_clicked_node = None
                        st.rerun()
            elif query:
                st.session_state.display_query = query
                with st.spinner("🧠 文科省の文脈・クエリ意図を解析中..."):
                    res = execute_search_for_ui(query, db, is_drilldown=False)
                    if res:
                        st.session_state.current_result = res
                st.session_state.history = []
                st.session_state.last_clicked_node = None
                st.rerun()


if __name__ == "__main__":
    main()