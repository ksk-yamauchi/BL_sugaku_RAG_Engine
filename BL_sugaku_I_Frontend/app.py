import json
import os
import re
import time
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
# 🧠 ベクトル化・類似度計算 ＆ 画像解析
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


def clean_q_label(q_num_str):
    if not q_num_str:
        return ""
    return str(q_num_str).replace("確認問題 ", "").replace("確認問題", "").replace("大問 ", "").replace("大問", "").strip()


def format_text_for_markdown(text):
    if not text: 
        return ""
    text = text.replace(r"\\\\", r"\\")
    parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    formatted_parts = []
    for p in parts:
        if p.startswith("$$") and p.endswith("$$"):
            formatted_parts.append(p)
        else:
            formatted_parts.append(p.replace("\n", "  \n"))
    return "".join(formatted_parts)


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


def generate_required_concepts(problem_text, retries=1):
    prompt = f"""
    あなたは優秀な高校数学教師です。以下の生徒が直面している問題（または質問）を解くために必要な「高校数学の概念や公式」を箇条書きで簡潔に提示・解説してください。
    数式はLaTeX形式とし、必ず $ または $$ 記号で囲んで生徒が理解しやすいように要点をまとめてください。
    
    【問題・質問】
    {problem_text}
    """
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.2),
            )
            return response.text
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"Concept Generation Error: {e}")
                return "⚠️ 概念情報の取得に失敗しました。"


def sanitize_query_for_search(raw_query):
    text = raw_query.replace("$", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else raw_query


# =========================================================
# 🔍 検索ロジック ＆ GraphRAG ネットワーク探索 (抽出一元化)
# =========================================================
def execute_search_for_ui(search_query, db, is_drilldown=False, is_image_query=False):
    embed_model = db.get("embed_model", "models/text-embedding-004")
    concept_nodes = db.get("global_concept_nodes", {})
    question_nodes = db.get("global_question_nodes", {})
    video_catalog = db.get("global_video_catalog", {})

    target_id = None
    is_id_query = False
    id_match = re.search(r"\(ID:\s*(.+?)\)$", search_query)
    
    if id_match:
        target_id = id_match.group(1).strip()
        is_id_query = True
        
    target_node = None
    if is_id_query:
        target_node = concept_nodes.get(target_id) or question_nodes.get(target_id)
        if not target_node:
            is_id_query = False 

    ai_explanation = ""
    text_to_embed = ""
    sanitized_query = sanitize_query_for_search(search_query)

    has_math = bool(re.search(r"[\$\^\=]", search_query))
    needs_hyde = (not is_id_query) and (has_math or is_image_query)

    if is_id_query:
        query_vector = target_node.get("concept_vector") or target_node.get("question_vector")
        sanitized_query = target_node.get("concept_name") or target_node.get("question_text", "")
    else:
        if needs_hyde:
            ai_explanation = generate_required_concepts(search_query)
            if ai_explanation and not ai_explanation.startswith("⚠️"):
                text_to_embed = f"{sanitized_query}\n{ai_explanation}"
            else:
                text_to_embed = sanitized_query
        else:
            text_to_embed = sanitized_query
            
        query_vector = get_embedding(text_to_embed, embed_model)

    if not query_vector:
        return None

    raw_concept_results_dict = {}
    for cid, node in concept_nodes.items():
        sim = cosine_similarity(query_vector, node["concept_vector"])
        
        if node.get("type") == "derived_knowledge":
            target_task_node = None
            for rel in ["prerequisite", "applied_to", "requires_logical"]:
                max_task_sim = -1.0
                for edge in node.get("outgoing_edges", {}).get(rel, []):
                    tgt_id = edge.get("target_id")
                    if tgt_id in concept_nodes and concept_nodes[tgt_id].get("type") == "tasks":
                        t_node = concept_nodes[tgt_id]
                        t_sim = cosine_similarity(query_vector, t_node["concept_vector"])
                        if t_sim > max_task_sim:
                            max_task_sim = t_sim
                            target_task_node = t_node
                if target_task_node:
                    break 
            
            if target_task_node:
                promoted_node = target_task_node.copy()
                promoted_node["promoted_from_node"] = node 
                t_id = promoted_node["global_c_id"]
                if t_id not in raw_concept_results_dict or sim > raw_concept_results_dict[t_id]["score"]:
                    raw_concept_results_dict[t_id] = {"id": t_id, "base_score": sim, "score": sim, "node": promoted_node}
            else:
                if cid not in raw_concept_results_dict or sim > raw_concept_results_dict[cid]["score"]:
                    raw_concept_results_dict[cid] = {"id": cid, "base_score": sim, "score": sim, "node": node}
        else:
            if cid not in raw_concept_results_dict or sim > raw_concept_results_dict[cid]["score"]:
                raw_concept_results_dict[cid] = {"id": cid, "base_score": sim, "score": sim, "node": node}

    raw_concept_results = list(raw_concept_results_dict.values())

    question_results = []
    for qid, node in question_nodes.items():
        sim = cosine_similarity(query_vector, node["question_vector"])
        question_results.append({"id": qid, "base_score": sim, "score": sim, "node": node})

    concept_keywords = ["指導", "教え方", "とは", "意味", "概念", "基礎", "仕組み", "について", "学びたい", "知りたい", "わからない", "教えて"]
    if any(kw in sanitized_query for kw in concept_keywords):
        for item in raw_concept_results:
            item["score"] += 0.05

    problem_keywords = ["問題", "解き方", "問", "例題", "演習", "確認問題", "解法", "計算", "を展開"]
    if any(kw in sanitized_query for kw in problem_keywords):
        for item in question_results:
            item["score"] += 0.05

    for item in question_results:
        if item["node"].get("type") == "question":
            item["score"] += 0.05

    stop_words = ["について", "指導したい", "教えて", "解き方", "解法", "とは", "意味", "概念", "基礎", "仕組み", "学びたい", "知りたい", "わからない", "問題", "ありますか", "探して", "の", "指導", "ドリル", "テスト"]
    clean_query = sanitized_query
    for w in stop_words:
        clean_query = clean_query.replace(w, " ")
    
    keywords = [kw.strip() for kw in clean_query.split() if kw.strip()]
    if not keywords:
        keywords = [sanitized_query.strip()]
    
    if not is_id_query:
        for item in raw_concept_results:
            node = item["node"]
            target_text = f"{node.get('concept_name', '')} {node.get('parent_concept', '')} {node.get('summary', '')}"
            if "promoted_from_node" in node:
                p_node = node["promoted_from_node"]
                target_text += f" {p_node.get('concept_name', '')} {p_node.get('parent_concept', '')} {p_node.get('summary', '')}"
            if any(len(kw) >= 2 and kw in target_text for kw in keywords):
                item["score"] += 0.30

        for item in question_results:
            node = item["node"]
            target_text = f"{node.get('question_text', '')} {node.get('matched_concept', '')}"
            if any(len(kw) >= 2 and kw in target_text for kw in keywords):
                item["score"] += 0.30
                
    if is_id_query and target_id in question_nodes:
        q_node = question_nodes[target_id]
        t_names = q_node.get("linked_task_names", [])
        k_names = q_node.get("linked_knowledge_names", [])
        c_name = t_names[0] if t_names else (k_names[0] if k_names else None)
        if c_name:
            for r in raw_concept_results:
                if r["node"].get("concept_name") == c_name:
                    r["score"] += 100.0

    raw_concept_results.sort(key=lambda x: x["score"], reverse=True)
    question_results.sort(key=lambda x: x["score"], reverse=True)

    top_concept_score = raw_concept_results[0]["score"] if raw_concept_results else 0.0
    top_question_score = question_results[0]["score"] if question_results else 0.0

    if is_id_query:
        is_concept_intent = (target_id in concept_nodes)
    else:
        is_concept_intent = top_concept_score >= top_question_score
        explicit_problem_kws = ["問題", "演習", "ドリル", "テスト", "解き方", "解法"]
        if any(kw in search_query for kw in explicit_problem_kws):
            is_concept_intent = False
        elif any(kw in search_query for kw in ["とは", "意味", "教えて", "概念", "仕組み"]):
            is_concept_intent = True

    result_data = {
        "intent": "concept" if is_concept_intent else "question",
        "top_match": None,
        "top_matches": [],  
        "runner_ups": [],
        "required_concepts_text": ai_explanation,
        "is_drilldown": is_drilldown,
        "sanitized_query": None if is_id_query else (sanitized_query if sanitized_query != search_query else None)
    }

    if is_concept_intent:
        target_list = raw_concept_results
    else:
        filtered_list = [r for r in raw_concept_results if r["node"].get("type") == "tasks"]
        target_list = filtered_list if filtered_list else raw_concept_results

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

        if top_time_idx is not None and this_time_idx is not None and this_time_idx > top_time_idx:
            if node.get("lecture_name") == top_raw["node"].get("lecture_name"):
                penalty_reason = "同じ講義内の先のステップ"
            else:
                penalty = (this_time_idx - top_time_idx) * PENALTY_WEIGHT
                final_score -= penalty
                penalty_reason = f"⚠️ 未習範囲減点 (-{penalty*100:.1f}%)"

        processed_results.append({
            "final_score": final_score, "base_score": base_sim, "boost_amount": boost_amount,
            "node": node, "penalty_reason": penalty_reason,
        })

    processed_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    if is_id_query:
        top_matches = []
        for r in processed_results:
            nid = r["node"].get("global_c_id") or r["node"].get("global_q_id")
            promoted_from = r["node"].get("promoted_from_node", {}).get("global_c_id")
            if nid == target_id or promoted_from == target_id:
                top_matches.append(r)
                break
        if not top_matches:
            top_matches = [processed_results[0]]
        runner_ups = []
    else:
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

    def get_concept_graph_data(target_name):
        prereqs, sibs, nxts = [], [], []
        if not target_name: return prereqs, sibs, nxts
        target_c_node = next((c for c in concept_nodes.values() if c["concept_name"] == target_name), None)
        if target_c_node:
            prereqs_raw = target_c_node.get("prerequisite_concepts", [])
            for p_item in prereqs_raw:
                p_name = p_item.get("concept_name", "") if isinstance(p_item, dict) else str(p_item)
                p_type = p_item.get("dependency_type", "mandatory") if isinstance(p_item, dict) else "mandatory"
                p_reason = p_item.get("reasoning", "") if isinstance(p_item, dict) else ""
                if not p_name: continue
                for cid, c_node in concept_nodes.items():
                    if c_node["concept_name"] == p_name:
                        if not any(x["node"]["global_c_id"] == c_node["global_c_id"] for x in prereqs):
                            prereqs.append({"node": c_node, "dependency_type": p_type, "reasoning": p_reason, "prereq_name": p_name})
            parent_name = target_c_node.get("parent_concept")
            if parent_name and parent_name != "未分類":
                for cid, c_node in concept_nodes.items():
                    if c_node.get("parent_concept") == parent_name and c_node["concept_name"] != target_name:
                        sibs.append(c_node)
            for cid, c_node in concept_nodes.items():
                p_list = c_node.get("prerequisite_concepts_list", [])
                if not p_list:
                    p_raw = c_node.get("prerequisite_concepts", [])
                    p_list = [p["concept_name"] if isinstance(p, dict) else p for p in p_raw]
                if target_name in p_list:
                    nxts.append(c_node)
        return prereqs, sibs, nxts

    # 🌟 メタタスク判定関数（Ver 4.19.3 強化版：真の再構成知識判定を導入）
    def check_is_meta_task(node_obj):
        # 内部関数: 再構成知識ノードが「真のメタタスク条件（subsumes以外を持つ）」を満たすかチェック
        def is_true_derived_knowledge(dk_node):
            if not dk_node: 
                return False
            incoming = dk_node.get("incoming_edges", {})
            for rel_type, edge_list in incoming.items():
                if rel_type != "subsumes" and len(edge_list) > 0:
                    return True
            return False

        p_raw = node_obj.get("prerequisite_concepts", [])
        for p_item in p_raw:
            p_name = p_item.get("concept_name", "") if isinstance(p_item, dict) else str(p_item)
            if not p_name: continue
            for cid, c_node in concept_nodes.items():
                if c_node.get("concept_name") == p_name and c_node.get("type") == "derived_knowledge":
                    if is_true_derived_knowledge(c_node):
                        return True
        
        incoming = node_obj.get("incoming_edges", {})
        for rel_type, edge_list in incoming.items():
            for edge_item in edge_list:
                src_id = edge_item.get("source_id")
                if src_id in concept_nodes and concept_nodes[src_id].get("type") == "derived_knowledge":
                    if is_true_derived_knowledge(concept_nodes[src_id]):
                        return True
        return False

    # =========================================================
    # 🌟 抽出ロジックの一元化 (MVC分離)
    # ルートの判定に関わらず、すべてのトップ候補に対して
    # 共通の純化ロジックで「動画」「大問」「確認問題」「前提問題」を生成・格納する
    # =========================================================
    
    # 🌟 [問題・解法ステップ優先ルート用] すべての本命タスクの前提概念(タスクのみ)を収集し、OKリストを作成
    all_kanban_names = set()
    all_valid_prereq_task_names = set()
    global_prereq_infos = []
    
    if not is_concept_intent:
        for tm in top_matches:
            c_name = tm["node"].get("concept_name", "")
            all_kanban_names.add(c_name)
            prereqs, _, _ = get_concept_graph_data(c_name)
            for pre_info in prereqs:
                p_node = pre_info["node"]
                if p_node.get("type") == "tasks":
                    p_name = pre_info["prereq_name"]
                    if p_name not in all_valid_prereq_task_names:
                        all_valid_prereq_task_names.add(p_name)
                        global_prereq_infos.append(pre_info)

        # 🌟 GNNトラバーサル: 統合された前提タスクから確認問題を逆引き抽出
        global_prereq_qs = []
        for pre_info in global_prereq_infos:
            p_name = pre_info["prereq_name"]

            for q in question_nodes.values():
                if any(k_name in q.get("linked_task_names", []) for k_name in all_kanban_names):
                    continue

                if q.get("type") == "question" and p_name in q.get("linked_task_names", []):
                    
                    is_valid = True
                    for t_name in q.get("linked_task_names", []):
                        if t_name in all_valid_prereq_task_names:
                            continue 
                        
                        t_node = next((c for c in concept_nodes.values() if c.get("concept_name") == t_name), None)
                        if t_node:
                            has_main_task_as_prereq = False
                            for p_item in t_node.get("prerequisite_concepts", []):
                                p_name_check = p_item.get("concept_name", "") if isinstance(p_item, dict) else str(p_item)
                                if p_name_check in all_kanban_names:
                                    has_main_task_as_prereq = True
                                    break
                            
                            if not has_main_task_as_prereq:
                                for rel_type, edge_list in t_node.get("incoming_edges", {}).items():
                                    for edge_item in edge_list:
                                        src_id = edge_item.get("source_id")
                                        if src_id in concept_nodes and concept_nodes[src_id].get("concept_name") in all_kanban_names:
                                            has_main_task_as_prereq = True
                                            break
                                    if has_main_task_as_prereq:
                                        break
                                        
                            if has_main_task_as_prereq:
                                is_valid = False
                                break

                            if check_is_meta_task(t_node):
                                is_valid = False
                                break
                    
                    if not is_valid:
                        continue

                    global_prereq_qs.append({
                        "node": q,
                        "matched_task_names": [p_name],
                        "final_score": pre_info["node"].get("global_timeline_index", 0),
                        "base_score": 0.0,
                        "boost_amount": 0.0,
                        "penalty_reason": "なし (既習・復習範囲)"
                    })
                    
        # 重複排除
        seen_q_ids = set()
        unique_global_prereq_qs = []
        for pq in global_prereq_qs:
            qid = pq["node"].get("global_q_id")
            if qid not in seen_q_ids:
                seen_q_ids.add(qid)
                unique_global_prereq_qs.append(pq)

    for tm in top_matches:
        c_node = tm["node"]
        c_name = c_node.get("concept_name", "")
        c_type = c_node.get("type", "")
        is_promoted = "promoted_from_node" in c_node
        
        # 1. メタタスク判定
        meta_task_flag = False
        if c_type == "tasks":
            meta_task_flag = check_is_meta_task(c_node)

        tm["kanban_concept_name"] = c_name
        tm["kanban_concept_node"] = c_node

        # 2. 純粋なインプット・復習動画の抽出（exercise_walkthroughを除外）
        all_videos = c_node.get("main_videos", []) + c_node.get("review_videos", [])
        pure_videos = []
        for v in all_videos:
            v_file = v.get("video_file")
            v_role = video_catalog.get(v_file, {}).get("role", "")
            if v_role != "exercise_walkthrough":
                pure_videos.append(v)
        tm["pure_concept_videos"] = pure_videos

        # 3. 大問と確認問題の抽出（純化フィルター＋2段階ソート適用）
        m_exs = []
        m_qs = []
        for qid, q in question_nodes.items():
            t_names = q.get("linked_task_names", [])
            k_names = q.get("linked_knowledge_names", [])
            
            if c_name in t_names or c_name in k_names:
                # 🌟 純化フィルター（他タスクのメタ判定による動的除外）
                if c_type == "tasks" and not is_promoted and not meta_task_flag:
                    other_tasks = [t for t in t_names if t != c_name]
                    is_invalid_composite = False
                    for ot_name in other_tasks:
                        ot_node = next((n for n in concept_nodes.values() if n.get("concept_name") == ot_name), None)
                        if ot_node:
                            if check_is_meta_task(ot_node):
                                is_invalid_composite = True
                                break
                    
                    if is_invalid_composite:
                        continue
                        
                if q.get("type") == "exercise":
                    m_exs.append(q)
                elif q.get("type") == "question":
                    m_qs.append(q)

        # ベクトルソート
        if query_vector is not None and len(query_vector) > 0 and not is_id_query:
            m_qs.sort(key=lambda q: cosine_similarity(query_vector, q.get("question_vector", [])), reverse=True)
            m_exs.sort(key=lambda q: cosine_similarity(query_vector, q.get("question_vector", [])), reverse=True)

        # IDワープ指定時は強制トップ
        if is_id_query and target_id in question_nodes:
            m_qs.sort(key=lambda q: 0 if q["global_q_id"] == target_id else 1)
            m_exs.sort(key=lambda q: 0 if q["global_q_id"] == target_id else 1)

        tm["pure_modeling_exercises"] = m_exs
        tm["pure_assessment_questions"] = m_qs

        # 4. Graph RAG データの取得
        prereqs, sibs, nxts = get_concept_graph_data(c_name)
        tm["graph_prerequisites"] = prereqs
        tm["graph_siblings"] = sibs
        tm["graph_next_steps"] = nxts

        # 5. 前提確認問題の格納
        if not is_concept_intent:
            tm["prereq_questions"] = unique_global_prereq_qs
        else:
            tm["prereq_questions"] = []

    return result_data


# 🌟 解答・解説の分割表示ヘルパー関数
def render_answer_explanation(answer_text):
    if not answer_text: return
    parts = re.split(r'\[解説\]|【解説】', answer_text)
    ans_part = parts[0].replace('[正解]', '').replace('【正解】', '').strip()
    exp_part = parts[1].strip() if len(parts) > 1 else ""
    if ans_part or exp_part:
        with st.expander("💡 解答・解説を見る"):
            if ans_part:
                st.markdown(f"**【解答】**\n{format_text_for_markdown(ans_part)}")
            if exp_part:
                st.markdown(f"**【解説】**\n{format_text_for_markdown(exp_part)}")


# 🌟 【UI共通化】動画アイテム（主従対応版）を描画するヘルパー関数
def render_video_item(v, unique_key, is_modeling=False):
    with st.container(border=True):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            a_type = v.get("alignment_type", "")
            if a_type in ["concept_introduction", "task_walkthrough", "direct_explanation"]:
                badge = "💡 【解説】" if is_modeling else "💡 【メイン解説】"
            elif a_type == "prerequisite_review":
                badge = "⏪ 【前提・復習】"
            else:
                badge = "🎬 【解説動画】"

            st.markdown(f"{badge} **{v.get('video_file')}** (`{v.get('start_time')}`〜)")
            if not is_modeling and v.get('reasoning'):
                st.markdown(f"**🤔 なぜこの動画？:** {v.get('reasoning')}")
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
                st.session_state.target_start_time = v.get("start_time", "00:00")
                st.rerun()

# =========================================================
# 🎨 画面描画 (UI)
# =========================================================
def main():
    st.title("🚀 スタサプRAG ナレッジエンジン")

    db = load_db()
    if not db:
        st.error("❌ データベースが見つかりません。先にバックエンドでDBを構築してください。")
        return

    st.sidebar.markdown(f"**⚙️ エンジンバージョン:**\n`{db.get('metadata', {}).get('engine_version', 'バージョン情報なし')}`")
    st.sidebar.markdown(f"**📱 UI バージョン:**\n`AIチューター UI Ver 4.19.3`")

    for key in ["history", "current_result", "display_query", "pending_image_choices", "last_clicked_node", "selected_video"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "history" else None

    # ==========================================
    # 🎬 タイムスタンプ（チャプター）表示 UI
    # ==========================================
    if st.session_state.get("selected_video"):
        v_file = st.session_state.selected_video
        v_start = st.session_state.get("target_start_time", "00:00")
        catalog = db.get("global_video_catalog", {})
        v_data = catalog.get(v_file)
        
        col_btn, _ = st.columns([1, 5])
        with col_btn:
            if st.button("🔙 検索結果に戻る", use_container_width=True):
                st.session_state.selected_video = None
                st.session_state.target_start_time = None
                st.rerun()
                
        st.markdown("<div style='background-color: #2e5c9e; padding: 12px; color: white; font-size: 1.2em; font-weight: bold;'>宿題配信</div>", unsafe_allow_html=True)
        st.markdown("<div style='background-color: #f0f2f5; padding: 8px; border-bottom: 1px solid #ccc;'><b>STEP1. 講座選択</b> &nbsp;&nbsp;|&nbsp;&nbsp; <span style='color: #666;'>STEP2. 配信設定</span></div>", unsafe_allow_html=True)
        st.write("")

        col_sidebar, col_main = st.columns([1, 3])

        with col_sidebar:
            with st.container(border=True):
                st.markdown("**[新版] ベーシックレベル数学Ⅰ**\n\n<span style='font-size: 0.8em; color: gray;'>60講義</span>", unsafe_allow_html=True)
                with st.expander("第1講 式の計算と展開", expanded=True):
                    st.checkbox("PART1 単項式と多項式", disabled=True)
                    st.checkbox("PART2 整式の整理", value=True, disabled=True)
                    st.checkbox("PART3 整式の加法と減法", disabled=True)
                    st.checkbox("PART4 展開", disabled=True)
                with st.expander("第2講 因数分解", expanded=False):
                    st.write("...")
                with st.expander("第3講 実数", expanded=False):
                    st.write("...")

        with col_main:
            lecture_name = v_data.get('lecture_name', '未設定') if v_data else '未設定'
            st.markdown(f"<div style='font-size: 1.1em; font-weight: bold; color: #333;'>[新版] ベーシックレベル数学Ⅰ<br>{lecture_name}</div>", unsafe_allow_html=True)
            st.write("")
            
            with st.container(border=True):
                st.markdown(f"**▶️ 講義動画** <span style='font-size: 0.8em; color: gray; margin-left: 10px;'>13分30秒 2チャプター</span>", unsafe_allow_html=True)
                
                col_vid, col_chap = st.columns([4, 1])
                
                with col_vid:
                    start_sec = 0
                    try:
                        m, s = v_start.split(':')
                        start_sec = int(m) * 60 + int(s)
                    except:
                        pass
                    st.video("[https://www.w3schools.com/html/mov_bbb.mp4](https://www.w3schools.com/html/mov_bbb.mp4)", start_time=start_sec)

                with col_chap:
                    st.markdown("<div style='font-size: 0.9em; font-weight: bold; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-bottom: 10px;'>チャプター</div>", unsafe_allow_html=True)
                    st.markdown("<div style='background-color: #e6f3ff; padding: 8px; border-radius: 4px; margin-bottom: 5px;'>1 🎥 00:04:32</div>", unsafe_allow_html=True)
                    st.markdown("<div style='padding: 8px; border-radius: 4px; margin-bottom: 5px;'>2 🎥 00:08:58</div>", unsafe_allow_html=True)

            st.markdown("#### 📑 動画内タイムスタンプ (DB抽出)")
            st.info("※DBから該当動画のセグメント情報を抽出し、再生箇所をピンポイントで指定できる想定のUIです。")
            if v_data:
                target_idx = 0
                for i, seg in enumerate(v_data.get("segments", [])):
                    if seg.get('start_time') == v_start:
                        target_idx = i
                        break
                        
                for idx, seg in enumerate(v_data.get("segments", [])):
                    s_time = seg.get('start_time', '00:00')
                    topic = seg.get('topic', '無題')
                    with st.expander(f"⏱️ {s_time} 〜 | 📌 {topic}", expanded=(idx == target_idx)):
                        col_ts1, col_ts2 = st.columns([4, 1])
                        with col_ts1:
                            st.markdown(f"**💬 概要:**\n{seg.get('explanation_summary', 'データなし')}")
                        with col_ts2:
                            if st.button("▶️ ここから再生", key=f"play_ts_{idx}", use_container_width=True):
                                st.session_state.target_start_time = s_time
                                st.rerun()
            else:
                st.warning("⚠️ この動画の詳細なチャプターカタログデータがデータベースに見つかりません。")
        return  


    # ==========================================
    # 📸 画像アップロード複数候補 UI
    # ==========================================
    if st.session_state.pending_image_choices:
        st.info("📸 画像から複数の項目が検出されました。学習したい問題を1つ選択してください。")
        img_data = st.session_state.pending_image_choices
        choices = img_data.get("extracted_items", [])
        img_type = img_data.get("image_type", "PROBLEM")

        with st.container(border=True):
            cols = st.columns(min(len(choices), 3))
            for i, item in enumerate(choices):
                with cols[i % 3]:
                    st.markdown(f"**候補 {i+1}:**")
                    st.markdown(item)
                    if st.button("🔍 この問題を検索", key=f"choice_{i}", use_container_width=True):
                        st.session_state.history.append({
                            "result": st.session_state.current_result,
                            "query": st.session_state.display_query,
                            "image_choices": st.session_state.pending_image_choices,
                        })
                        boost = " の解き方" if img_type == "PROBLEM" else " について詳しく知りたい"
                        new_query = f"{item}{boost}"
                        st.session_state.display_query = new_query
                        with st.spinner("検索中..."):
                            res = execute_search_for_ui(new_query, db, is_drilldown=False, is_image_query=True)
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
        displayed_q_ids = set()
        
        display_text_clean = re.sub(r"\s*\(ID:.+?\)$", "", st.session_state.display_query) if st.session_state.display_query else ""
        st.info(f"🔍 現在の学習テーマ: **{display_text_clean}**")
        
        if res.get("sanitized_query") and res.get("sanitized_query") != display_text_clean:
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
            
            if res.get("required_concepts_text") and res.get("intent") == "question":
                with st.chat_message("assistant"):
                    st.markdown("**💡 AIチューターからのアプローチ解説**")
                    st.markdown(format_text_for_markdown(res["required_concepts_text"]))
            
            st.subheader(f"🎯 第一候補: {intent_label}")
            
            top_matches = res["top_matches"]
            
            # ===============================================
            # 📘 概念インプット優先ルート (View)
            # ===============================================
            if res["intent"] == "concept":
                if len(top_matches) > 1:
                    tab_names = []
                    for m in top_matches:
                        if "promoted_from_node" in m["node"]:
                            t_name = f"{m['node']['promoted_from_node'].get('concept_name', '概念')} → {m['node'].get('concept_name', 'タスク')}"
                        else:
                            t_name = m["node"].get("concept_name", "概念")
                        tab_names.append(t_name)
                    
                    selected_tab_name = st.radio("🧠 表示する概念を選択してください:", tab_names, horizontal=True)
                    selected_idx = tab_names.index(selected_tab_name)
                    tm = top_matches[selected_idx]
                    tab_idx = selected_idx
                else:
                    tm = top_matches[0]
                    tab_idx = 0
                    
                with st.container(border=True):
                    node = tm["node"]
                    title = node.get("concept_name")
                    display_score = min(tm['final_score'], 1.0)
                    base = tm['base_score']
                    boost = tm['boost_amount']
                    pen_str = f" | {tm['penalty_reason']}" if tm['penalty_reason'] != "なし (既習・復習範囲)" else ""
                    
                    if node.get("type") == "derived_knowledge" and "promoted_from_node" not in node:
                        st.warning("⚠️ **【システム警告】** この再構成知識には、出力先となる「タスク（技能）」がオントロジー上で定義されていません。データの抽出漏れや構造の不備が疑われます。")

                    if "promoted_from_node" in node:
                        orig_name = node["promoted_from_node"].get("concept_name", "再構成知識")
                        st.markdown(f"#### {orig_name}")
                        if res.get("is_drilldown"):
                            st.markdown(f"### → {title}")
                        else:
                            st.markdown(f"### → {title} (総合適合度: {display_score*100:.1f}%)")
                            st.caption(f"📊 **【スコア内訳】** ベース類似度: {base*100:.1f}% | 加点ブースト: +{boost*100:.1f}%{pen_str}")
                    else:
                        if res.get("is_drilldown"):
                            st.markdown(f"### {title}")
                        else:
                            st.markdown(f"### {title} (総合適合度: {display_score*100:.1f}%)")
                            st.caption(f"📊 **【スコア内訳】** ベース類似度: {base*100:.1f}% | 加点ブースト: +{boost*100:.1f}%{pen_str}")
                        
                    st.caption(f"🎓 講義名: {node.get('lecture_name', '未設定')}")
                    st.info(f"**💡 概念要約:** {node.get('summary', '要約なし')}")
                    p_concept = node.get("parent_concept", "未分類")
                    comp_str = node.get("pillar", "未設定")
                    st.markdown(f"**🔼 親概念 (Level 3):** `{p_concept}` | **🏷️ 観点:** `{comp_str}`")
                    
                    with st.expander(f"🏛️ 指導要領: {node.get('mext_hierarchy', '未割り当て')} (コード: {node.get('mext_code', 'N/A')})"):
                        st.markdown(f"**【公式テキスト】** {node.get('mext_official_text', '情報なし')}")
                        st.markdown(f"**【解説要約】** {node.get('mext_explanation', '情報なし')}")

                    st.divider()

                    # 🎬 Action 1: 概念インプット講義
                    st.markdown("#### 🎬 第一アクション (概念インプット講義)")
                    input_videos = tm.get("pure_concept_videos", [])
                    if input_videos:
                        for idx, v in enumerate(input_videos):
                            render_video_item(v, f"action1_{node.get('global_c_id')}_{tab_idx}_{idx}", is_modeling=False)
                    else:
                        st.write("該当なし")

                    # 📘 Action 2: モデリング
                    st.markdown("#### 📘 第二アクション (モデリング: 大問・例題解説)")
                    exercises = tm.get("pure_modeling_exercises", [])
                    if exercises:
                        for i, ex in enumerate(exercises):
                            with st.container(border=True):
                                ex_q_id = ex.get("global_q_id")
                                displayed_q_ids.add(ex_q_id)
                                q_num_str = ex.get('local_q_num', '')
                                cleaned_num = clean_q_label(q_num_str)
                                st.caption(f"🎓 講義名: {ex.get('lecture_name', '未設定')}")
                                st.markdown(f"**📌 📘 大問 {cleaned_num}**")
                                q_text = format_text_for_markdown(ex.get("question_text", ""))
                                st.markdown(q_text)
                                render_answer_explanation(ex.get("answer_text", ""))
                                ex_videos = ex.get("aligned_videos", [])
                                for idx, v in enumerate(ex_videos):
                                    render_video_item(v, f"action2_ex_{ex_q_id}_{tab_idx}_{idx}", is_modeling=True)
                    else:
                        st.write("該当なし")

                    # 📗 Action 3: アセスメント
                    st.markdown("#### 📗 第三アクション (アセスメント: 確認問題演習)")
                    questions = tm.get("pure_assessment_questions", [])
                    if questions:
                        cols = st.columns(3)
                        for i, q in enumerate(questions):
                            with cols[i % 3]:
                                with st.container(border=True):
                                    q_id_val = q.get("global_q_id")
                                    displayed_q_ids.add(q_id_val)
                                    q_num_str = q.get('local_q_num', '')
                                    cleaned_num = clean_q_label(q_num_str)
                                    st.caption(f"🎓 講義名: {q.get('lecture_name', '未設定')}")
                                    st.markdown(f"**📌 📗 確認問題 {cleaned_num}**")
                                    q_text = format_text_for_markdown(q.get("question_text", ""))
                                    st.markdown(q_text)
                                    render_answer_explanation(q.get("answer_text", ""))
                    else:
                        st.write("該当なし")

                # 🥈 次点
                st.subheader("🥈 次点 (前提となる概念)")
                prereq_concepts = tm.get("graph_prerequisites", [])
                if prereq_concepts:
                    cols = st.columns(3)
                    drawn_count = 0
                    for pre_info in prereq_concepts:
                        p_node = pre_info["node"]
                        p_q_id = p_node.get("global_c_id")
                        
                        if p_node.get("type") in ["derived_knowledge", "perspective_condition"]:
                            continue

                        dep_type = pre_info.get("dependency_type", "mandatory")
                        reasoning = pre_info.get("reasoning", "")
                        prefix = "必須： " if dep_type == "mandatory" else "補足： "
                        
                        with cols[drawn_count % 3]:
                            with st.expander(f"{prefix}{p_node.get('concept_name', '')}"):
                                st.caption(f"🎓 講義名: {p_node.get('lecture_name', '未設定')}")
                                p_c = p_node.get("parent_concept", "")
                                if p_c:
                                    st.caption(f"🔼 親概念: {p_c}")
                                st.write(p_node.get("summary", ""))
                                if reasoning:
                                    st.info(f"💡 **前提となる理由:** {reasoning}")
                                st.markdown("<hr style='margin: 0.5em 0;'>", unsafe_allow_html=True)
                                for idx, v in enumerate(p_node.get("main_videos", []) + p_node.get("review_videos", [])):
                                    render_video_item(v, f"pre_runner_{p_q_id}_{tab_idx}_{idx}")
                                btn_key = f"btn_runner_c_{p_q_id}_{tab_idx}_{drawn_count}"
                                if st.button("🔍 この概念について深く学ぶ", key=btn_key, use_container_width=True):
                                    st.session_state.history.append({
                                        "result": st.session_state.current_result, "query": st.session_state.display_query,
                                        "image_choices": st.session_state.pending_image_choices,
                                    })
                                    new_query = f"{p_node.get('concept_name', '')} について詳しく知りたい (ID:{p_q_id})"
                                    st.session_state.display_query = new_query
                                    st.session_state.last_clicked_node = None
                                    with st.spinner("切り替え中..."):
                                        res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                        if res_next: st.session_state.current_result = res_next
                                    st.rerun()
                        drawn_count += 1
                    if drawn_count == 0:
                        with st.container(border=True): st.write("該当なし")
                else:
                    with st.container(border=True): st.write("該当なし")

                st.divider()

                # 🧭 Graph RAG
                st.subheader("🧭 Graph RAG: オントロジー探索 (学習の繋がり)")
                graph_nodes, graph_edges = [], []
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

                        if is_current and node_type == "problem": color, shape = "#FFD54F", "star"
                        elif is_current: color, shape = "#FFECB3", "star"
                        elif node_type == "foundation_knowledge": color, shape = "#BBDEFB", "box"
                        elif node_type == "perspective_condition": color, shape = "#E1BEE7", "hexagon"
                        elif node_type == "derived_knowledge": color, shape = "#C8E6C9", "box"
                        elif node_type == "tasks": color, shape = "#B0BEC5", "box"
                        else: color, shape = "#E0E0E0", "ellipse"
                        graph_nodes.append(Node(id=nid, label=display_label, size=30, color=color, shape=shape, title=display_tooltip))
                        node_ids.add(nid)

                c_name_target = tm["node"].get("concept_name")
                if c_name_target:
                    p_name_target = tm["node"].get("parent_concept", "未分類")
                    target_node_obj = next((c for c in db.get("global_concept_nodes", {}).values() if c.get("concept_name") == c_name_target), {})
                    add_graph_node(c_name_target, c_name_target, target_node_obj.get("type", "unknown"), tooltip=f"📍 現在地：{c_name_target}", is_current=True) 
                    added_edges = set()
                    promoted_node = tm["node"].get("promoted_from_node")
                    if promoted_node:
                        p_name = promoted_node.get("concept_name")
                        add_graph_node(p_name, p_name, promoted_node.get("type", "derived_knowledge"), tooltip="🚀 検索トリガー (再構成知識)")
                        is_in_prereq = any(p.get("node", {}).get("concept_name") == p_name for p in tm.get("graph_prerequisites", []))
                        if not is_in_prereq:
                            graph_edges.append(Edge(source=p_name, target=c_name_target, label="トリガー", dashes=True, color="#FF9800", width=2))
                            added_edges.add((p_name, c_name_target))
                    if p_name_target and p_name_target != "未分類":
                        add_graph_node(p_name_target, f"親: {p_name_target}", "unknown")
                        graph_edges.append(Edge(source=p_name_target, target=c_name_target, dashes=True, arrows="", width=3))
                    for pre_info in tm.get("graph_prerequisites", []):
                        pre_node = pre_info["node"]
                        pre_name = pre_node.get("concept_name")
                        dep_type = pre_info.get("dependency_type", "mandatory")
                        reasoning = pre_info.get("reasoning", "")
                        if pre_name:
                            is_mandatory = (dep_type == "mandatory")
                            badge_str = "🔵 [必須前提]" if is_mandatory else "🟡 [補足前提]"
                            tt_text = f"{badge_str} {pre_name}\n💡 理由: {reasoning}" if reasoning else f"{badge_str} {pre_name}"
                            add_graph_node(pre_name, pre_name, pre_node.get("type", "unknown"), tooltip=tt_text)
                            edge_label = "必須" if is_mandatory else "補足"
                            edge_color = None
                            edge_width = None
                            if promoted_node and pre_name == promoted_node.get("concept_name"):
                                edge_label = f"トリガー ({edge_label})"
                                edge_color = "#FF9800"
                                edge_width = 2
                            if (pre_name, c_name_target) not in added_edges:
                                kwargs = {"source": pre_name, "target": c_name_target, "label": edge_label, "dashes": not is_mandatory}
                                if edge_color: kwargs["color"] = edge_color
                                if edge_width: kwargs["width"] = edge_width
                                graph_edges.append(Edge(**kwargs))
                                added_edges.add((pre_name, c_name_target))
                    for sib in tm.get("graph_siblings", []):
                        sib_name = sib.get("concept_name")
                        if sib_name:
                            add_graph_node(sib_name, sib_name, sib.get("type", "unknown"))
                            if p_name_target and p_name_target != "未分類":
                                graph_edges.append(Edge(source=p_name_target, target=sib_name, dashes=True, arrows="", color="#BBDEFB", length=200))
                    for nxt in tm.get("graph_next_steps", []):
                        nxt_name = nxt.get("concept_name")
                        if nxt_name:
                            add_graph_node(nxt_name, nxt_name, nxt.get("type", "unknown")) 
                            graph_edges.append(Edge(source=c_name_target, target=nxt_name, label="必要", dashes=True))

                config = Config(width="100%", height=400, directed=True, physics=False, hierarchical={"enabled": True, "direction": "UD", "sortMethod": "directed"})
                with st.expander("🗺️ 学習スキルツリーを開く (クリックで探索可能)", expanded=True):
                    st.info("💡 **ヒント**: 気になるノードにカーソルを合わせるか、クリックするとその概念の世界へワープして探索を続けられます！")
                    st.caption("🟦 基礎知識 | 🟪 視点・条件(六角形) | 🟩 再構成知識 | ⬛ タスク(技能) | ⭐️ 現在地")
                    clicked_node_id = agraph(nodes=graph_nodes, edges=graph_edges, config=config)
                    target_to_check = c_name_target
                    if clicked_node_id and clicked_node_id != target_to_check:
                        if clicked_node_id != st.session_state.last_clicked_node:
                            st.session_state.last_clicked_node = clicked_node_id
                            st.session_state.history.append({
                                "result": st.session_state.current_result, "query": st.session_state.display_query,
                                "image_choices": st.session_state.pending_image_choices,
                            })
                            tgt_n = db.get("global_concept_nodes", {}).get(clicked_node_id) or db.get("global_question_nodes", {}).get(clicked_node_id)
                            c_n_name = tgt_n.get("concept_name", "") if tgt_n else clicked_node_id.replace("親: ", "")
                            new_query = f"{c_n_name} について詳しく知りたい (ID:{clicked_node_id})"
                            st.session_state.display_query = new_query
                            with st.spinner(f"「{c_n_name}」へワープ中..."):
                                res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                if res_next: st.session_state.current_result = res_next
                            st.rerun()

            # ===============================================
            # 📗 問題・解法ステップ優先ルート (View)
            # ===============================================
            else:
                top_matches = res.get("top_matches", [res["top_match"]])
                
                if len(top_matches) > 1:
                    tab_titles = []
                    for m in top_matches:
                        t_name = m["node"].get("concept_name", "タスク")
                        tab_titles.append(t_name)
                    selected_title = st.radio("🎯 表示するタスクを選択してください:", tab_titles, horizontal=True)
                    selected_idx = tab_titles.index(selected_title)
                    tm = top_matches[selected_idx]
                    tab_idx = selected_idx
                else:
                    tm = top_matches[0]
                    tab_idx = 0

                kanban_name = tm.get("kanban_concept_name", "概念未設定")
                kanban_node = tm.get("kanban_concept_node", {})
                
                main_qs = tm.get("pure_assessment_questions", [])
                m_exs = tm.get("pure_modeling_exercises", [])
                prereq_qs = tm.get("prereq_questions", [])
                c_videos = tm.get("pure_concept_videos", [])
                
                if res.get("is_drilldown"):
                    st.markdown(f"### {kanban_name}")
                else:
                    display_score = min(tm['final_score'], 1.0)
                    base = tm['base_score']
                    boost = tm['boost_amount']
                    pen_str = f" | {tm['penalty_reason']}" if tm['penalty_reason'] != "なし (既習・復習範囲)" else ""
                    st.markdown(f"### {kanban_name} (総合適合度: {display_score*100:.1f}%)")
                    st.caption(f"📊 **【スコア内訳】** ベース類似度: {base*100:.1f}% | 加点ブースト: +{boost*100:.1f}%{pen_str}")
                    
                st.caption(f"🎓 講義名: {kanban_node.get('lecture_name', '未設定')}")
                st.info(f"**💡 概念要約:** {kanban_node.get('summary', '要約なし')}")

                # 📝 本命の問題
                st.markdown("#### 📝 本命の問題")
                with st.container(border=True):
                    if main_qs:
                        for mq in main_qs:
                            if mq.get("global_q_id"): displayed_q_ids.add(mq.get("global_q_id"))

                        if len(main_qs) > 1:
                            q_tab_titles = [f"確認問題 {clean_q_label(q.get('local_q_num', ''))}" for q in main_qs]
                            selected_q_title = st.radio("📚 表示する問題を選択:", q_tab_titles, horizontal=True)
                            selected_q_idx = q_tab_titles.index(selected_q_title)
                            main_q = main_qs[selected_q_idx]
                        else:
                            main_q = main_qs[0]
                            
                        lecture_n = main_q.get('lecture_name', '未設定')
                        q_num_str = main_q.get('local_q_num', '')
                        cleaned_num = clean_q_label(q_num_str)
                        st.markdown(f"**📌 📗 {lecture_n} 確認問題 {cleaned_num}**")
                        
                        q_text = format_text_for_markdown(main_q.get("question_text", ""))
                        st.info(q_text)
                        render_answer_explanation(main_q.get("answer_text", ""))
                    else:
                        main_q = None
                        st.write("該当なし")

                st.divider()

                # 🎬 Action 1: モデリング
                st.markdown("#### 🎬 第一アクション (問題の直接解説・モデリング)")
                if m_exs:
                    ex = m_exs[0]
                    for mx in m_exs:
                        if mx.get("global_q_id"): displayed_q_ids.add(mx.get("global_q_id"))

                    with st.container(border=True):
                        ex_q_id = ex.get("global_q_id")
                        q_num_str = ex.get('local_q_num', '')
                        cleaned_num = clean_q_label(q_num_str)
                        st.caption(f"🎓 講義名: {ex.get('lecture_name', '未設定')}")
                        st.markdown(f"**📌 📘 大問 {cleaned_num}**")
                        q_text = format_text_for_markdown(ex.get("question_text", ""))
                        st.markdown(q_text)
                        render_answer_explanation(ex.get("answer_text", ""))
                        edges = ex.get("aligned_videos", [])
                        if edges:
                            for idx, e in enumerate(edges):
                                render_video_item(e, f"action1_q_{ex_q_id}_{tab_idx}_0_{idx}", is_modeling=True)
                        else:
                            st.write("該当する解説動画はありません。")
                else:
                    st.write("該当なし")

                # 📘 Action 2: 概念インプット
                st.markdown("#### 📘 第二アクション (概念インプット講義)")
                if c_videos:
                    for idx, v in enumerate(c_videos):
                        render_video_item(v, f"action2_vid_{kanban_node.get('global_c_id')}_{tab_idx}_{idx}", is_modeling=False)
                else:
                    st.write("該当なし")

                st.divider()
                
                # 🥈 次点
                st.subheader("🥈 次点 (前提となる概念を確認する問題)")
                drawn_pre = 0
                cols = st.columns(3)
                
                for r in prereq_qs:
                    r_node = r["node"]
                    r_q_id = r_node.get("global_q_id")
                    
                    if r_q_id and (r_q_id in displayed_q_ids):
                        continue
                        
                    with cols[drawn_pre % 3]:
                        with st.container(border=True):
                            displayed_q_ids.add(r_q_id)
                            q_num_str = r_node.get('local_q_num', '')
                            cleaned_num = clean_q_label(q_num_str)
                            st.caption(f"🎓 講義名: {r_node.get('lecture_name', '未設定')}")
                            st.markdown(f"**📌 📗 確認問題 {cleaned_num}**")
                            
                            matched_tasks = r.get("matched_task_names", [])
                            if matched_tasks:
                                t_str = "、".join(matched_tasks[:2])
                                st.info(f"💡 タスク「{t_str}」の確認")

                            q_text = format_text_for_markdown(r_node.get("question_text", ""))
                            st.markdown(q_text)
                            st.markdown("<hr style='margin: 0.5em 0;'>", unsafe_allow_html=True)
                            render_answer_explanation(r_node.get("answer_text", ""))

                    drawn_pre += 1
                    
                if drawn_pre == 0:
                    st.write("該当なし")

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("🔁 類題")
                with st.container(border=True):
                    st.info("※未実装（将来アップデートで追加予定）")

                st.divider()

                # 🧭 Graph RAG
                st.subheader("🧭 Graph RAG: オントロジー探索 (学習の繋がり)")
                graph_nodes, graph_edges = [], []
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

                        if is_current and node_type == "problem": color, shape = "#FFD54F", "star"
                        elif is_current: color, shape = "#FFECB3", "star"
                        elif node_type == "foundation_knowledge": color, shape = "#BBDEFB", "box"
                        elif node_type == "perspective_condition": color, shape = "#E1BEE7", "hexagon"
                        elif node_type == "derived_knowledge": color, shape = "#C8E6C9", "box"
                        elif node_type == "tasks": color, shape = "#B0BEC5", "box"
                        else: color, shape = "#E0E0E0", "ellipse"

                        graph_nodes.append(Node(id=nid, label=display_label, size=30, color=color, shape=shape, title=display_tooltip))
                        node_ids.add(nid)

                q_id = main_q.get("global_q_id") if main_q else None
                if q_id:
                    q_num_str = main_q.get('local_q_num', '')
                    cleaned_num = clean_q_label(q_num_str)
                    q_label = f"問題: {cleaned_num}"
                    add_graph_node(q_id, q_label, "problem", tooltip=f"📍 現在地：{q_label}", is_current=True)
                    if kanban_name:
                        add_graph_node(kanban_name, kanban_name, kanban_node.get("type", "tasks"), tooltip=f"⬛ [測られるタスク]\n{kanban_name}")
                        graph_edges.append(Edge(source=q_id, target=kanban_name, label="測られる技能", dashes=False))
                elif kanban_name:
                    add_graph_node(kanban_name, kanban_name, kanban_node.get("type", "tasks"), tooltip=f"📍 現在地（タスク）：{kanban_name}", is_current=True)
                    
                if kanban_name:
                    p_name_target = kanban_node.get("parent_concept", "未分類")
                    added_edges = set()
                    promoted_node = tm["node"].get("promoted_from_node")
                    if promoted_node:
                        p_name = promoted_node.get("concept_name")
                        add_graph_node(p_name, p_name, promoted_node.get("type", "derived_knowledge"), tooltip="🚀 検索トリガー (再構成知識)")
                        is_in_prereq = any(p.get("prereq_name") == p_name for p in tm.get("graph_prerequisites", []))
                        if not is_in_prereq:
                            graph_edges.append(Edge(source=p_name, target=kanban_name, label="トリガー", dashes=True, color="#FF9800", width=2))
                            added_edges.add((p_name, kanban_name))
                    if p_name_target and p_name_target != "未分類":
                        add_graph_node(p_name_target, f"親: {p_name_target}", "unknown")
                        graph_edges.append(Edge(source=p_name_target, target=kanban_name, dashes=True, arrows="", width=3, color="#BBDEFB"))

                    for pre_info in tm.get("graph_prerequisites", []):
                        pre_node = pre_info["node"]
                        pre_name = pre_node.get("concept_name")
                        dep_type = pre_info.get("dependency_type", "mandatory")
                        reasoning = pre_info.get("reasoning", "")
                        if pre_name:
                            is_mandatory = (dep_type == "mandatory")
                            badge_str = "🔵 [必須前提]" if is_mandatory else "🟡 [補足前提]"
                            tt_text = f"{badge_str} {pre_name}\n💡 理由: {reasoning}" if reasoning else f"{badge_str} {pre_name}"
                            add_graph_node(pre_name, pre_name, pre_node.get("type", "unknown"), tooltip=tt_text)
                            
                            edge_label = "必須" if is_mandatory else "補足"
                            edge_color = "#BBDEFB"
                            edge_width = None
                            
                            if promoted_node and pre_name == promoted_node.get("concept_name"):
                                edge_label = f"トリガー ({edge_label})"
                                edge_color = "#FF9800"
                                edge_width = 2
                                
                            if (pre_name, kanban_name) not in added_edges:
                                kwargs = {"source": pre_name, "target": kanban_name, "label": edge_label, "dashes": not is_mandatory}
                                if edge_color: kwargs["color"] = edge_color
                                if edge_width: kwargs["width"] = edge_width
                                graph_edges.append(Edge(**kwargs))
                                added_edges.add((pre_name, kanban_name))
                    for sib in tm.get("graph_siblings", []):
                        sib_name = sib.get("concept_name")
                        if sib_name:
                            add_graph_node(sib_name, sib_name, sib.get("type", "unknown"))
                            if p_name_target and p_name_target != "未分類":
                                graph_edges.append(Edge(source=p_name_target, target=sib_name, dashes=True, arrows="", color="#BBDEFB", length=200))
                    for nxt in tm.get("graph_next_steps", []):
                        nxt_name = nxt.get("concept_name")
                        if nxt_name:
                            add_graph_node(nxt_name, nxt_name, nxt.get("type", "unknown")) 
                            graph_edges.append(Edge(source=kanban_name, target=nxt_name, label="必要", dashes=True, color="#BBDEFB"))

                config = Config(width="100%", height=400, directed=True, physics=False, hierarchical={"enabled": True, "direction": "UD", "sortMethod": "directed"})
                with st.expander("🗺️ 学習スキルツリーを開く (クリックで探索可能)", expanded=True):
                    st.info("💡 **ヒント**: 気になるノードにカーソルを合わせるか、クリックするとその概念の世界へワープして探索を続けられます！")
                    st.caption("🟦 基礎知識 | 🟪 視点・条件(六角形) | 🟩 再構成知識 | ⬛ タスク(技能) | ⭐️ 現在地")
                    clicked_node_id = agraph(nodes=graph_nodes, edges=graph_edges, config=config)
                    target_to_check = q_id if q_id else kanban_name
                    if clicked_node_id and clicked_node_id != target_to_check:
                        if clicked_node_id != st.session_state.last_clicked_node:
                            st.session_state.last_clicked_node = clicked_node_id
                            st.session_state.history.append({
                                "result": st.session_state.current_result,
                                "query": st.session_state.display_query,
                                "image_choices": st.session_state.pending_image_choices,
                            })
                            tgt_n = db.get("global_concept_nodes", {}).get(clicked_node_id) or db.get("global_question_nodes", {}).get(clicked_node_id)
                            c_n_name = tgt_n.get("concept_name", "") if tgt_n else clicked_node_id.replace("親: ", "")
                            new_query = f"{c_n_name} について詳しく知りたい (ID:{clicked_node_id})"
                            st.session_state.display_query = new_query
                            with st.spinner(f"「{c_n_name}」へワープ中..."):
                                res_next = execute_search_for_ui(new_query, db, is_drilldown=True)
                                if res_next: st.session_state.current_result = res_next
                            st.rerun()

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
                    value=re.sub(r"\s*\(ID:.+?\)$", "", st.session_state.display_query) if st.session_state.display_query else "",
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
                            res = execute_search_for_ui(new_query, db, is_drilldown=False, is_image_query=True)
                            if res:
                                st.session_state.current_result = res
                            st.session_state.pending_image_choices = None
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