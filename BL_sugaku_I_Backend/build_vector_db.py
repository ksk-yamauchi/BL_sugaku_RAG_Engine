import os
import json
import time
import requests
from glob import glob
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# =========================================================
# ⚙️ 設定・初期化 (.env 複数APIキー対応)
# =========================================================
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

def clean_key(k_str):
    if not k_str:
        return ""
    return k_str.strip().strip("'\"").replace('\ufeff', '')

raw_keys = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [clean_key(k) for k in raw_keys.split(",") if clean_key(k)]

if not API_KEYS:
    raise ValueError("❌ .env ファイルに GEMINI_API_KEYS または GEMINI_API_KEY が設定されていません。")

current_key_index = 0
MODEL_NAME = "text-embedding-004"

def get_client():
    global current_key_index
    return genai.Client(api_key=API_KEYS[current_key_index])

def rotate_key():
    global current_key_index
    if len(API_KEYS) <= 1:
        return False
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    masked_key = f"{API_KEYS[current_key_index][:6]}...{API_KEYS[current_key_index][-4:]}" if len(API_KEYS[current_key_index]) > 10 else "INVALID"
    print(f"      🔄 APIキーを切り替えました (Key {current_key_index + 1}/{len(API_KEYS)}: {masked_key})")
    return True

def discover_embed_model(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    res = requests.get(url, proxies={"http": None, "https": None}, timeout=10)
    res.raise_for_status()
    embed_models = [m["name"] for m in res.json().get("models", []) if "embedContent" in m.get("supportedGenerationMethods", [])]
    if not embed_models: 
        raise RuntimeError("❌ 利用可能な埋め込みモデルが見つかりません。")
    target_model = embed_models[-1]
    for m in embed_models:
        if "text-embedding" in m: 
            target_model = m
            break
    if target_model.startswith("models/"):
        target_model = target_model.replace("models/", "")
    return target_model

def get_embedding(text, max_retries=None):
    if not text.strip(): return []
    if max_retries is None: max_retries = max(5, len(API_KEYS) * 2)
    for attempt in range(1, max_retries + 1):
        try:
            client = get_client()
            result = client.models.embed_content(model=MODEL_NAME, contents=text)
            return result.embeddings[0].values
        except errors.APIError as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["429", "quota", "resource_exhausted"]):
                if rotate_key(): continue
                else: time.sleep(10)
            else:
                if attempt == max_retries: raise e
                time.sleep(5)
        except Exception as e:
            if attempt == max_retries: raise e
            time.sleep(5)
    return []

def main():
    global MODEL_NAME
    
    print("=== 🏁 【Ver 14.0 動画主従関係分離・カタログ統合版】グローバルDB構築プロセス起動 ===")
    print(f"   🔑 読み込み済み有効APIキー数: {len(API_KEYS)} 個")
    try:
        MODEL_NAME = discover_embed_model(API_KEYS[0])
        print(f"   ✅ 使用ベクトルモデル: {MODEL_NAME}\n")
    except Exception as e:
        print(f"   ⚠️ モデルの探索に失敗しました（{e}）。デフォルトモデルを使用します。\n")
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FILE = os.path.join(CURRENT_DIR, "global_vector_db_cache.json")
    
    mext_paths = [
        os.path.join(CURRENT_DIR, "mext_master_dict.v2.json"),
        os.path.join(CURRENT_DIR, "mext_master_dict_v2.json"),
        os.path.join(CURRENT_DIR, "mext_master_dict.json")
    ]
    mext_dict_path = next((p for p in mext_paths if os.path.exists(p)), None)
    if not mext_dict_path: raise FileNotFoundError("❌ mext_master_dict.v2.json が見つかりません。")
        
    with open(mext_dict_path, "r", encoding="utf-8") as f:
        mext_dict_list = json.load(f).get("mext_dictionary", [])
        mext_dict = {item["mext_code"]: item for item in mext_dict_list}

    sub_dirs = [os.path.join(CURRENT_DIR, d) for d in os.listdir(CURRENT_DIR) if os.path.isdir(os.path.join(CURRENT_DIR, d)) and "BL_sugaku" in d]
    sub_dirs.sort()

    global_nodes_map = {}
    global_questions_map = {}
    global_mext_index = {}
    global_edges = []
    global_alignments = []
    global_video_catalog = {}  
    global_timeline_counter = 0

    # 1. データの収集とマージ
    for s_dir in sub_dirs:
        file_path = os.path.join(s_dir, "output_result", "final_knowledge_graph_complete.json")
        lecture_map_path = os.path.join(s_dir, "output_result", "lecture_map.json")  
        if not os.path.exists(file_path): continue
        part_name = os.path.basename(s_dir)
        
        with open(file_path, "r", encoding="utf-8") as f:
            try: data = json.load(f)
            except json.JSONDecodeError: continue
            
        engine_version = data.get("metadata", {}).get("engine_version", "")
        if not str(engine_version).startswith("14."): continue
        bundle_name = data.get("metadata", {}).get("bundle_name", part_name)
        
        if os.path.exists(lecture_map_path):
            with open(lecture_map_path, "r", encoding="utf-8") as f:
                try: 
                    l_map_data = json.load(f)
                    for v in l_map_data.get("videos", []):
                        v_file = v.get("video_file")
                        if v_file:
                            global_video_catalog[v_file] = {
                                "bundle_name": bundle_name,
                                "role": v.get("role", "unknown"),
                                "segments": v.get("segments", [])
                            }
                except json.JSONDecodeError: pass
        
        part_alignments = data.get("alignments", [])
        for align in part_alignments:
            align["bundle_name"] = bundle_name
            global_alignments.append(align)

        node_categories = {
            "foundation_knowledge": {"label": "基礎知識", "pillar": "知識及び技能", "role": "条件によって揺らがない純粋な数学的定義・用語（体系化の土台）"},
            "perspective_condition": {"label": "視点・条件", "pillar": "思考力，判断力，表現力等", "role": "事象の本質を捉え直す視点・条件・思考の枠組み（レンズ）"},
            "derived_knowledge": {"label": "再構成知識", "pillar": "思考の変容プロセス", "role": "基礎知識に視点を通した結果得られる新たな気付き（パラダイムシフト）"},
            "tasks": {"label": "タスク（技能）", "pillar": "技能", "role": "生徒が知識を用いて実際に行う計算手順・操作アクション（問題解決のゴール）"}
        }
        empty_edges_template = {"part_of": [], "is_a": [], "subsumes": [], "relative_to": [], "applies_condition": [], "requires_logical": [], "applied_to": [], "explanation": [], "prerequisite": []}

        for k_type, meta_def in node_categories.items():
            for node in data.get("nodes", {}).get(k_type, []):
                n_id = node.get("node_id")
                if not n_id: continue
                
                # 🌟 新規ノードの初期化（動画配列を main_videos と review_videos に分割）
                if n_id not in global_nodes_map:
                    mext_code = node.get("mext_code", "")
                    mext_info = mext_dict.get(mext_code, {})
                    node_name = node.get("name", "")
                    global_nodes_map[n_id] = {
                        "global_c_id": n_id, "id": n_id, "type": k_type, "type_label": meta_def["label"], "pillar": meta_def["pillar"], "role_desc": meta_def["role"],
                        "name": node_name, 
                        "concept_name": node_name,
                        "parent_concept": node.get("parent_concept", ""), "summary": node.get("summary", ""),
                        "mext_code": mext_code, "mext_hierarchy": mext_info.get("hierarchy_text", "不明な階層"), "mext_official_text": mext_info.get("official_text", ""), "mext_explanation": mext_info.get("explanation_summary", "解説なし"),
                        "main_videos": [], "review_videos": [], "aligned_questions_text": [], 
                        "incoming_edges": {k: [] for k in empty_edges_template}, "outgoing_edges": {k: [] for k in empty_edges_template},
                        "global_timeline_index": global_timeline_counter
                    }
                    global_timeline_counter += 1

                # 🌟 動画の振り分け処理 (新規・既存問わず実行し、アライメントタイプに応じて配列を分ける)
                for new_v in node.get("aligned_videos", []):
                    v_file = new_v.get("video_file")
                    a_type = new_v.get("alignment_type", "")
                    
                    if a_type in ["concept_introduction", "task_walkthrough"]:
                        # main_videosに重複なく追加
                        if not any(v.get("video_file") == v_file for v in global_nodes_map[n_id]["main_videos"]):
                            global_nodes_map[n_id]["main_videos"].append(new_v)
                    else:
                        # prerequisite_review や prerequisite の場合は review_videosに重複なく追加
                        if not any(v.get("video_file") == v_file for v in global_nodes_map[n_id]["review_videos"]):
                            global_nodes_map[n_id]["review_videos"].append(new_v)

        for q in data.get("questions", []):
            q_num = str(q.get("question_number", ""))
            q_id = f"Q_{bundle_name}_{q_num}"
            global_questions_map[q_id] = {
                "global_q_id": q_id, 
                "id": q_id, "type": "question", "bundle_name": bundle_name, "question_number": q_num, 
                "local_q_num": q_num, 
                "question_text": q.get("question_text", ""), "answer_text": q.get("answer_text", ""), "aligned_videos": q.get("aligned_videos", []), 
                "matched_concept": "",
                "global_timeline_index": global_timeline_counter
            }
            global_timeline_counter += 1

        for edge in data.get("edges", []):
            global_edges.append(edge)

    # 2. エッジ情報の構造化紐付け
    for edge in global_edges:
        src = edge.get("source_id")
        tgt = edge.get("target_id")
        rel = edge.get("relation_type", "prerequisite")
        reason = edge.get("reasoning", "")
        edge_data_out = {"target_id": tgt, "reasoning": reason}
        edge_data_in = {"source_id": src, "reasoning": reason}
        if src in global_nodes_map:
            if rel not in global_nodes_map[src]["outgoing_edges"]: global_nodes_map[src]["outgoing_edges"][rel] = []
            global_nodes_map[src]["outgoing_edges"][rel].append(edge_data_out)
        if tgt in global_nodes_map:
            if rel not in global_nodes_map[tgt]["incoming_edges"]: global_nodes_map[tgt]["incoming_edges"][rel] = []
            global_nodes_map[tgt]["incoming_edges"][rel].append(edge_data_in)

    # 2.5 知識の実体結合 (オートワイヤリング)
    name_to_node_id = {meta["name"]: n_id for n_id, meta in global_nodes_map.items()}
    for n_id, meta in global_nodes_map.items():
        p_name = meta.get("parent_concept", "")
        if p_name and p_name != "未分類" and p_name in name_to_node_id:
            p_id = name_to_node_id[p_name]
            subsumes_edges = global_nodes_map[p_id]["outgoing_edges"].setdefault("subsumes", [])
            if not any(e["target_id"] == n_id for e in subsumes_edges):
                subsumes_edges.append({"target_id": n_id, "reasoning": "システム自動結合 (親概念)"})
            global_nodes_map[n_id]["incoming_edges"].setdefault("subsumes", []).append({"source_id": p_id, "reasoning": "システム自動結合 (親概念)"})
            part_of_edges = global_nodes_map[n_id]["outgoing_edges"].setdefault("part_of", [])
            if not any(e["target_id"] == p_id for e in part_of_edges):
                part_of_edges.append({"target_id": p_id, "reasoning": "システム自動結合 (親概念)"})
            global_nodes_map[p_id]["incoming_edges"].setdefault("part_of", []).append({"source_id": n_id, "reasoning": "システム自動結合 (親概念)"})

    # 3. アライメント情報の紐付け
    for align in global_alignments:
        b_name = align.get("bundle_name")
        q_num = align.get("question_number")
        q_id = f"Q_{b_name}_{q_num}"
        q_data = global_questions_map.get(q_id)
        if not q_data: continue
        q_text_snippet = f"[問題] {q_data['question_text']}\n[解説] {q_data['answer_text']}"
        
        for t_id in align.get("linked_task_ids", []):
            if t_id in global_nodes_map and global_nodes_map[t_id]["type"] == "tasks":
                global_nodes_map[t_id]["aligned_questions_text"].append(q_text_snippet)
                
        k_ids = align.get("linked_knowledge_ids", [])
        if k_ids and not q_data["matched_concept"]:
            for k_id in k_ids:
                if k_id in global_nodes_map:
                    q_data["matched_concept"] = global_nodes_map[k_id]["name"]
                    break

    # prerequisite_concepts (フロント仕様) の作成
    for n_id, meta in global_nodes_map.items():
        pre_list = []
        for rel in ["prerequisite", "applies_condition"]:
            for item in meta["incoming_edges"].get(rel, []):
                src_id = item["source_id"]
                if src_id in global_nodes_map:
                    pre_list.append({
                        "concept_name": global_nodes_map[src_id]["name"],
                        "dependency_type": "mandatory",
                        "reasoning": item["reasoning"]
                    })
        meta["prerequisite_concepts"] = pre_list

    # 4. ベクトル化 (Embedding) 実行
    total_items = len(global_nodes_map) + len(global_questions_map)
    print(f"\n🧠 合計 {total_items} 件のデータをベクトル空間に埋め込みます...")
    current_count = 0
    for n_id, meta in global_nodes_map.items():
        current_count += 1
        print(f"  [{current_count}/{total_items}] Embedding Node: {meta['name']}")
        c_composite = f"【ノード分類】{meta['type_label']}\n【役割】{meta['role_desc']}\n【三つの柱】{meta['pillar']}\n【名称】{meta['name']}\n【上位概念】{meta['parent_concept']}\n【概要】{meta['summary']}\n"
        if meta.get("mext_code"): c_composite += f"【指導要領階層】{meta['mext_hierarchy']}\n【指導要領解説】{meta['mext_explanation']}\n"
        if meta["incoming_edges"]:
            prereqs = [f"理由: {e['reasoning']}" for e in meta["incoming_edges"].get("prerequisite", []) + meta["incoming_edges"].get("applies_condition", [])]
            if prereqs: c_composite += f"【前提条件】{' / '.join(prereqs)}\n"
        
        # 🌟 main_videos と review_videos の両方から解説要約を結合
        if meta["type"] != "tasks":
            for v in meta["main_videos"] + meta["review_videos"]:
                if v.get("explanation_summary"): c_composite += f"【講義要約】{v['explanation_summary']}\n"
        if meta["type"] == "tasks":
            for q_text in meta["aligned_questions_text"]: c_composite += f"【関連する演習問題】\n{q_text}\n"
            for v in meta["main_videos"] + meta["review_videos"]:
                if v.get("blackboard_ocr"): c_composite += f"【解説板書(数式)】{v['blackboard_ocr']}\n"
        
        vector = get_embedding(c_composite)
        meta["concept_vector"] = vector
        meta["vector"] = vector
        if meta.get("mext_code"): global_mext_index.setdefault(meta["mext_code"], []).append(n_id)
        time.sleep(0.5)

    for q_id, meta in global_questions_map.items():
        current_count += 1
        print(f"  [{current_count}/{total_items}] Embedding Question: {meta['bundle_name']} 問題 {meta['question_number']}")
        q_composite = f"【単元】{meta['bundle_name']}\n【問題・演習】\n問題文: {meta['question_text']}\n解説: {meta['answer_text']}\n"
        for v in meta["aligned_videos"]:
            if v.get("blackboard_ocr"): q_composite += f"【板書OCR】{v['blackboard_ocr']}\n"
        vector = get_embedding(q_composite)
        meta["question_vector"] = vector
        meta["vector"] = vector
        time.sleep(0.5)

    db_payload = {
        "embed_model": MODEL_NAME,
        "metadata": {"engine_version": "14.0_video_catalog_integrated", "embed_model": MODEL_NAME},
        "global_concept_nodes": global_nodes_map,
        "global_question_nodes": global_questions_map,
        "global_mext_index": global_mext_index,
        "global_video_catalog": global_video_catalog
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(db_payload, f, ensure_ascii=False, indent=2)

    print("\n=========================================================")
    print(f"🎉 グローバルベクトルDB（動画主従関係分離・カタログ統合版）構築完了！\n💾 保存先: {OUTPUT_FILE}")
    print("=========================================================")

if __name__ == "__main__":
    main()