import os
import json
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# =========================================================
# ⚙️ 設定・初期化 (.env 複数APIキー対応 ＆ 強制上書き)
# =========================================================
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
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
MODEL_NAME = "gemini-3.6-flash" # 膨大なコンテキスト（グラフ＋動画）の統合と高度な推論のためProを推奨

def get_client():
    global current_key_index
    return genai.Client(api_key=API_KEYS[current_key_index])

def rotate_key():
    global current_key_index
    if len(API_KEYS) <= 1:
        print("   ⚠️ 登録されているAPIキーが1つのため、キー切り替えができません。")
        return False
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    masked_key = f"{API_KEYS[current_key_index][:6]}...{API_KEYS[current_key_index][-4:]}" if len(API_KEYS[current_key_index]) > 10 else "INVALID"
    print(f"   🔄 APIキーを切り替えました (Key {current_key_index + 1}/{len(API_KEYS)}: {masked_key})")
    return True

def generate_content_and_parse_json(prompt, max_retries=None):
    if max_retries is None:
        max_retries = max(5, len(API_KEYS) * 2)

    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)

    for attempt in range(1, max_retries + 1):
        try:
            client = get_client()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            
            text = response.text
            text = re.sub(r'^```json\s*', '', text.strip(), flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                fixed_text = text.replace('\\', '\\\\').replace('\\\\"', '\\"').replace('\\\\n', '\\n')
                try:
                    return json.loads(fixed_text)
                except json.JSONDecodeError as je:
                    print(f"      ⚠️ JSON形式エラー(LaTeX起因等)。安全に再生成します... (試行 {attempt}/{max_retries})")
                    if attempt == max_retries:
                        raise RuntimeError(f"❌ JSONパースが{max_retries}回失敗しました: {je}")
                    time.sleep(3)
                    continue

        except errors.APIError as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["429", "quota", "resource_exhausted", "api_key_invalid", "invalid_argument"]):
                if rotate_key():
                    print("      ⏩ 新しいAPIキーで即座にリトライします...")
                    continue
                else:
                    print("      ⏳ 40秒待機後にリトライします...")
                    time.sleep(40)
            elif "503" in err_str or "unavailable" in err_str:
                print(f"      ⚠️ 503エラー: 30秒待機後リトライ...")
                time.sleep(30)
            else:
                if attempt == max_retries: raise e
                time.sleep(20)
        except Exception as e:
            if attempt == max_retries: raise e
            time.sleep(20)
    raise RuntimeError("❌ リトライ上限超過")

# =========================================================
# 📂 パス定義とマスター管理関数
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_result")
PHASE1_FILE = os.path.join(OUTPUT_DIR, "final_knowledge_graph.json")
PHASE2_FILE = os.path.join(OUTPUT_DIR, "lecture_map.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_knowledge_graph_complete.json")

KNOWLEDGE_MASTER_PATH = os.path.join(PARENT_DIR, "knowledge_master.json")
TASK_MASTER_PATH = os.path.join(PARENT_DIR, "task_master.json")

def load_json(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, "r", encoding="utf-8") as f: return json.load(f)

def time_to_seconds(t_str):
    try:
        m, s = map(int, t_str.split(':'))
        return m * 60 + s
    except:
        return -1

def assign_or_get_code(master_path, mext_code, node_name, summary, bundle_name, prefix=""):
    """Phase 3で動的に追加されたノードをマスターに登録し、枝番を採番する"""
    master_data = {}
    if os.path.exists(master_path):
        try:
            with open(master_path, "r", encoding="utf-8") as f: master_data = json.load(f)
        except: pass

    mext_code = str(mext_code).strip()
    if not mext_code: mext_code = "UNKNOWN"
    if mext_code not in master_data: master_data[mext_code] = []

    for item in master_data[mext_code]:
        if item["name"] == node_name: return item["branch_code"]

    existing_nums = []
    for item in master_data[mext_code]:
        b_code = item.get("branch_code", f"_{prefix}000")
        match = re.search(r"_([A-Z]?)(\d+)", b_code)
        if match: existing_nums.append(int(match.group(2)))
    
    next_num = max(existing_nums) + 1 if existing_nums else 1
    new_branch_code = f"_{prefix}{next_num:03d}"

    master_data[mext_code].append({
        "branch_code": new_branch_code,
        "name": node_name,
        "summary_snippet": summary[:100] if summary else "",
        "first_appeared_in": bundle_name
    })

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    return new_branch_code

# =========================================================
# 🧠 動的補完 ＆ アライメント実行
# =========================================================
def execute_dynamic_alignment(phase1_data, phase2_data, bundle_name):
    print("🚀 [Phase 3] マルチモーダル・オントロジー統合とグラフ自己増殖を実行中...")

    # Phase 1のグラフとPhase 2の動画データをプロンプトに渡す
    phase1_summary = {
        "nodes": phase1_data.get("nodes", {}),
        "edges": phase1_data.get("edges", []),
        "questions": phase1_data.get("questions", [])
    }
    video_segments = phase2_data.get("videos", [])

    prompt = f"""あなたは教育工学とカリキュラム・アライメントのエキスパートです。
以下の【Phase 1: テキストから抽出した基礎グラフ】と【Phase 2: 動画タイムライン・板書データ】を読み込み、以下の2つのミッションを実行してください。

【ミッション1：暗黙知の可視化とグラフの動的補完（差分抽出）】
テキストには書かれていないが、動画の「口頭解説（explanation_summary）」や「板書（blackboard_ocr）」から、以下の要素が新たに発見された場合は、それらを【追加ノード・追加エッジ】として抽出してください。
- 基礎知識（foundation_knowledge）
- 新しい視点・条件（perspective_condition / レンズ）★特に重要
- 再構成された知識（derived_knowledge）
- 高度なタスク（tasks）
※ 追加するノードのIDは "NEW_K1", "NEW_T1" などの形式を使用してください。追加要素がない場合は空配列で構いません。

【ミッション2：三位一体アライメント】
既存のノード、新たに追加したノード、および確認問題に対して、それを解説している「動画セグメント」を紐づけてください。
- `alignment_type` は "concept_input" (概念の解説), "direct_explanation" (例題の直接解説), "prerequisite" (前提解説) のいずれかを指定してください。

【★最重要: LaTeXとJSONエスケープの絶対ルール】
`reasoning`等にLaTeX数式を含める場合は、必ずバックスラッシュを二重にエスケープ（例: \\\\frac, \\\\subset）してください。

---
■ 【Phase 1】テキスト抽出ベースグラフ:
{json.dumps(phase1_summary, ensure_ascii=False)}

■ 【Phase 2】動画タイムライン・板書データ:
{json.dumps(video_segments, ensure_ascii=False)}
---

【出力JSONフォーマット】:
{{
  "added_nodes": {{
    "foundation_knowledge": [
      {{ "node_id": "NEW_K1", "name": "...", "summary": "...", "parent_concept": "...", "mext_code": "..." }}
    ],
    "perspective_condition": [],
    "derived_knowledge": [],
    "tasks": []
  }},
  "added_edges": [
    {{
      "source_id": "NEW_K1",
      "target_id": "T1",
      "relation_type": "applies_condition | prerequisite | relative_to | part_of | is_a | subsumes | requires_logical | applied_to | explanation",
      "reasoning": "動画内で先生が〇〇と解説していたため"
    }}
  ],
  "node_video_alignments": [
    {{
      "node_id": "K1",
      "aligned_videos": [
        {{
          "video_file": "...",
          "start_time": "MM:SS",
          "alignment_type": "concept_input",
          "reasoning": "板書に〇〇とあり、この概念の解説に該当するため"
        }}
      ]
    }}
  ],
  "question_video_alignments": [
    {{
      "question_number": "1",
      "aligned_videos": [
        {{ "video_file": "...", "start_time": "MM:SS", "alignment_type": "direct_explanation", "reasoning": "..." }}
      ]
    }}
  ]
}}
"""
    return generate_content_and_parse_json(prompt)

def main():
    print("=== 🏁 【Ver 13.0 動的オントロジー補完＆アライメント対応】Phase 3 起動 ===")
    print(f"   🔑 読み込み済み有効APIキー数: {len(API_KEYS)} 個")
    
    phase1_data = load_json(PHASE1_FILE)
    phase2_data = load_json(PHASE2_FILE)
    
    if not phase1_data:
        print("❌ Phase 1 のデータが見つかりません。")
        return

    bundle_name = phase1_data.get("metadata", {}).get("bundle_name", "Unknown_Bundle")

    if not phase2_data:
        print("⚠️ Phase 2 の動画データがありません。アライメントをスキップします。")
        final_graph = phase1_data.copy()
        final_graph["metadata"]["engine_version"] = "13.0_video_skipped"
    else:
        # LLMによる動的補完とアライメントの実行
        result = execute_dynamic_alignment(phase1_data, phase2_data, bundle_name)
        video_segments = phase2_data.get("videos", [])
        
        added_nodes = result.get("added_nodes", {})
        added_edges = result.get("added_edges", [])
        node_alignments = {item["node_id"]: item["aligned_videos"] for item in result.get("node_video_alignments", [])}
        question_alignments = {str(item["question_number"]): item["aligned_videos"] for item in result.get("question_video_alignments", [])}

        print("   🌐 追加されたノードをマスター辞書に登録・採番中...")
        # 新規ノードに枝番を採番
        knowledge_lists = ["foundation_knowledge", "perspective_condition", "derived_knowledge"]
        for k_type in knowledge_lists:
            for node in added_nodes.get(k_type, []):
                m_code = node.get("mext_code", "")
                name = node.get("name", "")
                summary = node.get("summary", "")
                node["branch_code"] = assign_or_get_code(KNOWLEDGE_MASTER_PATH, m_code, name, summary, bundle_name, prefix="K")
                # Phase 1のグラフに結合
                phase1_data["nodes"].setdefault(k_type, []).append(node)

        for node in added_nodes.get("tasks", []):
            m_code = node.get("mext_code", "")
            name = node.get("name", "")
            summary = node.get("summary", "")
            node["branch_code"] = assign_or_get_code(TASK_MASTER_PATH, m_code, name, summary, bundle_name, prefix="T")
            phase1_data["nodes"].setdefault("tasks", []).append(node)

        # 追加エッジの結合
        phase1_data["edges"].extend(added_edges)

        # 🌟 動画データのファジーマッチ結合処理
        def enrich_videos(aligned_list, p2_videos):
            enriched = []
            for v in aligned_list:
                v_file = v.get("video_file", "")
                s_time = v.get("start_time", "")
                s_sec = time_to_seconds(s_time)
                new_v = v.copy()
                
                matched_seg = None
                for p2v in p2_videos:
                    if p2v.get("video_file") == v_file:
                        segments = p2v.get("segments", [])
                        # 1. 完全一致
                        for seg in segments:
                            if seg.get("start_time") == s_time:
                                matched_seg = seg
                                break
                        # 2. ファジーマッチ（最寄時間）
                        if not matched_seg and s_sec >= 0 and segments:
                            matched_seg = min(segments, key=lambda seg: abs(time_to_seconds(seg.get("start_time", "")) - s_sec))
                
                if matched_seg:
                    new_v["end_time"] = matched_seg.get("end_time", "")
                    new_v["topic"] = matched_seg.get("topic", "")
                    new_v["blackboard_ocr"] = matched_seg.get("blackboard_ocr", "")
                    new_v["explanation_summary"] = matched_seg.get("explanation_summary", "")
                enriched.append(new_v)
            return enriched

        print("   🔗 ノードおよび問題に動画データを結合（ファジーマッチ）しています...")
        # ノードへの動画結合
        for category, node_list in phase1_data.get("nodes", {}).items():
            for node in node_list:
                n_id = node.get("node_id")
                raw_aligned = node_alignments.get(n_id, [])
                node["aligned_videos"] = enrich_videos(raw_aligned, video_segments)

        # 問題への動画結合
        for q in phase1_data.get("questions", []):
            q_num = str(q.get("question_number", ""))
            raw_aligned = question_alignments.get(q_num, [])
            q["aligned_videos"] = enrich_videos(raw_aligned, video_segments)

        final_graph = phase1_data
        final_graph["metadata"]["engine_version"] = "13.0_dynamic_ontology_completed"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_graph, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 統合完了！暗黙知の補完と動画リンクが完了しました。\n💾 保存先: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()