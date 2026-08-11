import os
import re
import json
import sys
import time
from glob import glob
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
MODEL_NAME = "gemini-3.6-flash"

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

# 🌟 堅牢なJSON生成＆パース関数
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
                    print(f"   ⚠️ AI出力のJSON形式エラー。安全に再生成します... (試行 {attempt}/{max_retries})")
                    if attempt == max_retries:
                        raise RuntimeError(f"❌ JSONパースが{max_retries}回失敗しました: {je}")
                    time.sleep(3)
                    continue

        except errors.APIError as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["429", "quota", "resource_exhausted", "api_key_invalid", "invalid_argument"]):
                if rotate_key():
                    print("   ⏩ 次のAPIキーへ切り替えて即座にリトライします...")
                    continue
                else:
                    time.sleep(40)
            elif "503" in err_str or "unavailable" in err_str:
                time.sleep(30)
            else:
                if attempt == max_retries: raise e
                time.sleep(15)
        except Exception as e:
            if attempt == max_retries: raise e
            time.sleep(15)
            
    raise RuntimeError("❌ リトライ上限超過")

# =========================================================
# 📂 パス定義・マスター処理
# =========================================================
md_files = glob("*_clean.md")
if not md_files:
    raise FileNotFoundError("❌ 教材Markdown(*_clean.md)が見つかりません。")
TEXTBOOK_MD_PATH = md_files[0]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

MEXT_DICT_PATH = os.path.join(PARENT_DIR, "mext_master_dict_v2.json")
INDEX_MASTER_PATH = os.path.join(PARENT_DIR, "lecture_index_master.json")
KNOWLEDGE_MASTER_PATH = os.path.join(PARENT_DIR, "knowledge_master.json")
TASK_MASTER_PATH = os.path.join(PARENT_DIR, "task_master.json")

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_result")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_lecture_name_from_master(folder_name):
    if not os.path.exists(INDEX_MASTER_PATH): return "Unknown_Lecture"
    with open(INDEX_MASTER_PATH, "r", encoding="utf-8") as f: index_master = json.load(f)
    match = re.search(r"(\d{2}-\d+)", folder_name)
    if not match: return "Unknown_Lecture"
    key = match.group(1)
    return index_master.get(key, {}).get("lecture_name", "Unknown_Lecture")

def load_and_prepare_inputs():
    with open(TEXTBOOK_MD_PATH, "r", encoding="utf-8") as f: textbook_content = f.read()
    folder_name = os.path.basename(CURRENT_DIR)
    lecture_name = get_lecture_name_from_master(folder_name)
    with open(MEXT_DICT_PATH, "r", encoding="utf-8") as f: mext_master_dict = f.read()
    return textbook_content, mext_master_dict, lecture_name

def assign_or_get_code(master_path, mext_code, node_name, summary, lecture_name, prefix=""):
    master_data = {}
    if os.path.exists(master_path):
        try:
            with open(master_path, "r", encoding="utf-8") as f: master_data = json.load(f)
        except: pass

    for m_code, items in master_data.items():
        for item in items:
            if item["name"] == node_name: 
                return f"{m_code}{item['branch_code']}"

    mext_code = str(mext_code).strip()
    if not mext_code: mext_code = "UNKNOWN"
    if mext_code not in master_data: master_data[mext_code] = []

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
        "first_appeared_in": lecture_name
    })

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    return f"{mext_code}{new_branch_code}"

# 🌟 既存のオントロジーリストを取得する関数
def get_existing_ontology_names():
    names = set()
    for p in [KNOWLEDGE_MASTER_PATH, TASK_MASTER_PATH]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    master = json.load(f)
                    for items in master.values():
                        for item in items:
                            names.add(item["name"])
            except: pass
    if names:
        return ", ".join(sorted(list(names)))
    else:
        return "まだありません（この単元が最初の処理です）"

# =========================================================
# 🧠 Phase 1 / Step 1: オントロジー抽出
# =========================================================
def execute_step1_extraction(textbook_content, mext_master_dict, lecture_name):
    print(f"\n🚀 [Phase 1 / Step 1] 概念・タスク抽出を実行中（対象: {lecture_name}）...")

    existing_names = get_existing_ontology_names()

    prompt = f"""あなたは高等学校数学科の教材分析・学習オントロジー構築のエキスパートです。
以下の「教材データ」と「指導要領マスター辞書」を解析し、GNN-KT（学習状態推論）およびGraph RAGに最適化されたナレッジグラフ（ノードとエッジ）を構築してください。

【対象単元】: {lecture_name}

【★最重要：用語の統合と分離のルール★】
過去の単元解析で、以下の概念が既にシステムに登録されています。
■ 登録済み概念リスト: [{existing_names}]

新しく抽出するノードの `name`（名称）や `parent_concept`（上位概念）を決定する際、以下の【統合と分離のルール】を厳守してください。

1. 【単なる表記揺れは統合する】: 上記の登録済みリストにある概念と比較したとき、「特定の文字に着目する」と「特定文字への着目」のような、単なる言い回しや表現の違いである場合は、新語を作らず、上記の登録済みリストにある名称と【一言一句同じ名称】を優先して使用・継承してください。
2. 【数学的定義が異なるものは厳格に分離する】: 「整式」と「多項式」、「方程式」と「恒等式」のように、高校数学の指導において数学的な定義・対象が明確に異なる用語は、一般的な意味が似ていても厳格に区別し、新しい用語として抽出してください。

【★最重要: テキスト抽出とLaTeXの絶対ルール（ノイズ防止）★】
1. 問題文や解説の穴埋め記号は、`\\\\text{{[ア]}}` や `\\\\[ア]` といった複雑なLaTeX表記にせず、必ずシンプルに `[ア]` と記述してください。
2. 日本語のテキスト内に数式や変数（x, y, aなど）を含める場合は、文章中であっても必ず `$` で囲んでください（例: `$x$ に着目すると...`）。
3. 文字化けの原因となるため、文末や行末に不要なバックスラッシュ（`\\\\` や `￥`）を絶対に残さないでください。

【ノードの分類】
1. foundation_knowledge (基礎知識): 単元のベースとなる静的な知識。
2. perspective_condition (視点・条件): 「特定の文字に着目する」など、思考の枠組み（レンズ）。
3. derived_knowledge (再構成知識): 【★重要★】タスクを単に名詞化しただけの結果論（例：「特定文字に着目した多項式の次数と定数項」など）は抽出しないでください。抽出が許されるのは、「定数としての文字の扱い（着目外の文字は数とみなす）」のような【概念的なルールの変化・パラダイムシフト（ゲシュタルトの変換）】のみです。
4. tasks (技能・タスク): 「〜を特定する」「展開する」など、生徒が実行する具体的な学習アクション。※これがGNN-KTの確率計算の主役となります。
   ★【重要: タスク抽出の細分化ルール】: 行うタスクが同じでも、対象となる数式や図形（例：「単項式」と「多項式」）が異なる場合は、GNN-KTで別々の技能として追跡するため、必ず別々のタスクノードとして分割して抽出してください。
※ 一時的なIDとして "K1", "T1" などの文字列を `node_id` に指定してください。

【エッジの生成ルール】
- タスク（tasks）の前提知識（`requires_logical` または `prerequisite`）として、直接「視点（perspective_condition）」と「ルールの変化（derived_knowledge）」を繋ぐようにエッジを生成してください。

【★重要：大問（exercises）と確認問題（questions）の役割の違いと抽出ルール★】
教材テキストに含まれる問題を、その役割に応じて2つの配列に厳格に分離して抽出してください。
1. `exercises`（大問）:
   - それまでに学んだ知識・技能の使い方を学ぶことをメインとする問題（先生によるモデリングの場）です。
   - テキスト上で数字だけの見出し（例: `**1**` や `### 4`）となっているものが該当します。部分的な見落としがないように必ず抽出してください。
   - 【🌟抽出のガイドライン】: 大問は既習の知識・技能の使い方の確認がメインですが、もし大問の解法を通じて知識の組み合わせ方や新しい視点の使い方が示され、そこから【新たなタスク】や【再構成された知識】が見られた場合は、萎縮せずに `tasks` や `derived_knowledge` ノードとしてポジティブに抽出してください。
2. `questions`（確認問題）:
   - 大問で学んだことを生徒が自力で解いて確かめるアセスメントの問題です。
   - 「確認問題」と明記されているものが該当します。こちらもすべて漏れなく抽出してください。

番号のフォーマットは以下に厳格に統一してください。
- 大問の場合: `大問X (Y) (Z)` （例: `大問1 (1)`, `大問4 (2) (i)`）
- 確認問題の場合: `確認問題 X` （例: `確認問題 1`, `確認問題 8`）

【出力JSONフォーマット】:
{{
  "nodes": {{
    "foundation_knowledge": [
      {{
        "node_id": "K1",
        "name": "抽出した用語（テキストまたは既存リスト通り）",
        "parent_concept": "属する用語（既存リスト優先）",
        "summary": "要約",
        "extracted_from": "テキストの該当箇所をそのまま引用",
        "mext_code": "16桁コード"
      }}
    ],
    "perspective_condition": [],
    "derived_knowledge": [],
    "tasks": [
      {{
        "node_id": "T1",
        "name": "生徒の具体的なアクション",
        "parent_concept": "関連用語",
        "summary": "要約",
        "extracted_from": "テキストの該当箇所",
        "mext_code": "16桁コード"
      }}
    ]
  }},
  "edges": [
    {{
      "source_id": "K1",
      "target_id": "T1",
      "relation_type": "applies_condition | prerequisite | relative_to | applied_to | requires_logical",
      "reasoning": "なぜこの関係があるかの理由"
    }}
  ],
  "exercises": [
    {{
      "exercise_number": "大問番号（例: 大問1 (1)）",
      "exercise_text": "問題文（LaTeXエスケープ厳守）",
      "answer_text": "[正解] ... \\n[解説] ..."
    }}
  ],
  "questions": [
    {{
      "question_number": "確認問題 X",
      "question_text": "問題文（LaTeXエスケープ厳守）",
      "answer_text": "[正解] ... \\n[解説] ..."
    }}
  ]
}}

【★最重要: LaTeXエスケープ★】
数式を含める場合は、必ずバックスラッシュを二重にエスケープ（例: \\\\frac, \\\\subset）してください。

■ 教材Markdownデータ:
{textbook_content}
■ 指導要領マスター辞書:
{mext_master_dict}
"""
    return generate_content_and_parse_json(prompt)

# =========================================================
# 🧠 Phase 1 / Step 2: 精密アライメント
# =========================================================
def execute_step2_alignment(mapped_step1_data):
    print("\n🧠 [Phase 1 / Step 2] 問題とGNN-KTタスクの精密アライメントを実行中...")
    
    prompt_data = {
        "nodes": mapped_step1_data["nodes"],
        "exercises": mapped_step1_data["exercises"],
        "questions": mapped_step1_data["questions"]
    }

    prompt = f"""あなたは教育工学のエキスパートです。
以下の整理済みデータから、大問（exercises）および確認問題（questions）と、学習タスク（GNN-KT推論用）のアライメント構造を構築してください。

【整理済みデータ】:
{json.dumps(prompt_data, ensure_ascii=False, indent=2)}

【ルール】
各「大問（exercise_number）」および「確認問題（question_number）」を解くために、どのタスク（tasksノードのID）と、どの知識（foundation_knowledge等のID）が必要になるかを分析し、配列で紐づけてください。

【出力JSONフォーマット】:
{{
  "exercise_alignments": [
    {{
      "exercise_number": "大問番号",
      "linked_task_ids": ["_T001", "_T002"],
      "linked_knowledge_ids": ["_K001"],
      "reasoning": "なぜこれらのタスクや知識が必要かの理由",
      "formula_used": "使用する公式や解法の要点（LaTeXエスケープ厳守）"
    }}
  ],
  "question_alignments": [
    {{
      "question_number": "確認問題番号",
      "linked_task_ids": ["_T001", "_T002"],
      "linked_knowledge_ids": ["_K001"],
      "reasoning": "なぜこれらのタスクや知識が必要かの理由",
      "formula_used": "使用する公式や解法の要点（LaTeXエスケープ厳守）"
    }}
  ]
}}
"""
    return generate_content_and_parse_json(prompt)

def main():
    print("=== 🏁 【Ver 14.5 大問/確認問題 分離抽出 ＆ lecture_name統一版】Phase 1 起動 ===")
    print(f"   🔑 読み込み済み有効APIキー数: {len(API_KEYS)} 個")

    textbook_content, mext_master_dict, lecture_name = load_and_prepare_inputs()
    
    # [Step 1] 抽出
    step1_output = execute_step1_extraction(textbook_content, mext_master_dict, lecture_name)
    
    print("   🌐 マスター辞書との照合・正式ID (_Kxxx, _Txxx) への変換処理中...")
    id_map = {}
    
    # 知識ノードの採番とID置換
    knowledge_lists = ["foundation_knowledge", "perspective_condition", "derived_knowledge"]
    for k_type in knowledge_lists:
        for node in step1_output.get("nodes", {}).get(k_type, []):
            old_id = node.get("node_id")
            new_id = assign_or_get_code(KNOWLEDGE_MASTER_PATH, node.get("mext_code"), node.get("name"), node.get("summary"), lecture_name, "K")
            id_map[old_id] = new_id
            node["node_id"] = new_id
            
    # タスクノードの採番とID置換
    for node in step1_output.get("nodes", {}).get("tasks", []):
        old_id = node.get("node_id")
        new_id = assign_or_get_code(TASK_MASTER_PATH, node.get("mext_code"), node.get("name"), node.get("summary"), lecture_name, "T")
        id_map[old_id] = new_id
        node["node_id"] = new_id

    # エッジのID置換
    for edge in step1_output.get("edges", []):
        edge["source_id"] = id_map.get(edge.get("source_id"), edge.get("source_id"))
        edge["target_id"] = id_map.get(edge.get("target_id"), edge.get("target_id"))

    # [Step 2] アライメント
    step2_output = execute_step2_alignment(step1_output)

    final_knowledge_graph = {
        "metadata": {"lecture_name": lecture_name, "engine_version": "14.5_exercise_question_separation", "model_used": MODEL_NAME},
        "nodes": step1_output.get("nodes", {}),
        "edges": step1_output.get("edges", []),
        "exercises": step1_output.get("exercises", []),
        "questions": step1_output.get("questions", []),
        "exercise_alignments": step2_output.get("exercise_alignments", []),
        "question_alignments": step2_output.get("question_alignments", []),
    }
    
    output_filepath = os.path.join(OUTPUT_DIR, "final_knowledge_graph.json")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_knowledge_graph, f, ensure_ascii=False, indent=2)
    print(f"🎉 処理完了！ 💾 保存先: {output_filepath}")

if __name__ == "__main__":
    main()