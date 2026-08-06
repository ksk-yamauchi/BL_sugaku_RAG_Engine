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
    key = API_KEYS[current_key_index]
    return genai.Client(api_key=key)

def rotate_key():
    global current_key_index
    if len(API_KEYS) <= 1:
        print("   ⚠️ 登録されているAPIキーが1つのため、キー切り替えができません。")
        return False
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    masked_key = f"{API_KEYS[current_key_index][:6]}...{API_KEYS[current_key_index][-4:]}" if len(API_KEYS[current_key_index]) > 10 else "INVALID"
    print(f"   🔄 APIキーを切り替えました (Key {current_key_index + 1}/{len(API_KEYS)}: {masked_key})")
    return True

# 🌟 JSONパースエラー(LaTeXエスケープ等)も検知・修復してリトライする最強関数
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
            
            # Markdownブロックの除去
            text = response.text
            text = re.sub(r'^```json\s*', '', text.strip(), flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)
            
            # JSONのパースと自動修復
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                fixed_text = text.replace('\\', '\\\\').replace('\\\\"', '\\"').replace('\\\\n', '\\n')
                try:
                    return json.loads(fixed_text)
                except json.JSONDecodeError as je:
                    print(f"   ⚠️ AI出力のJSON形式エラー(LaTeX起因等)。安全に再生成します... (試行 {attempt}/{max_retries})")
                    if attempt == max_retries:
                        raise RuntimeError(f"❌ JSONパースが{max_retries}回失敗しました: {je}")
                    time.sleep(3)
                    continue

        except errors.APIError as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["429", "quota", "resource_exhausted", "api_key_invalid", "invalid_argument"]):
                curr_k = API_KEYS[current_key_index]
                masked_k = f"{curr_k[:6]}...{curr_k[-4:]}" if len(curr_k) > 10 else "INVALID"
                print(f"   ⚠️ API制限/無効キーを検知しました (Key: {masked_k}, 試行 {attempt}/{max_retries})")
                if rotate_key():
                    print("   ⏩ 次のAPIキーへ切り替えて即座にリトライします...")
                    continue
                else:
                    print("   ⏳ 40秒待機後に再トライします...")
                    time.sleep(40)
            elif "503" in err_str or "unavailable" in err_str:
                print(f"   ⚠️ 503サーバーエラー (試行 {attempt}/{max_retries}): 30秒待機後に再トライ...")
                time.sleep(30)
            else:
                if attempt == max_retries: raise e
                print(f"   ⚠️ APIエラー ({e}) (試行 {attempt}/{max_retries}): 15秒待機後に再トライ...")
                time.sleep(15)
        except Exception as e:
            if attempt == max_retries: raise e
            print(f"   ⚠️ 通信エラー ({e}) (試行 {attempt}/{max_retries}): 15秒待機後に再トライ...")
            time.sleep(15)
            
    raise RuntimeError("❌ リトライ上限超過")


md_files = glob("*_clean.md")
if not md_files:
    raise FileNotFoundError("❌ 教材Markdown(*_clean.md)が見つかりません。")
TEXTBOOK_MD_PATH = md_files[0]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# 🌟 新しい辞書と、分離されたマスター群のパス
MEXT_DICT_PATH = os.path.join(PARENT_DIR, "mext_master_dict_v2.json")
INDEX_MASTER_PATH = os.path.join(PARENT_DIR, "lecture_index_master.json")
KNOWLEDGE_MASTER_PATH = os.path.join(PARENT_DIR, "knowledge_master.json")
TASK_MASTER_PATH = os.path.join(PARENT_DIR, "task_master.json")

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_result")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_bundle_name_from_master(folder_name):
    if not os.path.exists(INDEX_MASTER_PATH):
        print("\n❌ 【エラー】 目次マスターが見つかりません。")
        sys.exit(1)
        
    with open(INDEX_MASTER_PATH, "r", encoding="utf-8") as f:
        index_master = json.load(f)

    match = re.search(r"(\d{2}-\d+)", folder_name)
    if not match:
        print(f"\n❌ 【エラー】 フォルダ名 ({folder_name}) から単元番号(XX-Y)を抽出できませんでした。")
        sys.exit(1)
        
    key = match.group(1)
    if key not in index_master:
        print(f"\n❌ 【エラー】 マスターファイルに単元 [{key}] の情報が登録されていません。")
        sys.exit(1)
        
    return index_master[key]["bundle_name"]


def assign_or_get_code(master_path, mext_code, node_name, summary, bundle_name, prefix=""):
    """知識(K)とタスク(T)を分離してマスターを管理・採番する関数"""
    master_data = {}
    if os.path.exists(master_path):
        try:
            with open(master_path, "r", encoding="utf-8") as f:
                master_data = json.load(f)
        except Exception:
            master_data = {}

    mext_code = str(mext_code).strip()
    if not mext_code:
        mext_code = "UNKNOWN"

    if mext_code not in master_data:
        master_data[mext_code] = []

    # 既存の同名ノードがあればそのコードを返す
    for item in master_data[mext_code]:
        if item["name"] == node_name:
            return item["branch_code"]

    # 連番の発行
    existing_nums = []
    for item in master_data[mext_code]:
        b_code = item.get("branch_code", f"_{prefix}000")
        match = re.search(r"_([A-Z]?)(\d+)", b_code)
        if match:
            existing_nums.append(int(match.group(2)))
    
    next_num = max(existing_nums) + 1 if existing_nums else 1
    new_branch_code = f"_{prefix}{next_num:03d}"

    # 新規登録
    master_data[mext_code].append({
        "branch_code": new_branch_code,
        "name": node_name,
        "summary_snippet": summary[:100] if summary else "",
        "first_appeared_in": bundle_name
    })

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=2)

    return new_branch_code


def load_and_prepare_inputs():
    with open(TEXTBOOK_MD_PATH, "r", encoding="utf-8") as f:
        textbook_content = f.read()
        
    folder_name = os.path.basename(CURRENT_DIR)
    bundle_name = get_bundle_name_from_master(folder_name)
    
    if not os.path.exists(MEXT_DICT_PATH):
        raise FileNotFoundError(f"❌ マスター辞書(v2)が見つかりません: {MEXT_DICT_PATH}")
        
    with open(MEXT_DICT_PATH, "r", encoding="utf-8") as f:
        mext_master_dict = f.read()
        
    return textbook_content, mext_master_dict, bundle_name


def execute_integrated_ontology_analysis(textbook_content, mext_master_dict, bundle_name):
    print(f"\n🚀 [Phase 1] 動的オントロジー抽出を実行中（三元構造＆技能分離: {bundle_name}）...")

    prompt = f"""あなたは高等学校数学科の教材分析・学習オントロジー構築のエキスパートです。
以下の「教材データ」と「指導要領マスター辞書」を解析し、学習指導要領が示す「三つの柱」および数学的活動（体系化、説明、発見）を動的に可視化する【高度なナレッジグラフ（ノードとエッジ）】を構築してください。

【対象単元】: {bundle_name}

【★最重要: ノード（頂点）の抽出・分離ルール★】
概念を以下の4つの型（Type）に厳密に分離して抽出してください。
単なるキーワード抽出ではなく、「数学的見方・考え方」による知識の変容（パラダイムシフト）を表現することが目的です。

1. `foundation_knowledge` (基礎知識ノード / 名詞ベース)
   - 条件によって揺らがない純粋な概念、定義、用語、公式。（例：「単項式」「係数」「定数項（数だけの項）」）
   - `parent_concept` (Level 3) として属する一般的な高校数学の標準用語を1つ指定。
   - 辞書から `pillar` が `knowledge_skill` または `general` のコードを紐付け。

2. `perspective_condition` (視点・条件ノード / レンズ)
   - 基礎知識を相対化させ、生徒に「見方・考え方」の切り替えを要求する条件やルール。（例：「ある文字に着目するルール」）
   - 辞書から `pillar` が `thinking_judgment` のコードを紐付け（事象の本質を認識する力）。

3. `derived_knowledge` (再構成された知識ノード)
   - 基礎知識に「視点・条件」を通した結果、新しく変容した知識。（例：「特定の文字に着目した場合の定数項（文字を含む）」）
   - これにより「定数項＝数字だけ」という固定観念が破られるパラダイムシフトを表現。
   - `parent_concept` を指定。

4. `task_nodes` (技能ノード / 動詞ベース)
   - 生徒が知識を用いて実際に行う具体的な計算手順や操作アクション。（例：「特定の文字に着目して多項式の次数と定数項を特定する」）
   - 辞書から `pillar` が `knowledge_skill` または `thinking_judgment` のコードを紐付け。

【★最重要: エッジ（辺）の抽出ルール★】
抽出したノード間の「思考と活動の軌跡（関係性）」を `edges` として定義してください。
関係性 (`relation_type`) は以下のいずれかから厳密に選択してください。

- `part_of` (構成要素): 知識が別の知識の一部である場合。
- `is_a` (特殊例・分類): 知識が別の知識の特殊な状態である場合。
- `subsumes` (包摂・統合): 上位概念が下位概念を包み込む場合（例：「整式」が「多項式」を subsumes）。
- `relative_to` (相対化される): 基礎知識が「視点・条件ノード」によって意味を変えられる関係。
- `applies_condition` (条件の適用): 視点・条件から、再構成された知識へと向かう推論のプロセス。
- `requires_logical` (論理的判断の要求): 再構成された知識を用いて、高度な技能（タスク）を実行する関係。
- `applied_to` (単純適用): 基礎知識を用いて、基本的な技能（タスク）を実行する関係。
- `explanation` (理由・説明): 例外やルールの論理的な理由付け（例：「0の扱い」）。
- `prerequisite` (前提知識): 技能や知識を学ぶために不可欠な過去の知識（多重化可能）。

【★絶対ルール: LaTeXとJSONエスケープ★】
`summary`や`reasoning`、`question_text`等のテキスト内にLaTeX数式（$...$）を含める場合、**必ずバックスラッシュを二重にエスケープ（例: \\\\frac, \\\\subset）** してください。JSONとしてInvalidにならないよう細心の注意を払ってください。

---
■ 教材Markdownデータ:
{textbook_content}

■ 指導要領マスター辞書 (JSON / 関連部分):
{mext_master_dict}
---

【出力JSONフォーマット】:
以下のスキーマに厳密に従って出力してください。Markdownのコードブロックは使用せず、JSON文字列のみを出力してください。

{{
  "bundle_name": "{bundle_name}",
  "nodes": {{
    "foundation_knowledge": [
      {{ "node_id": "FK1", "name": "...", "summary": "...", "parent_concept": "...", "mext_code": "..." }}
    ],
    "perspective_condition": [
      {{ "node_id": "PC1", "name": "...", "summary": "...", "mext_code": "..." }}
    ],
    "derived_knowledge": [
      {{ "node_id": "DK1", "name": "...", "summary": "...", "parent_concept": "...", "mext_code": "..." }}
    ],
    "tasks": [
      {{ "node_id": "T1", "name": "...", "summary": "...", "mext_code": "..." }}
    ]
  }},
  "edges": [
    {{
      "source_id": "FK1",
      "target_id": "T1",
      "relation_type": "applied_to | part_of | is_a | subsumes | relative_to | applies_condition | requires_logical | explanation | prerequisite",
      "reasoning": "なぜこの関係性が成り立つかの数学的・論理的理由（LaTeXエスケープ厳守）"
    }}
  ],
  "questions": [
    {{
      "question_number": "問題番号",
      "question_text": "問題文（LaTeXエスケープ厳守）",
      "linked_task_ids": ["T1"] 
    }}
  ]
}}
"""
    
    ontology_json = generate_content_and_parse_json(prompt)

    print("   🌐 分離型マスター（知識・タスク）と照合・採番中...")
    nodes_group = ontology_json.get("nodes", {})
    
    # --- 知識(K)系ノードの採番 ---
    knowledge_lists = [
        nodes_group.get("foundation_knowledge", []),
        nodes_group.get("perspective_condition", []),
        nodes_group.get("derived_knowledge", [])
    ]
    for k_list in knowledge_lists:
        for node in k_list:
            m_code = node.get("mext_code", "")
            name = node.get("name", "")
            summary = node.get("summary", "")
            node["branch_code"] = assign_or_get_code(KNOWLEDGE_MASTER_PATH, m_code, name, summary, bundle_name, prefix="K")

    # --- タスク(T)系ノードの採番 ---
    for node in nodes_group.get("tasks", []):
        m_code = node.get("mext_code", "")
        name = node.get("name", "")
        summary = node.get("summary", "")
        node["branch_code"] = assign_or_get_code(TASK_MASTER_PATH, m_code, name, summary, bundle_name, prefix="T")

    return ontology_json


def main():
    print("=== 🏁 【Ver 13.0 三元構造＆技能分離 オントロジー対応】Phase 1 起動 ===")
    print(f"   🔑 読み込み済み有効APIキー数: {len(API_KEYS)} 個")
    first_key_masked = f"{API_KEYS[0][:6]}...{API_KEYS[0][-4:]}" if len(API_KEYS[0]) > 10 else "INVALID"
    print(f"   👉 現在使用中のキー: {first_key_masked}")

    textbook_content, mext_master_dict, bundle_name = load_and_prepare_inputs()
    
    ontology_output = execute_integrated_ontology_analysis(textbook_content, mext_master_dict, bundle_name)

    # 最終出力用JSONの成形
    final_knowledge_graph = {
        "metadata": {"bundle_name": bundle_name, "engine_version": "13.0_dynamic_ontology", "model_used": MODEL_NAME},
        "nodes": ontology_output.get("nodes", {}),
        "edges": ontology_output.get("edges", []),
        "questions": ontology_output.get("questions", [])
    }
    
    output_filepath = os.path.join(OUTPUT_DIR, "final_knowledge_graph.json")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_knowledge_graph, f, ensure_ascii=False, indent=2)
    print(f"🎉 処理完了！分離型オントロジーの抽出が完了しました。 💾 保存先: {output_filepath}")

if __name__ == "__main__":
    main()