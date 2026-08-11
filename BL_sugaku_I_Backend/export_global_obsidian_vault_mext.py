import os
import re
import json
import zipfile
import shutil

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR_NAME = "Global_Obsidian_Vault_MEXT"
VAULT_PATH = os.path.join(PARENT_DIR, VAULT_DIR_NAME)
DB_PATH = os.path.join(PARENT_DIR, "global_vector_db_cache.json")

def clean_filename(name):
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    return cleaned if cleaned else "untitled"

def clean_tag_name(name):
    cleaned = re.sub(r'[\s\\/*?:"<>|()\[\]{}.#,\'`$=+~!@%^&-]', "_", name)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned if cleaned else "untitled"

EDGE_LABELS = {
    "part_of": "🧩 構成要素 (part_of)",
    "is_a": "🏷️ 特殊例・分類 (is_a)",
    "subsumes": "📦 包摂・統合 (subsumes)",
    "relative_to": "🪞 相対化される (relative_to)",
    "applies_condition": "👓 条件の適用・レンズ (applies_condition)",
    "requires_logical": "🧠 論理的判断の要求 (requires_logical)",
    "applied_to": "🛠️ 単純適用 (applied_to)",
    "explanation": "💬 理由・説明 (explanation)",
    "prerequisite": "⏪ 前提知識 (prerequisite)"
}

TYPE_PREFIX = {
    "foundation_knowledge": "知識",
    "perspective_condition": "視点",
    "derived_knowledge": "再構成",
    "tasks": "タスク"
}

def main():
    print("🚀 [Ver 14.3 講義名ハブノード・topic_nameタグ追加版] Obsidian Vault パッケージ化を開始します...")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ {DB_PATH} が見つかりません。先に build_vector_db.py を実行してください。")

    if os.path.exists(VAULT_PATH): 
        shutil.rmtree(VAULT_PATH)
    os.makedirs(VAULT_PATH, exist_ok=True)

    print("   📖 グローバルベクトルDBを読み込み中...")
    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    nodes = db.get("global_concept_nodes", {})
    questions = db.get("global_question_nodes", {})
    video_catalog = db.get("global_video_catalog", {})

    id_to_filename = {}
    name_to_id = {}
    used_names = set()
    
    # 🌟 講義名（bundle_name）ごとに中身を集計する辞書を新設
    global_bundles = {}

    for nid, ndata in nodes.items():
        prefix = TYPE_PREFIX.get(ndata["type"], "概念")
        base_name = f"[{prefix}] {clean_filename(ndata['name'])}"
        if base_name in used_names:
            base_name = f"{base_name}_{nid}"
        used_names.add(base_name)
        id_to_filename[nid] = base_name
        name_to_id[ndata['name']] = nid

    for qid, qdata in questions.items():
        b_name = clean_filename(qdata.get('bundle_name', 'Unknown'))
        q_num = qdata.get('question_number', 'X')
        base_name = f"[問題] {b_name}_{q_num}"
        id_to_filename[qid] = base_name

    global_parent_concepts = {}
    global_videos = {}
    node_children = {nid: set() for nid in nodes}

    for nid, ndata in nodes.items():
        p_name = ndata.get("parent_concept", "未分類")
        if p_name in name_to_id:
            node_children[name_to_id[p_name]].add(id_to_filename[nid])

    print("   🧠 ノード群 (4つの分類) のMarkdownを生成中...")
    for nid, ndata in nodes.items():
        filename = id_to_filename[nid]
        p_name = ndata.get("parent_concept", "未分類")
        parent_link_str = ""

        if p_name != "未分類":
            if p_name in name_to_id:
                parent_link_str = f"[[{id_to_filename[name_to_id[p_name]]}]]"
            else:
                p_filename = f"【親概念】{clean_filename(p_name)}"
                parent_link_str = f"[[{p_filename}]]"
                if p_name not in global_parent_concepts:
                    global_parent_concepts[p_name] = set()
                global_parent_concepts[p_name].add(filename)

        all_videos = ndata.get("main_videos", []) + ndata.get("review_videos", [])
        for v in all_videos:
            v_file = v.get("video_file")
            if v_file:
                if v_file not in global_videos:
                    global_videos[v_file] = {"concepts": set(), "tasks": set(), "questions_direct": set(), "questions_prereq": set()}
                if ndata.get("type") == "tasks":
                    global_videos[v_file]["tasks"].add(f"[[{filename}]]")
                else:
                    global_videos[v_file]["concepts"].add(f"[[{filename}]]")

        pillar_tag = clean_tag_name(ndata.get("pillar", "未定義"))
        node_type = ndata.get("type", "unknown")
        
        content = f"---\n"
        content += f"tags:\n"
        content += f"  - node/{node_type}\n"
        content += f"  - pillar/{pillar_tag}\n"
        content += f"---\n"
        content += f"# {filename}\n\n"

        content += f"## 🏛️ オントロジー基本情報\n"
        content += f"- **役割分類**: {ndata.get('type_label', '')}\n"
        content += f"- **役割定義**: {ndata.get('role_desc', '')}\n"
        content += f"- **三つの柱**: {ndata.get('pillar', '')}\n"
        
        bundle_name_clean = clean_filename(ndata.get('bundle_name', 'Unknown_Bundle'))
        content += f"- **🎓 スタディサプリの講義名**: [[{bundle_name_clean}]]\n"
        
        # 🌟 講義名（bundle_name）のハブにノードを追加
        if bundle_name_clean not in global_bundles:
            global_bundles[bundle_name_clean] = {"nodes": set(), "questions": set(), "videos": set()}
        global_bundles[bundle_name_clean]["nodes"].add(f"[[{filename}]]")
        
        if parent_link_str:
            content += f"- **上位概念**: {parent_link_str}\n"
        
        summary_text = ndata.get('summary', '').replace('\\n', '\n')
        content += f"\n> **【概要】**\n> {summary_text}\n\n"

        if ndata.get("mext_code"):
            content += f"## 📚 指導要領アライメント\n"
            content += f"- **階層**: {ndata.get('mext_hierarchy', '')}\n"
            content += f"- **コード**: `{ndata['mext_code']}`\n"
            content += f"\n> **【公式テキスト】**\n> {ndata.get('mext_official_text', '')}\n"
            content += f"\n> **【解説要約】**\n> {ndata.get('mext_explanation', '')}\n\n"

        if node_children[nid]:
            content += f"## 🔽 属する知識・タスク (下位概念)\n"
            for child in sorted(list(node_children[nid])):
                content += f"- [[{child}]]\n"
            content += "\n"

        content += f"## 🔗 思考の軌跡 (ネットワーク)\n"
        
        content += f"### ⬅️ 入ってくる関係 (前提・適用される条件など)\n"
        in_edges = ndata.get("incoming_edges", {})
        has_in = False
        for rel, items in in_edges.items():
            if items:
                has_in = True
                content += f"#### {EDGE_LABELS.get(rel, rel)}\n"
                for item in items:
                    src_id = item.get("source_id")
                    src_name = id_to_filename.get(src_id, src_id)
                    content += f"- [[{src_name}]]\n"
                    if item.get("reasoning"):
                        content += f"  - 💡 理由: {item['reasoning'].replace('\\n', '\n')}\n"
        if not has_in:
            content += "- 特記なし\n"
        content += "\n"

        content += f"### ➡️ 出ていく関係 (応用・適用先など)\n"
        out_edges = ndata.get("outgoing_edges", {})
        has_out = False
        for rel, items in out_edges.items():
            if items:
                has_out = True
                content += f"#### {EDGE_LABELS.get(rel, rel)}\n"
                for item in items:
                    tgt_id = item.get("target_id")
                    tgt_name = id_to_filename.get(tgt_id, tgt_id)
                    content += f"- [[{tgt_name}]]\n"
                    if item.get("reasoning"):
                        content += f"  - 💡 理由: {item['reasoning'].replace('\\n', '\n')}\n"
        if not has_out:
            content += "- 特記なし\n"
        content += "\n"

        main_videos = ndata.get("main_videos", [])
        review_videos = ndata.get("review_videos", [])
        
        if main_videos or review_videos:
            content += f"## 🎬 紐づく講義・解説動画\n\n"
            
            if main_videos:
                content += f"### 💡 【メイン教材】この概念・タスクを直接学ぶ動画\n"
                for v in main_videos:
                    v_file = v.get("video_file", "")
                    time_str = f"`{v.get('start_time', '')}`〜`{v.get('end_time', '')}`"
                    content += f"- **[[【動画】{clean_filename(v_file)}]]** ({time_str})\n"
                    if v.get("explanation_summary"):
                        exp_text = v['explanation_summary'].replace('\\n', '  \n    ')
                        content += f"  - 💬 解説要約: {exp_text}\n"
                content += "\n"
                
            if review_videos:
                content += f"### ⏪ 【前提・復習】この概念を前提知識として利用している動画\n"
                for v in review_videos:
                    v_file = v.get("video_file", "")
                    time_str = f"`{v.get('start_time', '')}`〜`{v.get('end_time', '')}`"
                    content += f"- **[[【動画】{clean_filename(v_file)}]]** ({time_str})\n"
                    if v.get("explanation_summary"):
                        exp_text = v['explanation_summary'].replace('\\n', '  \n    ')
                        content += f"  - 💬 解説要約: {exp_text}\n"
                content += "\n"

        q_texts = ndata.get("aligned_questions_text", [])
        if q_texts:
            content += f"## 📝 関連する演習問題\n"
            for q_text in q_texts:
                clean_q_text = q_text.replace('\\n', '\n')
                content += f"```text\n{clean_q_text}\n```\n"

        with open(os.path.join(VAULT_PATH, f"{filename}.md"), "w", encoding="utf-8") as f:
            f.write(content)

    print("   📝 問題ノードのMarkdownを生成中...")
    for qid, qdata in questions.items():
        filename = id_to_filename[qid]
        b_name = clean_filename(qdata.get("bundle_name", "Unknown_Bundle"))

        # 🌟 講義名（bundle_name）のハブに問題を追加
        if b_name not in global_bundles:
            global_bundles[b_name] = {"nodes": set(), "questions": set(), "videos": set()}
        global_bundles[b_name]["questions"].add(f"[[{filename}]]")

        for v in qdata.get("aligned_videos", []):
            v_file = v.get("video_file")
            align_type = v.get("alignment_type", "")
            if v_file:
                if v_file not in global_videos:
                    global_videos[v_file] = {"concepts": set(), "tasks": set(), "questions_direct": set(), "questions_prereq": set()}
                if align_type in ["direct_explanation", "task_walkthrough"]:
                    global_videos[v_file]["questions_direct"].add(f"[[{filename}]]")
                else:
                    global_videos[v_file]["questions_prereq"].add(f"[[{filename}]]")

        q_text = qdata.get('question_text', '').replace('\\n', '\n')
        a_text = qdata.get('answer_text', '').replace('\\n', '\n')

        content = f"---\n"
        content += f"tags:\n  - node/question\n"
        content += f"---\n"
        content += f"# {filename}\n\n"
        
        content += f"**🎓 スタディサプリの講義名**: [[{b_name}]]\n\n"
        
        content += f"## 📝 問題文\n{q_text}\n\n"
        content += f"## 💡 解説・解答\n{a_text}\n\n"
        
        if qdata.get("aligned_videos"):
            content += f"## 🎬 紐づく解説動画\n"
            for v in qdata["aligned_videos"]:
                v_file = v.get("video_file", "")
                time_str = f"`{v.get('start_time', '')}`〜`{v.get('end_time', '')}`"
                content += f"- **[[【動画】{clean_filename(v_file)}]]** ({time_str})\n"
                
                if v.get("explanation_summary"):
                    exp_text = v['explanation_summary'].replace('\\n', '  \n    ')
                    content += f"  - 💬 解説要約: {exp_text}\n"

        with open(os.path.join(VAULT_PATH, f"{filename}.md"), "w", encoding="utf-8") as f:
            f.write(content)

    print("   🌐 親概念ハブと動画ノードを生成中...")
    for p_name, children in global_parent_concepts.items():
        filename = f"【親概念】{clean_filename(p_name)}"
        content = f"---\ntags:\n  - node/parent_concept\n---\n# {filename}\n\n## 🔽 属する知識・タスク\n"
        for child in sorted(list(children)):
            content += f"- [[{child}]]\n"
        with open(os.path.join(VAULT_PATH, f"{filename}.md"), "w", encoding="utf-8") as f:
            f.write(content)

    for v_file, v_data in global_videos.items():
        filename = f"【動画】{clean_filename(v_file)}"
        content = f"---\ntags:\n  - node/video\n---\n# {filename}\n\n"
        
        catalog_info = video_catalog.get(v_file, {})
        b_name_catalog = clean_filename(catalog_info.get("bundle_name", "Unknown_Bundle"))
        
        if b_name_catalog:
            content += "## 🎓 スタディサプリの講義名\n"
            content += f"- [[{b_name_catalog}]]\n\n"
            
            # 🌟 講義名（bundle_name）のハブに動画を追加
            if b_name_catalog not in global_bundles:
                global_bundles[b_name_catalog] = {"nodes": set(), "questions": set(), "videos": set()}
            global_bundles[b_name_catalog]["videos"].add(f"[[{filename}]]")
            
        if v_data.get("concepts") or v_data.get("tasks"):
            content += "## 🧠 📘 関連する知識・タスク（インプット講義）\n"
            for n in sorted(list(v_data.get("concepts", set()))): 
                content += f"- {n}\n"
            for n in sorted(list(v_data.get("tasks", set()))): 
                content += f"- {n}\n"
            content += "\n"
            
        if v_data.get("questions_direct") or v_data.get("questions_prereq"):
            content += "## 📝 紐付けられた確認問題\n"
            if v_data.get("questions_direct"):
                content += "### 📗 【例題演習】（直接解説している問題）\n"
                for q in sorted(list(v_data["questions_direct"])):
                    content += f"- {q}\n"
                content += "\n"
            if v_data.get("questions_prereq"):
                content += "### 📘 【概念理解】（前提概念として紐づく問題）\n"
                for q in sorted(list(v_data["questions_prereq"])):
                    content += f"- {q}\n"
                content += "\n"
        
        with open(os.path.join(VAULT_PATH, f"{filename}.md"), "w", encoding="utf-8") as f:
            f.write(content)

    # 🌟 講義名（所属単元）のハブノードを生成
    print("   🎓 講義名（所属単元）のハブノードを生成中...")
    for b_name, b_data in global_bundles.items():
        if b_name == "Unknown_Bundle":
            continue
            
        content = f"---\ntags:\n  - node/topic_name\n---\n# {b_name}\n\n"
        
        if b_data["videos"]:
            content += "## 🎬 講義動画\n"
            for v in sorted(list(b_data["videos"])):
                content += f"- {v}\n"
            content += "\n"
            
        if b_data["nodes"]:
            content += "## 🧠 抽出された知識・タスク\n"
            for n in sorted(list(b_data["nodes"])):
                content += f"- {n}\n"
            content += "\n"
            
        if b_data["questions"]:
            content += "## 📝 演習問題\n"
            for q in sorted(list(b_data["questions"])):
                content += f"- {q}\n"
            content += "\n"
            
        with open(os.path.join(VAULT_PATH, f"{b_name}.md"), "w", encoding="utf-8") as f:
            f.write(content)

    print("   📦 ZIPパッケージに圧縮中...")
    zip_path = os.path.join(PARENT_DIR, f"{VAULT_DIR_NAME}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(VAULT_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, PARENT_DIR)
                zipf.write(file_path, arcname)

    print(f"🎉 🎉 【成功】Obsidian Vault（スタディサプリ講義名ハブ表示版）の生成完了！\n💾 保存先: {zip_path}")

if __name__ == "__main__":
    main()