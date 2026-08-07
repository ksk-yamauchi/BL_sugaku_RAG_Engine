# 🎓 スタサプ RAG ナレッジエンジン ＆ 次世代AIチューター

**〜 「GNN-KT（動詞）」×「オントロジー（名詞）」のデュアルエンジンが実現する、個別最適なAI指導・学習アシスタントパートナー 〜**

---

## 🌟 プロジェクト概要

本プロジェクトは、スタディサプリの各種教材データ（PDF、講義動画、字幕、文部科学省学習指導要領コード）を統合解析し、**「GNN-KT（知識追跡）の計算対象となるタスク（動詞）」**と**「意味ネットワークを構成する知識・視点（名詞）」**を完全に両立させた【デュアルエンジン・オントロジー】を構築するプロジェクトです。[cite: 30, 33]

さらに、文科省の「三つの柱（知識及び技能、思考力・判断力等など）」をデータベースのメタデータとして直接組み込み、テキスト教材の静的な知識にとどまらず、動画内の先生の「口頭解説」や「板書」から暗黙知をAIが動的に発見し、自己増殖（Dynamic Ontology）する次世代の教育コンテンツ基盤を実現しています。[cite: 30, 33]

単なるテキスト類似度による検索を超え、「なぜその問題で躓いているのか」「前提となるどの概念や見方に戻って復習すべきか」を4つの役割別ノードと9つの関係性エッジからなるグラフ構造上で推論し、先生（for Teachers：FT）と生徒（Learn）双方に最高品質の教育体験を提供します。[cite: 30, 33]

---

## 🚀 主な機能と次世代機能構想

### 🏫 教員向け機能（FT: for Teachers）
* **習熟度の可視化（Semantic Zoom ＆ GNN-KT）**:
  クラス全体の学習ログから、単元レベル（マクロ）から極小な躓きアクション（ミクロ）までGoogleマップのように自在にズームイン・アウトして真の不理解原因を特定。[cite: 30, 33]
* **年間学習計画（シラバス）マッチング**:
  校内の年間指導計画表（PDF/テキスト）を流し込むだけで、シラバス特有の省略表現を自動補完し、毎週の授業に対応する最適教材プレイリストを自動生成。[cite: 30, 33]
* **マルチモーダル画像RAG（自作プリントマッチング）**:
  先生の自作プリント画像から数理概念を即座に判定し、100%信頼できる公式類題と解説動画秒数をサジェスト。[cite: 30, 33]

### 🎓 生徒向け機能（Learn: AIパーソナルチューター）
* **対話型 / マルチモーダル AIチューター**:
  解けない問題や手書きノートの画像から、直ちに答えを教えるのではなく「ヒントとなる動画の秒数」と「前提知識」を提示して自律的解決を支援。[cite: 30, 33]
* **適応型学習ナビ (Adaptive Journey)**:
  GNN-KTモデルによる潜在的弱点の推論に基づき、忘却曲線に合わせた「今日やるべき復習・学習ミッション」を全自動生成。[cite: 30, 33]
* **RPG風スキルツリー可視化**:
  学習の現在地と概念の繋がり（必須前提・補足前提）をスキルツリーとしてインタラクティブに可視化。[cite: 30, 33]

---

## 📁 ディレクトリ構成

本システムは、拡張性・セキュリティ・デプロイの容易性を確保するため、**バックエンド（解析・DB構築）**と**フロントエンド（UIアプリ）**に分離して設計されています。[cite: 30, 33]

```text
BL_sugaku/
├── 📄 README.md                             # ★本ドキュメント (プロジェクト概要・Quick Start)
├── 📄 CHANGELOG.md                          # バージョン・変更履歴
├── 📄 requirements.txt                      # プロジェクト共通 依存ライブラリ定義
├── 📁 docs/                                # 詳細仕様書・設計ドキュメント群
│   ├── 📄 01_system_architecture.md         # システムアーキテクチャ & オントロジー構造仕様
│   ├── 📄 02_search_engine.md              # AIチューター検索アルゴリズム仕様
│   ├── 📄 03_pipeline_and_setup.md          # パイプライン処理フロー & セットアップ手順
│   ├── 📄 04_obsidian_guide.md              # Obsidian ナレッジグラフ可視化ガイド
│   ├── 📄 05_frontend_applications.md       # フロントエンドUI詳細仕様書
│   ├── 📄 06_image_rag_prompt_design.md     # マルチモーダル画像RAG 専用プロンプト仕様書
│   ├── 📄 07_syllabus_mapping_prompt.md     # シラバスマッチング 専用プロンプト仕様書
│   └── 📄 08_mock_log_simulation_spec.md    # ダミー学習ログ生成 ＆ GNN-KTシミュレーション仕様書
│
├── 🏭 BL_sugaku_I_Backend/                  # バックエンド (解析エンジン ＆ DB構築)
│   ├── 📄 .env                             # バックエンド用 APIキー設定 (複数キー自動ローテーション対応)
│   ├── 📄 mext_master_dict.v2.json         # 【自動生成】統合指導要領マスター辞書 (Pillar対応版)
│   ├── 📄 knowledge_master.json            # 【自動生成】全講共通 独自知識マスター
│   ├── 📄 task_master.json                 # 【自動生成】全講共通 独自タスクマスター
│   ├── 📄 lecture_index_master.json        # 全講共通 目次マスター
│   ├── 📜 build_mext_master_dict.py        # 指導要領辞書生成スクリプト
│   ├── 📜 build_index_master.py            # 目次マスター生成スクリプト
│   ├── 📜 run_single_part_batch.py         # PART一括解析バッチ処理スクリプト
│   ├── 📜 build_vector_db.py               # グローバルベクトルDB構築スクリプト (Ver 13.x 新オントロジー対応)
│   ├── 📜 export_global_obsidian_vault_mext.py # Obsidian用ZIPパッケージ出力スクリプト
│   ├── 📜 generate_mock_logs.py            # GNN-KT検証用 ダミー学習ログ生成スクリプト
│   ├── 📜 visualize_syllabus_content.py    # シラバス ➔ コンテンツ可視化テストスクリプト (対話型CLI)
│   ├── 📄 global_vector_db_cache.json      # 【成果物】統合ベクトルキャッシュDB
│   ├── 📄 mock_student_logs.json           # 【成果物】生成されたダミー学習ログ
│   ├── 📦 Global_Obsidian_Vault_MEXT.zip   # 【成果物】Obsidian Vault パッケージ
│   └── 📁 BL_sugaku_Ⅰ_XX-Y/                # PART単位の素材データ＆個別解析スクリプト
│       ├── 📄 BL_sugaku_I_XX-Y.pdf
│       ├── 📄 *.mp4 / *.vtt
│       ├── 📜 phase0_pdf_to_md.py          # Phase 0: PDF変換 (動的再アップロード対応)
│       ├── 📜 phase1_text_analysis_ontology.py # Phase 1: デュアルエンジン・オントロジー解析 (Ver 13.x)
│       ├── 📜 phase2_video_analysis.py     # Phase 2: 動画解析 (403リカバリー対応) 
│       └── 📜 phase3_alignment_graph.py    # Phase 3: 動的補完 & 粒度吸収アライメント (Ver 13.x)
│
└── 📱 BL_sugaku_I_Frontend/                 # フロントエンド (Streamlit Web UI)
    ├── 📄 .env                             # フロントエンド用 APIキー設定
    ├── 📜 app.py                           # 生徒向け Streamlit AIチューター アプリ
    ├── 📜 app_ft.py                        # 先生向け Semantic Zoom ダッシュボード (モック)
    └── 📄 global_vector_db_cache.json      # バックエンドから同期する検索用DB
```

---

## 🚀 クイックスタート (Quick Start)

### 1. 依存ライブラリの一括インストール

プロジェクトルート（`BL_sugaku/`）にて、以下のコマンドを実行し、必要な全パッケージをインストールします。[cite: 30, 33]

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定 (`.env`)

バックエンド（`BL_sugaku_I_Backend/`）およびフロントエンド（`BL_sugaku_I_Frontend/`）の各ディレクトリに `.env` ファイルを作成し、Gemini APIキーを設定します。[cite: 30, 33]
※カンマ区切りで複数のキーを登録することで、API制限到達時にシステムが自動でキーを切り替えて処理を継続します。[cite: 30, 33]

```env
GEMINI_API_KEYS=AIzaSyA123...,AIzaSyB456...,AIzaSyC789...
```

### 3. バックエンドデータの構築 (DB生成)

```bash
cd BL_sugaku_I_Backend

# ① PARTデータの一括解析パイプライン実行 (Phase 0〜3)
python run_single_part_batch.py

# ② 統合ベクトルDB (global_vector_db_cache.json) の構築
python build_vector_db.py

# ③ Obsidian用ナレッジグラフパッケージ (.zip) の出力
python export_global_obsidian_vault_mext.py

# ④ ダミー学習ログの生成（GNN-KT推論・適応型ナビの検証用）
python generate_mock_logs.py
```

### 4. フロントエンド (AIチューターアプリ / FTダッシュボード) の起動

生成された `global_vector_db_cache.json` をフロントエンドディレクトリへコピーして起動します。[cite: 30, 33]

```bash
# キャッシュDBのコピー
cp global_vector_db_cache.json ../BL_sugaku_I_Frontend/

# フロントエンドへ移動
cd ../BL_sugaku_I_Frontend

# 生徒向けアプリの起動
streamlit run app.py

# 先生向けダッシュボード(モック)の起動
streamlit run app_ft.py
```

ブラウザが立ち上がり、AIチューター画面やダッシュボード画面が表示されます。[cite: 30, 33]

---

## 📚 ドキュメントインデックス

より詳細な仕様や設計思想については、`docs/` フォルダー内の各種仕様書をご参照ください。[cite: 30, 33]

1. [🏛️ システムアーキテクチャ & オントロジー構造 (`docs/01_system_architecture.md`)](docs/01_system_architecture.md)
   * デュアルエンジン（GNN-KT×オントロジー）モデル、**三つの柱と4つのノード（辞書・レンズ・パラダイムシフト・ゴール）**の役割定義、思考の軌跡を表す**9つのエッジ**の詳細解説。[cite: 30, 33]
2. [🔍 検索エンジンの仕組み (`docs/02_search_engine.md`)](docs/02_search_engine.md)
   * サニタイズ処理、ハイブリッドスコアリング、Graph RAG探索アルゴリズム。[cite: 30, 33]
3. [🔄 パイプライン処理フロー & セットアップ手順 (`docs/03_pipeline_and_setup.md`)](docs/03_pipeline_and_setup.md)
   * Phase 0〜3の役割、複数APIキー自動ローテーション、403エラー自動復旧、JSON自動修復ロジック、詳細バッチ手順。[cite: 30, 33]
4. [🎨 Obsidian ナレッジグラフ設定ガイド (`docs/04_obsidian_guide.md`)](docs/04_obsidian_guide.md)
   * Obsidian上でのGraph View色分け設定ルール（観点別カラー対応）、必須/補足前提関係の可視化。[cite: 30, 33]
5. [📱 フロントエンドアプリケーション（UI）詳細仕様 (`docs/05_frontend_applications.md`)](docs/05_frontend_applications.md)
   * 生徒向けAIチューター(`app.py`)と教員向けダッシュボード(`app_ft.py`)のUI/UX設計思想および詳細機能仕様。[cite: 30, 33]
6. [📸 マルチモーダル画像RAG 専用プロンプト設計書 (`docs/06_image_rag_prompt_design.md`)](docs/06_image_rag_prompt_design.md)
   * 画像解析からスタサプオントロジーへ変換するGemini Vision用システムプロンプト。[cite: 30, 33]
7. [📅 シラバスマッチング 専用プロンプト設計書 (`docs/07_syllabus_mapping_prompt.md`)](docs/07_syllabus_mapping_prompt.md)
   * 学校独自のシラバス表記を補完し、スタサプオントロジーに正確に翻訳・紐付けするためのシステムプロンプト。[cite: 30, 33]
8. [📝 ダミー学習ログ生成 ＆ GNN-KTシミュレーション仕様書 (`docs/08_mock_log_simulation_spec.md`)](docs/08_mock_log_simulation_spec.md)
   * ペルソナ定義、必須/補足前提関係に基づく連鎖ペナルティのアルゴリズム、シミュレーション出力データのスキーマ。[cite: 30, 33]