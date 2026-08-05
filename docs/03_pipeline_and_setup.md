# 🔄 パイプライン処理フロー & セットアップ手順

本ドキュメントでは、スタサプRAGエンジンのデータ生成パイプライン（Phase 0〜Phase 3）の仕組みおよび、環境構築・実行手順について解説します。

---

## 🔄 パイプライン処理フロー

各PARTフォルダー内で順次実行される個別解析パイプラインおよび、バックエンド全体の統合処理フローです。

```text
[ PDF / MP4 / VTT 元データ ]
           │
           ▼
[ Phase 0 ] phase0_pdf_to_md.py
   └─ PDFテキスト抽出・LaTeX数式変換・図形/グラフの言語化 (動的再アップロード対応)
           │
           ▼
[ Phase 1 ] phase1_text_analysis_ontology.py (Ver 12.1)
   └─ 4階層オントロジー抽出・枝番自動採番・観点タグ・重み付き前提知識 (JSON自動修復対応)
           │
           ▼
[ Phase 2 ] phase2_video_analysis.py
   └─ 動画役割自動判定・極細チャプター分割・板書OCR生成 (403検知・リカバリー機構付)
           │
           ▼
[ Phase 3 ] phase3_alignment_graph.py
   └─ 概念・問題・動画の三位一体アライメント & ファジーマッチ安全結合 (JSON自動修復対応)
           │
           ▼
 ─────────────────────────────────────────────────────────────
 [ バックエンド統合処理 ]
   ├─ build_vector_db.py ➔ グローバルベクトルDB (global_vector_db_cache.json) 構築
   ├─ export_global_obsidian_vault_mext.py ➔ Obsidian Vault ZIP 出力
   ├─ generate_mock_logs.py ➔ GNN-KT検証用 ダミー学習ログ (mock_student_logs.json) 出力
   └─ visualize_syllabus_content.py ➔ 対話型シラバスマッチング可視化CLI
```

---

## ⚙️ 各Phaseの役割と堅牢化仕様

本システムは、長時間のバッチ処理を安定稼働させるため、API制限（429）やJSONパースエラーに対する強力な自動リカバリー機構を備えています。

* **Phase 0 (`phase0_pdf_to_md.py`)**:
  * 教材PDFをMarkdownへ高精度変換。
  * 関数グラフ、数直線、ベン図、幾何図形等を `[図の説明: 〇〇]` の形でメタデータテキスト化します。
  * **[堅牢化]** API制限やアカウント変更に伴うファイルアクセス権限エラー（HTTP 403）を検知した場合、古いキャッシュを破棄し、新しいキーでPDFを自動再アップロードして解析を継続します。
* **Phase 1 (`phase1_text_analysis_ontology.py` - Ver 12.1 GNN-KT & メタデータ拡張対応)**:
  * 教材Markdownから具体的アクション（Level 4）や親概念（Level 3）、前提知識を抽出。
  * 枝番管理マスター（`concept_branch_master.json`）とリアルタイム連携。既存の登録概念を「お手本」として動的注入し、表記揺れを防ぎながら自動採番を行います。
  * 前提知識を親概念レベル（Level 3）に正規化し、依存タイプと理由を構造化出力。
  * **[堅牢化]** LLM出力のJSONパースエラー（LaTeXのバックスラッシュエスケープ忘れ等）を検知し、システム側で自動修復してパースを試みる機構を搭載。
* **Phase 2 (`phase2_video_analysis.py`)**:
  * 動画と字幕から講義の役割（`concept_lecture`, `exercise_walkthrough`, `concept_application`）を自動推定。
  * 1〜3分単位のミクロセグメント化と黒板の板書OCR（LaTeX形式）を出力します。
  * **[堅牢化]** API速度制限（HTTP 429）時のキー自動切り替え、およびキー変更に伴う動画アクセス拒否エラー（HTTP 403）を検知し、動画の再アップロードと解析ループの再開を無人で行います。
* **Phase 3 (`phase3_alignment_graph.py`)**:
  * Phase 1 の知識グラフと Phase 2 の動画タイムラインをマッチングし、`final_knowledge_graph_complete.json` を生成。
  * Phase 1側の新構造データ（観点タグ・重み付き前提知識）を引き継ぎ、タイムスタンプのずれに対してはファジーマッチ（最寄時間検索）を実行します。
  * **[堅牢化]** Phase 1同様、LaTeXエスケープ起因のJSONパースエラー自動修復に対応。
* **一括実行スクリプト (`run_single_part_batch.py`)**:
  * 選択されたPARTフォルダーに対して処理を連続実行。
  * テキストオントロジー改修時などは、Phase 2（動画解析）の再処理をスキップし、Phase 1 ➔ Phase 3 のみを選択実行してAPI費用と時間を大幅に短縮可能です。

---

## 🛠️ セットアップ & 実行手順

### 1. 依存ライブラリのインストール
```bash
pip install python-dotenv google-genai numpy requests streamlit plyer
# (※ Excel・PDF読み込み用の各種モジュール含む)
```

### 2. 環境変数の設定 (`.env`) と複数APIキーのローテーション仕様
バックエンド・フロントエンドそれぞれのディレクトリに `.env` ファイルを作成し、Gemini APIキーを設定します。

本システムは長時間のバッチ処理を前提としているため、**複数のAPIキーを用いた自動ローテーション**に対応しています。クォーテーション（`"` や `'`）は不要です。

```env
# GEMINI_API_KEYS (複数形) または GEMINI_API_KEY (単数形) のいずれかで指定。
# カンマ区切りで複数のキーを登録することで、制限到達時（429エラー時）にシステムが自動で次のキーへ切り替えて処理を続行します。
GEMINI_API_KEYS=AIzaSyA123...,AIzaSyB456...,AIzaSyC789...
```
*(※システム起動時に `.env` の内容で環境変数を強制上書き（override）し、BOMやクォーテーションなどの不可視文字も自動除去する安全設計となっています。最大リトライ回数は「登録されたキーの数 × 2」に動的設定されます。)*

### 3. バックエンド処理の実行
① **マスターデータの構築（初回のみ）**
```bash
python build_mext_master_dict.py
python build_index_master.py
# (指導要領辞書および目次マスターを生成)
```

② **PARTデータの一括解析バッチ実行**
```bash
python run_single_part_batch.py
# (対話メニューから実行対象のPART番号を入力。既存動画解析がある場合は Phase 1 ➔ Phase 3 を選択実行)
```

③ **グローバルベクトルDBおよびObsidian Vaultの構築**
```bash
python build_vector_db.py
python export_global_obsidian_vault_mext.py
# (GNN-KT対応の global_vector_db_cache.json および Global_Obsidian_Vault_MEXT.zip が生成されます)
```

④ **ダミー学習ログの生成（検証用）**
```bash
python generate_mock_logs.py
# (DBのオントロジー構造を読み込み、生徒の正誤ログ mock_student_logs.json を生成します)
```

⑤ **シラバスマッチング可視化ツールのテスト実行**
```bash
python visualize_syllabus_content.py
# (ターミナル上で対話的にシラバステキストを入力し、紐づくLevel 3/4概念や動画・問題をリアルタイム検証可能)
```

### 4. フロントエンド (AIチューター) の起動
バックエンドで生成された `global_vector_db_cache.json` を `BL_sugaku_I_Frontend/` フォルダーへコピーします。

フロントエンドディレクトリでアプリケーションを起動します。
```bash
cd BL_sugaku_I_Frontend

# 生徒向け AIチューター アプリの起動
streamlit run app.py

# 教員向け Semantic Zoom ダッシュボードの起動
streamlit run app_ft.py
```