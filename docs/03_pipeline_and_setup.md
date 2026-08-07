# 🔄 パイプライン処理フロー & セットアップ手順

本ドキュメントでは、スタサプRAGエンジンのデータ生成パイプライン（Phase 0〜Phase 3）の仕組みおよび、環境構築・実行手順について解説します。[cite: 31]

---

## 🔄 パイプライン処理フロー

各PARTフォルダー内で順次実行される個別解析パイプラインおよび、バックエンド全体の統合処理フローです。[cite: 31]

```text
[ PDF / MP4 / VTT 元データ ]
           │
           ▼
[ Phase 0 ] phase0_pdf_to_md.py
   └─ PDFテキスト抽出・LaTeX数式変換・図形/グラフの言語化 (動的再アップロード対応)
           │
           ▼
[ Phase 1 ] phase1_text_analysis_ontology.py (Ver 13.x)
   └─ デュアルエンジン（タスク/知識）抽出・マスター照合・自動採番 (JSON自動修復対応)
           │
           ▼
[ Phase 2 ] phase2_video_analysis.py
   └─ 動画役割自動判定・極細チャプター分割・板書OCR生成 (403検知・リカバリー機構付)
           │
           ▼
[ Phase 3 ] phase3_alignment_graph.py (Ver 13.x)
   └─ 動的オントロジー補完・粒度吸収アライメント & ファジーマッチ結合 (JSON自動修復対応)
           │
           ▼
 ─────────────────────────────────────────────────────────────
 [ バックエンド統合処理 ]
   ├─ build_vector_db.py ➔ グローバルベクトルDB構築 (Ver 13.x 対応・APIキー自動ローテーション搭載)
   ├─ export_global_obsidian_vault_mext.py ➔ Obsidian Vault ZIP 出力
   ├─ generate_mock_logs.py ➔ GNN-KT検証用 ダミー学習ログ (mock_student_logs.json) 出力
   └─ visualize_syllabus_content.py ➔ 対話型シラバスマッチング可視化CLI
```

---

## ⚙️ 各Phaseの役割と堅牢化仕様

本システムは、長時間のバッチ処理を安定稼働させるため、API制限（429）やJSONパースエラーに対する強力な自動リカバリー機構を備えています。[cite: 31]

* **Phase 0 (`phase0_pdf_to_md.py`)**:
  * 教材PDFをMarkdownへ高精度変換。[cite: 31]
  * 関数グラフ、数直線、ベン図、幾何図形等を `[図の説明: 〇〇]` の形でメタデータテキスト化します。[cite: 31]
  * **[堅牢化]** API制限やアカウント変更に伴うファイルアクセス権限エラー（HTTP 403）を検知した場合、古いキャッシュを破棄し、新しいキーでPDFを自動再アップロードして解析を継続します。[cite: 31]

* **Phase 1 (`phase1_text_analysis_ontology.py` - Ver 13.x デュアルエンジン対応)**:
  * 教材MarkdownからGNN-KTの計算対象となる「タスク（_Tノード）」と、解釈対象となる「知識・視点（_Kノード）」を厳格に分離して抽出します。[cite: 31]
  * 知識マスター（`knowledge_master.json`）およびタスクマスター（`task_master.json`）とリアルタイム連携し、表記揺れを防ぎながら自動採番を行います。[cite: 31]
  * **[堅牢化]** LLM出力のJSONパースエラー（LaTeXのバックスラッシュエスケープ忘れ等）を検知し、システム側で自動修復してパースを試みる機構を搭載。[cite: 31]

* **Phase 2 (`phase2_video_analysis.py`)**:
  * 動画と字幕から講義の役割（`concept_lecture`, `exercise_walkthrough`, `concept_application`）を自動推定。[cite: 31]
  * 1〜3分単位のミクロセグメント化と黒板の板書OCR（LaTeX形式）を出力します。[cite: 31]
  * **[堅牢化]** API速度制限（HTTP 429）時のキー自動切り替え、およびキー変更に伴う動画アクセス拒否エラー（HTTP 403）を検知し、動画の再アップロードと解析ループの再開を無人で行います。[cite: 31]

* **Phase 3 (`phase3_alignment_graph.py` - Ver 13.x 動的補完対応)**:
  * Phase 1の知識グラフとPhase 2の動画タイムラインをマッチングします。[cite: 31]
  * **[動的補完]** 動画内の解説からテキストにない「暗黙知」や「新しい視点」を発見した場合、知識ノードとしてグラフに自動追加（Dynamic Ontology）します。[cite: 31]
  * **[粒度吸収]** 動画がUI用に極細分割されている場合、動画のロールに応じて一連のプロセスを配列として既存タスク内に束ねて吸収させます。[cite: 31]
  * タイムスタンプのずれに対してはファジーマッチ（最寄時間検索）を実行します。[cite: 31]
  * **[堅牢化]** Phase 1同様、LaTeXエスケープ起因のJSONパースエラー自動修復に対応。[cite: 31]

* **グローバルDB構築 (`build_vector_db.py` - Ver 13.x 堅牢化対応)**:
  * 全PARTの出力結果を統合し、フロントエンド用のベクトルデータベースを構築します。
  * **[堅牢化] バージョンフィルター**: 読み込むJSONの `engine_version` を検証し、`13.x` 系の新構造データのみを処理。旧バージョンのデータ混入によるクラッシュやキメラ化を完全に防ぎます。
  * **[堅牢化] 動的モデル探索**: Google APIへリクエストを送り、現在利用可能な最新の埋め込みモデル（例: `text-embedding-004`）を自動で探索・設定します。
  * **[堅牢化] APIキー自動ローテーション**: 数千件におよぶノードと問題の連続ベクトル化において、API制限（429等）を検知すると登録された複数のキーを自動で切り替え、長時間の構築バッチを停止することなく完遂させます。

* **一括実行スクリプト (`run_single_part_batch.py`)**:
  * 選択されたPARTフォルダーに対して処理を連続実行。[cite: 31]
  * テキストオントロジー改修時などは、Phase 2（動画解析）の再処理をスキップし、Phase 1 ➔ Phase 3 のみを選択実行してAPI費用と時間を大幅に短縮可能です。[cite: 31]

---

## 🛠️ セットアップ & 実行手順

### 1. 依存ライブラリのインストール
```bash
pip install python-dotenv google-genai numpy requests streamlit plyer
# (※ Excel・PDF読み込み用の各種モジュール含む)
```

### 2. 環境変数の設定 (`.env`) と複数APIキーのローテーション仕様
バックエンド・フロントエンドそれぞれのディレクトリに `.env` ファイルを作成し、Gemini APIキーを設定します。[cite: 31]

本システムは長時間のバッチ処理を前提としているため、**複数のAPIキーを用いた自動ローテーション**に対応しています。クォーテーション（`"` や `'`）は不要です。[cite: 31]

```env
# GEMINI_API_KEYS (複数形) または GEMINI_API_KEY (単数形) のいずれかで指定。
# カンマ区切りで複数のキーを登録することで、制限到達時（429エラー時）にシステムが自動で次のキーへ切り替えて処理を続行します。
GEMINI_API_KEYS=AIzaSyA123...,AIzaSyB456...,AIzaSyC789...
```
*(※システム起動時に `.env` の内容で環境変数を強制上書き（override）し、BOMやクォーテーションなどの不可視文字も自動除去する安全設計となっています。最大リトライ回数は「登録されたキーの数 × 2」に動的設定されます。)*[cite: 31]

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
バックエンドで生成された `global_vector_db_cache.json` を `BL_sugaku_I_Frontend/` フォルダーへコピーします。[cite: 31]

フロントエンドディレクトリでアプリケーションを起動します。[cite: 31]
```bash
cd BL_sugaku_I_Frontend

# 生徒向け AIチューター アプリの起動
streamlit run app.py

# 教員向け Semantic Zoom ダッシュボードの起動
streamlit run app_ft.py
```