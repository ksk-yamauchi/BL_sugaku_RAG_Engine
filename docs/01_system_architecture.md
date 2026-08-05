# 🏛️ システムアーキテクチャ & オントロジー構造

本ドキュメントでは、スタサプRAGエンジンのシステム全体の構成および、教育コンテンツのメタデータモデル（4階層オントロジー ＋ GNN-KT構造）について解説します。

---

## 🧬 概念の4階層オントロジー構造

本システムでは、単なるキーワード検索にとどまらず、「国の指導基準 ➔ 一般的な数学概念 ➔ 生徒の具体的な躓き」という4段階の階層構造で知識をモデル化しています。

ID（枝番）自体を複雑な階層型にするのではなく、**「IDはフラットなまま、JSONの属性（プロパティ）で階層・観点・依存度を表現する」**設計を採用することで、システムの堅牢性と拡張性を担保しています。

### 📊 4階層オントロジーとメタデータ構造

| 階層レベル | 役割と意味合い | システム上の実装（ID / プロパティ） | メタデータ拡張プロパティ | 具体例 |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** (大項目) | 分野（国の基準） | **文科省16桁コード** の上位桁 | - | 「数と式」 |
| **Level 2** (中項目) | 単元（国の基準） | **文科省16桁コード** の下位桁 | - | 「因数分解」 |
| **Level 3** (親概念) | **一般的な数学用語**<br>（単元を跨ぐハブ） | **`parent_concept`**<br>（文字列タグ・専用ハブノード） | - | 「たすき掛け」「平方完成」 |
| **Level 4** (極小項目) | **具体的なアクション**<br>（生徒の実際の躓きポイント） | **枝番（Branch Code）**<br>（例: `_001`） | **`competency`**<br>(`knowledge_skill` / `thinking_judgment`) | 「分数が含まれる平方完成を処理する」<br>「文字が含まれるたすき掛け」 |

---

## 🗂️ 独自概念枝番マスター (`concept_branch_master.json`) のメタデータ仕様

全講（PART1〜15）を通して一意な枝番 ID（Level 4）を保持し、かつ文言の揺らぎや重複数値を自動防ぐため、`concept_branch_master.json` には以下の拡張メタデータが保持されます。

```json
{
  "8451503113000000": [
    {
      "branch_code": "_001",
      "concept_name": "単項式の係数と次数を見分ける",
      "summary_snippet": "数や文字の掛け合わせで表される単項式において、係数と次数を正しく特定する。",
      "first_appeared_in": "第1講 式の計算と展開 PART1 単項式と多項式"
    }
  ]
}
```

* **`summary_snippet`**: 概念が初出（採番）された際の要点・説明の冒頭テキスト。2回目以降の解析時にLLMへの「お手本（In-Context Learning）」として読み込まれ、同種の概念が別名で重複登録されるのを防ぎます。
* **`first_appeared_in`**: この枝番が初めて定義された講義単元（PART）名。トレーサビリティの確保に寄与します。

---

## 🔗 横の繋がり：重み付き前提知識 (`prerequisite_concepts`)

単なる文字列のリストではなく、GNN-KTでの伝播計算やGraph RAGでの「遡り学習」の精度を高めるため、前提知識を構造化オブジェクトとして保持します。

```json
"prerequisite_concepts": [
  {
    "concept_name": "展開の公式",
    "dependency_type": "mandatory",    // "mandatory" (必須) or "supplementary" (補足)
    "reasoning": "因数分解の逆算プロセスを理解するために必須となるため"
  }
]
```

* **`mandatory` (必須前提)**: 理解できていないと解法プロセスが破綻する決定的な前提知識（スキルツリー上で太い実線描画）。
* **`supplementary` (補足前提)**: 知っていると計算効率や理解がスムーズになる補足的知識（スキルツリー上で点線描画）。

---

## 🚀 この「階層化・構造化」がもたらす FT / Learn 機能への絶大な効果

この縦と横の厳格なネットワーク構造を持たせてデータを生成することで、FT（教員向け）およびLearn（生徒向け）の次世代機能が真の力を発揮します。

### 1. FT: 「習熟度の可視化」での Semantic Zoom ＆ GNN-KT 推論
* **ズームアウト（マクロ表示）**: 単元レベル（Level 2）でクラス全体の平均理解度をヒートマップ表示。
* **ズームイン（ミクロ表示）**: ドリルダウンすると親概念（Level 3）、さらに学習アクション（Level 4）へと展開。「二次関数が苦手なのではなく、『分数が含まれる平方完成（Level 4）』でクラス全員が躓いている」といった根本原因を特定できます。
* **GNN-KT（理解度推論）**: ネットワークの完全結合性を保証することで、正答ログから「この問題で躓いたということは、前提となるあの概念も理解できていない可能性が高い」と真の理解度を高精度に推論します。

### 2. Learn: 「適応型学習ナビ」での正確な遡り学習
* 生徒が Level 4（分数を含む平方完成）の問題で間違えた際、AIチューターは `dependency_type`（依存度）と `reasoning`（理由）を辿り、「まずは『分数式の計算』の動画（他単元）に戻って復習しよう！」といった、優先順位の明確な根拠あるレコメンド（アダプティブ・ラーニング）を実現します。

### 3. Learn: 「スキルツリー」のビジュアル生成
* Level 3（親概念ハブ）を中心に、Level 4（学習アクション）がノードとして接続されるスキルツリーを動的描画。
* 必須前提（`mandatory`）は太い実線、補足前提（`supplementary`）は点線としてビジュアル表現し、生徒が自分の成長過程や次に進むべきパスを一目で実感できるようになります。

### 4. マルチモーダル画像RAGへのシームレスな統合 (新機能)
* 画像から抽出された情報を単なるテキストとして扱うのではなく、Gemini Visionに `inferred_parent_concept` (Level 3) や `missing_prerequisite` (前提知識) として推論・翻訳させることで、画像データが直接オントロジーネットワークの検索クエリとして機能します。これにより、画像検索から「高精度な動画紐付け」と「遡り学習のサジェスト」が同時に実現します。

### 5. シラバスマッチングと宿題自動生成への応用 (新機能)
* 構築された `parent_concept` (Level 3) のリストは、学校独自のシラバス（年間授業計画）をシステムにマッピングする際の「辞書」として機能します。シラバスのテキストから該当する親概念を引き当て、その配下にある Level 4 の学習アクションから基礎・応用問題をバランスよく抽出することで、「今週の宿題自動生成」が可能になります。

---

## 📁 全体ディレクトリ構成

システムはセキュリティ・保守性・デプロイの容易性を確保するため、**バックエンド（解析・DB構築）**と**フロントエンド（UIアプリ）**に分離した設計となっています。

```text
.
├── 📄 README.md                             # プロジェクト全体概要・インデックス
├── 📁 docs/                                # 詳細ドキュメント群 (本フォルダー)
│   ├── 📄 01_system_architecture.md         # ★本ドキュメント (アーキテクチャ・構造仕様)
│   ├── 📄 02_search_engine.md              # 検索アルゴリズム仕様
│   ├── 📄 03_pipeline_and_setup.md          # パイプライン・詳細セットアップ
│   ├── 📄 04_obsidian_guide.md              # Obsidian可視化ガイド
│   ├── 📄 05_frontend_applications.md       # フロントエンドUI詳細仕様書
│   ├── 📄 06_image_rag_prompt_design.md     # マルチモーダル画像RAG 専用プロンプト仕様書
│   ├── 📄 07_syllabus_mapping_prompt.md     # シラバスマッチング 専用プロンプト仕様書
│   ├── 📄 RAG化をベースとした次世代機能構想(FT).md
│   ├── 📄 RAG化をベースとした次世代機能構想(Learn).md
│   └── 📄 RAG化をベースとした次世代機能構想(FT&Learn同時).md
│
├── 🏭 BL_sugaku_I_Backend/                  # バックエンド専用ディレクトリ
│   ├── 📄 .env                             # APIキー設定ファイル (GEMINI_API_KEYS: 複数キー自動ローテーション対応)
│   ├── 📄 mext_code_math_high.xlsx         # 指導要領コード表 (Excel)
│   ├── 📄 mext_math_high.md                # 指導要領解説データ (Markdown)
│   ├── 📄 BL_sugaku_I_index_clean.md       # 講座全体の目次データ (Markdown)
│   ├── 📄 mext_master_dict.json            # 【自動生成】統合指導要領マスター辞書
│   ├── 📄 concept_branch_master.json       # 【自動生成】全講共通 独自概念枝番マスター
│   ├── 📄 lecture_index_master.json        # 【自動生成】全講共通 目次マスター
│   ├── 📜 build_mext_master_dict.py        # 指導要領マスター辞書生成スクリプト
│   ├── 📜 build_index_master.py            # 目次マスター生成スクリプト
│   ├── 📜 run_single_part_batch.py         # 全PART一括解析バッチ (リトライ・通知付)
│   ├── 📜 build_vector_db.py               # グローバルベクトルDB構築スクリプト (Ver 12.0 GNN-KT対応)
│   ├── 📜 export_global_obsidian_vault_mext.py # Obsidian用ZIPパッケージ出力スクリプト (Ver 12.0)
│   ├── 📜 generate_mock_logs.py            # GNN-KT検証用 ダミー学習ログ生成スクリプト
│   ├── 📜 visualize_syllabus_content.py    # シラバス ➔ コンテンツ可視化テストスクリプト (対話型CLI)
│   ├── 📄 global_vector_db_cache.json      # 【成果物】全PART統合ベクトルキャッシュDB
│   ├── 📄 mock_student_logs.json           # 【成果物】生成されたダミー学習ログ
│   ├── 📦 Global_Obsidian_Vault_MEXT.zip   # 【成果物】Obsidian Vault パッケージ
│   │
│   └── 📁 BL_sugaku_Ⅰ_XX-Y/                # PART単位の解析フォルダー
│       ├── 📄 BL_sugaku_I_XX-Y.pdf         # 教材PDFファイル
│       ├── 📄 BL_sugaku_I_XX-Y_clean.md   # 【生成】クリーンテキスト (図の説明付)
│       ├── 📄 *.mp4 / *.vtt               # 講義動画および字幕ファイル
│       ├── 📜 phase0_pdf_to_md.py         # Phase 0: PDF ➔ Markdown変換 (自動リトライ・再アップロード機構付)
│       ├── 📜 phase1_text_analysis_ontology.py # Phase 1: 階層オントロジー解析 (Ver 12.1 自動修復パース対応)
│       ├── 📜 phase2_video_analysis.py    # Phase 2: 動画マルチモーダル解析 (403検知・再アップロード機構付)
│       ├── 📜 phase3_alignment_graph.py   # Phase 3: 三位一体データ結合 (ファジーマッチ・自動修復パース対応)
│       └── 📁 output_result/              # 中間出力結果格納フォルダー
│
└── 📱 BL_sugaku_I_Frontend/                 # フロントエンド専用ディレクトリ
    ├── 📄 .env                             # APIキー設定ファイル (フロントエンド用)
    ├── 📜 app.py                           # AIチューター UI アプリ (Streamlit)
    ├── 📜 app_ft.py                        # 教員向け Semantic Zoom ダッシュボード
    ├── 📄 global_vector_db_cache.json      # 【バックエンドからコピー】検索用DB
    └── 📄 requirements.txt                 # フロントエンド依存パッケージ定義
```