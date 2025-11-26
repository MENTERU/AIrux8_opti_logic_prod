# AIrux8_opti_logic
#05_project_富士通ゼネラル_家庭エアコン節電ナッジ

Repository for developing services, jobs, functions to deploy to production on the cloud (GCP)

クラウド（GCP）上の本番環境にデプロイするジョブ/関数を開発するためのリポジトリ

The project on GCP is `airux8-opti-logic` / GCP 上のプロジェクトは `airux8-opti-logic`

## 📁 Project Structure / プロジェクト構造

```
AIrux8_opti_logic/
├── services/                      # Cloud Run Services / クラウドランサービス
│   ├── ****/                   # Multiple service modules / 複数のサービスモジュール
│   │   ├── main.py              # Service entry point / サービスエントリーポイント
│   │   ├── Dockerfile           # Container configuration / コンテナ設定
│   │   ├── requirements.txt     # Dependencies / 依存関係
│   │   ├── README.md            # Service-specific documentation / サービス固有のドキュメント
│   │   └── ...                  # Additional modules and configs / 追加モジュールと設定
│   └── ...
├── jobs/                         # Cloud Run Jobs / クラウドランジョブ
│   ├── ****/                   # Multiple job modules / 複数のジョブモジュール
│   │   ├── main.py              # Job entry point / ジョブエントリーポイント
│   │   ├── Dockerfile           # Container configuration / コンテナ設定
│   │   ├── requirements.txt     # Dependencies / 依存関係
│   │   ├── README.md            # Job-specific documentation / ジョブ固有のドキュメント
│   │   └── ...                  # Additional modules and configs / 追加モジュールと設定
│   └── ...
├── functions/                    # Cloud Functions / クラウド関数
│   ├── ****/                   # Multiple function modules / 複数の関数モジュール
│   │   ├── main.py              # Function entry point / 関数エントリーポイント
│   │   ├── requirements.txt     # Dependencies / 依存関係
│   │   ├── README.md            # Function-specific documentation / 関数固有のドキュメント
│   │   └── ...                  # Additional modules and configs / 追加モジュールと設定
│   └── ...

└── README.md                     # This file / このファイル
```

## 🏗️ Development / 開発

- **Services / サービス**: Containerized services deployed to Google Cloud Run
  Google Cloud Run にデプロイされるコンテナ化されたサービス
- **Jobs / ジョブ**: Containerized jobs deployed to Google Cloud Run
  Google Cloud Run にデプロイされるコンテナ化されたジョブ
- **Functions / 関数**: Serverless functions deployed to Google Cloud Functions
  Google Cloud Functions にデプロイされるサーバーレス関数

## 🚀 Deployment / デプロイ

This repository uses GitHub Actions for automated deployment to Google Cloud Platform. The deployment workflow is configured in `.github/workflows/deploy.yml`.
このリポジトリは、Google Cloud Platform への自動デプロイに GitHub Actions を使用しています。デプロイワークフローは `.github/workflows/deploy.yml` で設定されています。

## 📚 Documentation / ドキュメント

Each service, job, and function contains its own detailed documentation:
各サービス、ジョブ、関数には独自の詳細なドキュメントが含まれています：

- **Services / サービス**: See individual `README.md` files in each `svc-*/` directory
  各 `svc-*/` ディレクトリ内の個別の `README.md` ファイルを参照
- **Jobs / ジョブ**: See individual `README.md` files in each `job-*/` directory
  各 `job-*/` ディレクトリ内の個別の `README.md` ファイルを参照
- **Functions / 関数**: See individual `README.md` files in each `func-*/` directory
  各 `func-*/` ディレクトリ内の個別の `README.md` ファイルを参照