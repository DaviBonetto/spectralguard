"""
Push SpectralGuard files to HuggingFace Hub.
Usage: python scripts/push_to_hf.py --token <your_hf_token>
"""
import argparse
import os
import sys
import tempfile
import shutil
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="Push SpectralGuard to HuggingFace Hub")
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    parser.add_argument("--skip-space", action="store_true", help="Skip Space push")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip Dataset push")
    args = parser.parse_args()

    from huggingface_hub import HfApi, login
    login(token=args.token)
    api = HfApi()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ---------------------------------------------------------------
    # 1. Push Space: DaviBonetto/spectralguard-demo
    # ---------------------------------------------------------------
    if not args.skip_space:
        space_id = "DaviBonetto/spectralguard-demo"
        print(f"\n{'='*60}")
        print(f"  Pushing Space: {space_id}")
        print(f"{'='*60}")

        try:
            api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        except Exception as e:
            print(f"  [info] Repo creation: {e}")

        # Upload Space README
        space_readme = os.path.join(project_root, "docs", "SPACE_README.md")
        api.upload_file(
            path_or_fileobj=space_readme,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
        )
        print("  ✅ README.md uploaded")

        # Upload app.py
        app_path = os.path.join(project_root, "app.py")
        api.upload_file(
            path_or_fileobj=app_path,
            path_in_repo="app.py",
            repo_id=space_id,
            repo_type="space",
        )
        print("  ✅ app.py uploaded")

        # Create and upload requirements.txt for Space
        space_requirements = (
            "gradio>=4.0.0\n"
            "numpy>=1.26.0\n"
            "matplotlib>=3.8.0\n"
            "scikit-learn>=1.4.0\n"
            "joblib>=1.3.0\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(space_requirements)
            req_path = f.name
        api.upload_file(
            path_or_fileobj=req_path,
            path_in_repo="requirements.txt",
            repo_id=space_id,
            repo_type="space",
        )
        os.unlink(req_path)
        print("  ✅ requirements.txt uploaded")

        # Upload spectralguard package if it exists
        sg_dir = os.path.join(project_root, "spectralguard")
        if os.path.isdir(sg_dir):
            api.upload_folder(
                folder_path=sg_dir,
                path_in_repo="spectralguard",
                repo_id=space_id,
                repo_type="space",
            )
            print("  ✅ spectralguard/ package uploaded")

        # Upload setup.py
        setup_path = os.path.join(project_root, "setup.py")
        if os.path.exists(setup_path):
            api.upload_file(
                path_or_fileobj=setup_path,
                path_in_repo="setup.py",
                repo_id=space_id,
                repo_type="space",
            )
            print("  ✅ setup.py uploaded")

        print(f"\n  🌐 Space URL: https://huggingface.co/spaces/{space_id}")

    # ---------------------------------------------------------------
    # 2. Push Dataset: DaviBonetto/spectralguard-dataset
    # ---------------------------------------------------------------
    if not args.skip_dataset:
        dataset_id = "DaviBonetto/spectralguard-dataset"
        print(f"\n{'='*60}")
        print(f"  Pushing Dataset: {dataset_id}")
        print(f"{'='*60}")

        try:
            api.create_repo(repo_id=dataset_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"  [info] Repo creation: {e}")

        # Upload Dataset README
        ds_readme = os.path.join(project_root, "docs", "DATASET_README.md")
        api.upload_file(
            path_or_fileobj=ds_readme,
            path_in_repo="README.md",
            repo_id=dataset_id,
            repo_type="dataset",
        )
        print("  ✅ README.md uploaded")

        # Upload benchmark CSV
        csv_path = os.path.join(project_root, "data", "dataset", "spectralguard_benchmark.csv")
        if os.path.exists(csv_path):
            api.upload_file(
                path_or_fileobj=csv_path,
                path_in_repo="spectralguard_benchmark.csv",
                repo_id=dataset_id,
                repo_type="dataset",
            )
            print("  ✅ spectralguard_benchmark.csv uploaded")

        # Upload summary CSV
        summary_path = os.path.join(project_root, "data", "dataset", "dataset_summary.csv")
        if os.path.exists(summary_path):
            api.upload_file(
                path_or_fileobj=summary_path,
                path_in_repo="dataset_summary.csv",
                repo_id=dataset_id,
                repo_type="dataset",
            )
            print("  ✅ dataset_summary.csv uploaded")

        print(f"\n  🌐 Dataset URL: https://huggingface.co/datasets/{dataset_id}")

    print(f"\n{'='*60}")
    print("  Done! All files pushed to HuggingFace Hub.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
