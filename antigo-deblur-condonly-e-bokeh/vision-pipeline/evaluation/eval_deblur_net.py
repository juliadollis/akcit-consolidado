import argparse

try:
    from .src.deblur_evaluator import CloudDeblurEvaluator, setup_logging
except ImportError:
    from src.deblur_evaluator import CloudDeblurEvaluator, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Image Quality Evaluation Pipeline with HF Datasets"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        nargs="+",
        required=True,
        help="One or more Hugging Face dataset names",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being evaluated (e.g., 'DeblurNet-V1')",
    )
    parser.add_argument(
        "--hf_split", type=str, default="test", help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run metrics"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed debug logs"
    )
    parser.add_argument(
        "--output_hf_repo",
        type=str,
        required=True,
        help="Destination HF repo ID to save results (e.g., 'username/metrics')",
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="Number of parallel metric threads"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for private/gated datasets and pushing",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Instantiate our new class interface
    evaluator_app = CloudDeblurEvaluator(
        device=args.device, threads=args.threads, verbose=args.verbose
    )

    # Run the pipeline
    evaluator_app.run_pipeline(
        hf_datasets=args.hf_dataset,
        output_repo_id=args.output_hf_repo,
        model_name=args.model_name,
        hf_split=args.hf_split,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
