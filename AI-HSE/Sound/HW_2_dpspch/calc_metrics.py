
# Usage:
# python calc_metrics.py predictions_dir=/path/to/predictions 
#        ground_truth_dir=/path/to/transcriptions


import argparse
from pathlib import Path
from typing import Dict, Tuple

from src.metrics.utils import calc_wer, calc_cer
from src.text_encoder import CTCTextEncoder


def load_text_files(directory: Path) -> Dict[str, str]:
    """
    Load all .txt files from directory, using filename as key
    
    Args:
        directory: Path to directory containing .txt files
        
    Returns:
        Dictionary mapping utterance_id -> text content
    """
    texts = {}
    if not directory.exists():
        return texts
    
    for txt_file in directory.glob("*.txt"):
        utterance_id = txt_file.stem
        with txt_file.open('r', encoding='utf-8') as f:
            texts[utterance_id] = f.read().strip()
    
    return texts


def calculate_metrics(
    predictions_dir: Path,
    ground_truth_dir: Path,
    text_encoder: CTCTextEncoder
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """
    Calculate WER and CER metrics on predictions vs ground truth.
    
    Args:
        predictions_dir: Directory containing prediction .txt files
        ground_truth_dir: Directory containing ground truth .txt files
        text_encoder: Text encoder for normalization
        
    Returns:
        Tuple of (wer, cer, per_utterance_metrics)
        per_utterance_metrics: dict mapping utterance_id -> {"wer": float, "cer": float}
    """

    # load
    predictions = load_text_files(predictions_dir)
    ground_truth = load_text_files(ground_truth_dir)
    
    # find pairs by ID
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    
    if not common_ids:
        raise ValueError(
            f"No matching utterance IDs found between predictions and ground truth.\n"
            f"Predictions: {list(predictions.keys())[:5]}...\n"
            f"Ground truth: {list(ground_truth.keys())[:5]}..."
        )
    
    print(f"Found {len(common_ids)} matching utterances")
    
    total_wer = 0.0
    total_cer = 0.0
    per_utterance = {}
    
    for utterance_id in sorted(common_ids):
        pred_text = predictions[utterance_id]
        gt_text = ground_truth[utterance_id]
        
        # norm texts
        pred_normalized = text_encoder.normalize_text(pred_text)
        gt_normalized = text_encoder.normalize_text(gt_text)
        
        # calc metrics
        wer = calc_wer(gt_normalized, pred_normalized)
        cer = calc_cer(gt_normalized, pred_normalized)
        
        total_wer += wer
        total_cer += cer
        per_utterance[utterance_id] = {"wer": wer, "cer": cer}
    
    # avg metrics
    avg_wer = total_wer / len(common_ids) if common_ids else 0.0
    avg_cer = total_cer / len(common_ids) if common_ids else 0.0
    
    return avg_wer, avg_cer, per_utterance


def main():
    parser = argparse.ArgumentParser(
        description="Calculate WER/CER metrics on predictions and ground truth transcriptions"
    )
    parser.add_argument(
        "predictions_dir",
        type=str,
        help="Directory containing prediction .txt files (filenames = utterance IDs)"
    )
    parser.add_argument(
        "ground_truth_dir",
        type=str,
        help="Directory containing ground truth .txt files (filenames = utterance IDs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: path to save detailed metrics JSON file"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample predictions to display (default: 5)"
    )
    
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    ground_truth_dir = Path(args.ground_truth_dir)
    
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")
    
    # init text encoder
    text_encoder = CTCTextEncoder()
    
    print("CALCULATING METRICS")
    print(f"Predictions: {predictions_dir}")
    print(f"Ground truth: {ground_truth_dir}")
    
    # calc metrics
    avg_wer, avg_cer, per_utterance = calculate_metrics(
        predictions_dir, ground_truth_dir, text_encoder
    )
    
    # print results
    print("RESULTS")
    print(f"WER:  {avg_wer:.4f}")
    print(f"CER: {avg_cer:.4f}")
    print(f"Number of utterances: {len(per_utterance)}")
    print()
    
    # show sample predictions
    if args.show_samples > 0:
        print(f"SAMPLE PREDICTIONS (showing {min(args.show_samples, len(per_utterance))})")
        
        sample_ids = list(per_utterance.keys())[:args.show_samples]
        predictions = load_text_files(predictions_dir)
        ground_truth = load_text_files(ground_truth_dir)
        
        for utterance_id in sample_ids:
            pred = predictions[utterance_id]
            gt = ground_truth[utterance_id]
            wer = per_utterance[utterance_id]["wer"]
            cer = per_utterance[utterance_id]["cer"]
            
            print(f"\nID: {utterance_id}")
            print(f"  WER: {wer:.4f}, CER: {cer:.4f}")
            print(f"  Ground truth: {gt}")
            print(f"  Prediction:   {pred}")
    
    # Save detailed metrics
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "average_wer": avg_wer,
            "average_cer": avg_cer,
            "num_utterances": len(per_utterance),
            "per_utterance": per_utterance
        }
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n Detailed metrics saved to: {output_path}")


if __name__ == "__main__":
    main()

