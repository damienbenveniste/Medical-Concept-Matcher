"""Evaluation utilities for medical concept matching.

This module provides functions to evaluate the performance of the medical
concept matcher against test datasets.
"""
import asyncio
import aiohttp
import csv
import os
from pydantic import BaseModel
from typing import Optional, Literal
from data import Data


class CodeResult(BaseModel):
    """Result model for a matched medical code.
    
    Attributes:
        code: The medical code identifier.
        text: The human-readable description of the code.
        confidence: Confidence score (0-100) for the match.
        concept_type: The type of medical concept.
    """
    code: str
    text: str
    confidence: int
    concept_type: Optional[Literal['diagnose', 'procedure', 'measurement', 'drug']] = None


class ConceptResponse(BaseModel):
    """Response model for concept matching API.
    
    Attributes:
        concept: List of matched codes for the concept.
    """
    concept: list[CodeResult]


class EvaluationResult(BaseModel):
    """Result model for evaluating a single test item.
    
    Attributes:
        test_data: The original test data item.
        predictions: List of predicted codes from the matcher.
        correct_match: Whether the correct code was found in predictions.
    """
    test_data: Data
    predictions: list[CodeResult]
    correct_match: bool


async def match_concept(concept: str, base_url: str = "http://127.0.0.1:8000") -> ConceptResponse:
    """Match a medical concept using the API.
    
    Args:
        concept: The medical concept text to match.
        base_url: Base URL of the API server.
        
    Returns:
        ConceptResponse containing matched codes.
        
    Raises:
        Exception: If the API call fails.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/concept/match",
            json={"concept": concept}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return ConceptResponse(**data)
            else:
                error_text = await response.text()
                raise Exception(f"API call failed with status {response.status}: {error_text}")


def load_test_data(dataset_name: str) -> list[Data]:
    """Load test data from CSV file.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'icd', 'atc').
        
    Returns:
        List of Data objects loaded from the CSV file.
    """
    test_data_root = os.path.join(os.path.dirname(__file__), '..', 'test_data')
    file_path = os.path.join(test_data_root, f'{dataset_name}_test.csv')
    
    test_data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            test_data.append(Data(code=row['code'], text=row['text']))
    
    return test_data


async def evaluate_single_item(test_item: Data, base_url: str = "http://127.0.0.1:8000") -> EvaluationResult:
    """Evaluate a single test item against the matcher.
    
    Args:
        test_item: The test data item to evaluate.
        base_url: Base URL of the API server.
        
    Returns:
        EvaluationResult containing predictions and match status.
    """
    try:
        predictions = await match_concept(test_item.text, base_url)
        
        # Check if the correct code is in the predictions
        correct_match = any(pred.code == test_item.code for pred in predictions.concept)
        
        return EvaluationResult(
            test_data=test_item,
            predictions=predictions.concept,
            correct_match=correct_match
        )
    except Exception as e:
        print(f"Error evaluating {test_item.code}: {e}")
        return EvaluationResult(
            test_data=test_item,
            predictions=[],
            correct_match=False
        )


async def evaluate_dataset(dataset_name: str, base_url: str = "http://127.0.0.1:8000") -> list[EvaluationResult]:
    """Evaluate entire dataset with parallel processing.
    
    Args:
        dataset_name: Name of the dataset to evaluate.
        base_url: Base URL of the API server.
        
    Returns:
        List of EvaluationResult objects for all test items.
    """
    print(f"Loading test data for {dataset_name}...")
    test_data = load_test_data(dataset_name)
    
    print(f"Evaluating {len(test_data)} items...")
    
    # Create tasks for parallel processing
    tasks = [
        asyncio.create_task(evaluate_single_item(item, base_url))
        for item in test_data
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return valid results
    valid_results = [r for r in results if isinstance(r, EvaluationResult)]
    
    return valid_results


def calculate_metrics(results: list[EvaluationResult]) -> dict:
    """Calculate evaluation metrics including precision, recall, and F1.
    
    Args:
        results: List of evaluation results.
        
    Returns:
        Dictionary containing calculated metrics.
    """
    total = len(results)
    
    # True positives: correct matches found
    true_positives = sum(1 for r in results if r.correct_match)
    
    # False positives: incorrect predictions made
    false_positives = sum(len(r.predictions) - (1 if r.correct_match else 0) for r in results)
    
    # False negatives: correct answers not found
    false_negatives = sum(1 for r in results if not r.correct_match)
    
    # True negatives: not applicable in this context (would be infinite)
    
    # Calculate metrics
    accuracy = true_positives / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total': total,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_metrics_to_file(all_metrics: dict, filename: str = "evaluation_results.txt") -> None:
    """Save evaluation metrics to a text file.
    
    Args:
        all_metrics: Dictionary containing metrics for all datasets.
        filename: Name of the output file.
    """
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("=== MEDICAL CONCEPT MATCHER EVALUATION RESULTS ===\n\n")
        
        for dataset, metrics in all_metrics.items():
            f.write(f"Dataset: {dataset.upper()}\n")
            f.write(f"Total items: {metrics['total']}\n")
            f.write(f"True Positives: {metrics['true_positives']}\n")
            f.write(f"False Positives: {metrics['false_positives']}\n")
            f.write(f"False Negatives: {metrics['false_negatives']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})\n")
            f.write(f"Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})\n")
            f.write(f"Recall: {metrics['recall']:.4f} ({metrics['recall']:.2%})\n")
            f.write(f"F1 Score: {metrics['f1']:.4f} ({metrics['f1']:.2%})\n")
            f.write("\n" + "="*50 + "\n\n")
        
        # Overall summary using existing metrics
        total_items = sum(m['total'] for m in all_metrics.values())
        total_tp = sum(m['true_positives'] for m in all_metrics.values())
        total_fp = sum(m['false_positives'] for m in all_metrics.values())
        total_fn = sum(m['false_negatives'] for m in all_metrics.values())
        
        overall_accuracy = total_tp / total_items if total_items > 0 else 0
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        f.write("OVERALL SUMMARY\n")
        f.write(f"Total items: {total_items}\n")
        f.write(f"True Positives: {total_tp}\n")
        f.write(f"False Positives: {total_fp}\n")
        f.write(f"False Negatives: {total_fn}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.2%})\n")
        f.write(f"Overall Precision: {overall_precision:.4f} ({overall_precision:.2%})\n")
        f.write(f"Overall Recall: {overall_recall:.4f} ({overall_recall:.2%})\n")
        f.write(f"Overall F1 Score: {overall_f1:.4f} ({overall_f1:.2%})\n")
    
    print(f"\nResults saved to: {filepath}")


async def main() -> None:
    """Main evaluation function.
    
    Evaluates all datasets and saves results to file.
    """
    datasets = ['icd', 'atc', 'loinc', 'cpt']
    all_metrics = {}
    
    for dataset in datasets:
        print(f"\n=== Evaluating {dataset.upper()} dataset ===")
        
        try:
            results = await evaluate_dataset(dataset)
            metrics = calculate_metrics(results)
            all_metrics[dataset] = metrics
            
            print(f"Results for {dataset}:")
            print(f"  Total: {metrics['total']}")
            print(f"  True Positives: {metrics['true_positives']}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall: {metrics['recall']:.2%}")
            print(f"  F1 Score: {metrics['f1']:.2%}")
            
        except Exception as e:
            print(f"Error evaluating {dataset}: {e}")
            all_metrics[dataset] = {
                'total': 0, 'true_positives': 0, 'false_positives': 0, 
                'false_negatives': 0, 'accuracy': 0, 'precision': 0, 
                'recall': 0, 'f1': 0
            }
    
    # Save results to file
    save_metrics_to_file(all_metrics)


if __name__ == "__main__":
    asyncio.run(main())