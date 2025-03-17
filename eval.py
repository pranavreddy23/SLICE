from typing import List, Tuple, Dict, Any
import numpy as np

def evaluate_path(path: List[Tuple[int, int]], environment, constraints: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate a path based on constraint satisfaction and other metrics"""
    if not path:
        return {
            "success": 0.0,
            "constraint_satisfaction": 0.0,
            "path_length": float('inf'),
            "smoothness": 0.0
        }
    
    metrics = {}
    
    # Path success
    metrics["success"] = 1.0
    
    # Path length
    metrics["path_length"] = len(path) - 1
    
    # Path smoothness (fewer direction changes is smoother)
    direction_changes = 0
    for i in range(1, len(path) - 1):
        prev_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        next_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if prev_dir != next_dir:
            direction_changes += 1
    
    smoothness = 1.0 - (direction_changes / (len(path) - 2)) if len(path) > 2 else 1.0
    metrics["smoothness"] = smoothness
    
    # Constraint satisfaction
    constraint_scores = []
    
    # Avoidance constraints
    if 'avoidance' in constraints:
        for region, weight in constraints['avoidance'].items():
            if region in environment.regions:
                # Count cells in path that are in the avoidance region
                avoid_cells = sum(1 for x, y in path if environment.is_in_region(x, y, region))
                avoid_score = 1.0 - (avoid_cells / len(path))
                constraint_scores.append((avoid_score, weight))
    
    # Proximity constraints
    if 'proximity' in constraints:
        for region, weight in constraints['proximity'].items():
            if region in environment.regions:
                # Average distance to region
                distances = [environment.distance_to_region(x, y, region) for x, y in path]
                max_possible_dist = environment.width + environment.height
                avg_dist = sum(distances) / len(distances)
                proximity_score = 1.0 - (avg_dist / max_possible_dist)
                constraint_scores.append((proximity_score, weight))
    
    # Preference constraints
    if 'preference' in constraints:
        for region, weight in constraints['preference'].items():
            if region in environment.regions:
                # Count cells in path that are in the preference region
                pref_cells = sum(1 for x, y in path if environment.is_in_region(x, y, region))
                pref_score = pref_cells / len(path)
                constraint_scores.append((pref_score, weight))
    
    # Calculate weighted average of constraint scores
    if constraint_scores:
        total_weight = sum(weight for _, weight in constraint_scores)
        weighted_score = sum(score * weight for score, weight in constraint_scores) / total_weight
        metrics["constraint_satisfaction"] = weighted_score
    else:
        metrics["constraint_satisfaction"] = 1.0  # No constraints to satisfy
    
    return metrics


def compare_paths(constraint_path, baseline_path, environment, constraints):
    """Compare constraint-based path with baseline path"""
    constraint_metrics = evaluate_path(constraint_path, environment, constraints)
    baseline_metrics = evaluate_path(baseline_path, environment, constraints)
    
    comparison = {
        "constraint_path": constraint_metrics,
        "baseline_path": baseline_metrics,
        "improvement": {}
    }
    
    # Calculate improvements
    for metric in constraint_metrics:
        if metric in baseline_metrics:
            if metric == "path_length":
                # For path length, lower is better
                if baseline_metrics[metric] > 0:
                    relative_change = (baseline_metrics[metric] - constraint_metrics[metric]) / baseline_metrics[metric]
                    comparison["improvement"][metric] = relative_change
            else:
                # For other metrics, higher is better
                if baseline_metrics[metric] > 0:
                    relative_change = (constraint_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]
                    comparison["improvement"][metric] = relative_change
    
    return comparison