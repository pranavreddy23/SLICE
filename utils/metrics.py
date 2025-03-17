import numpy as np
import time
from typing import List, Tuple, Dict, Any

class PlannerMetrics:
    """Metrics for evaluating planner performance"""
    
    @staticmethod
    def calculate_path_length(path: List[Tuple[int, int]]) -> float:
        """Calculate the length of a path"""
        if not path or len(path) < 2:
            return 0.0
            
        length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
        return length
    
    @staticmethod
    def calculate_path_smoothness(path: List[Tuple[int, int]]) -> float:
        """Calculate path smoothness (lower is smoother)"""
        if not path or len(path) < 3:
            return 0.0
            
        # Calculate angles between consecutive segments
        angles = []
        for i in range(len(path) - 2):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            x3, y3 = path[i+2]
            
            # Vectors for segments
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)
            
            # Calculate angle
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag_v1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag_v2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
            
            if mag_v1 * mag_v2 == 0:
                continue
                
            cos_angle = dot_product / (mag_v1 * mag_v2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Ensure within valid range
            angle = np.arccos(cos_angle)
            angles.append(angle)
            
        if not angles:
            return 0.0
            
        # Average angle change (in radians)
        return np.mean(angles)
    
    @staticmethod
    def calculate_constraint_compliance(path, environment, constraints):
        """Calculate how well a path complies with constraints"""
        compliance = {}
        
        # Check proximity constraints
        if 'proximity' in constraints:
            proximity_scores = []
            for region_name, weight in constraints['proximity'].items():
                if region_name in environment.regions:
                    # Calculate average distance to region
                    distances = []
                    for x, y in path:
                        dist = environment.distance_to_region(x, y, region_name)
                        distances.append(dist)
                    
                    avg_dist = sum(distances) / len(distances) if distances else float('inf')
                    # Lower distance is better for proximity
                    score = 1.0 / (1.0 + avg_dist)  # Normalize to 0-1 range
                    proximity_scores.append(score)
            
            compliance["proximity"] = sum(proximity_scores) / len(proximity_scores) if proximity_scores else 1.0
        
        # Check avoidance constraints
        if 'avoidance' in constraints:
            avoidance_scores = []
            for region_name, weight in constraints['avoidance'].items():
                if region_name in environment.regions:
                    # Count how many points are in the region
                    in_region_count = sum(1 for x, y in path if environment.is_in_region(x, y, region_name))
                    # Lower count is better for avoidance
                    score = 1.0 - (in_region_count / len(path)) if path else 1.0
                    avoidance_scores.append(score)
            
            compliance["avoidance"] = sum(avoidance_scores) / len(avoidance_scores) if avoidance_scores else 1.0
        
        # Check preference constraints
        if 'preference' in constraints:
            preference_scores = []
            for region_name, weight in constraints['preference'].items():
                if region_name in environment.regions:
                    # Count how many points are in the region
                    in_region_count = sum(1 for x, y in path if environment.is_in_region(x, y, region_name))
                    # Higher count is better for preference
                    score = in_region_count / len(path) if path else 0.0
                    preference_scores.append(score)
            
            compliance["preference"] = sum(preference_scores) / len(preference_scores) if preference_scores else 1.0
        
        # Calculate overall compliance
        scores = []
        if 'proximity' in compliance:
            scores.append(compliance["proximity"])
        if 'avoidance' in compliance:
            scores.append(compliance["avoidance"])
        if 'preference' in compliance:
            scores.append(compliance["preference"])
        
        compliance["overall"] = sum(scores) / len(scores) if scores else 1.0
        
        return compliance
    
    @staticmethod
    def calculate_search_efficiency(nodes_expanded: int, path_length: float) -> float:
        """Calculate search efficiency (path length / nodes expanded)"""
        if nodes_expanded == 0:
            return 0.0
        return path_length / nodes_expanded
    
    @staticmethod
    def evaluate_planner(environment, start, goal, planner_func, constraints=None, baseline_func=None):
        """Evaluate a planner against a baseline"""
        metrics = {}
        
        # Evaluate constrained planner
        start_time = time.time()
        constrained_path, constrained_stats = planner_func(environment, start, goal, constraints)
        constrained_time = time.time() - start_time
        
        metrics["constrained"] = {
            "success": constrained_path is not None,
            "time": constrained_time,
            "path_length": PlannerMetrics.calculate_path_length(constrained_path) if constrained_path else 0,
            "path_smoothness": PlannerMetrics.calculate_path_smoothness(constrained_path) if constrained_path else 0,
            "nodes_expanded": constrained_stats.get("nodes_expanded", 0),
            "search_efficiency": PlannerMetrics.calculate_search_efficiency(
                constrained_stats.get("nodes_expanded", 0),
                PlannerMetrics.calculate_path_length(constrained_path) if constrained_path else 0
            )
        }
        
        # If constraints provided, evaluate compliance
        if constraints and constrained_path:
            metrics["constrained"]["compliance"] = PlannerMetrics.calculate_constraint_compliance(
                constrained_path, environment, constraints
            )
        
        # If baseline function provided, evaluate it
        if baseline_func:
            start_time = time.time()
            baseline_path, baseline_stats = baseline_func(environment, start, goal)
            baseline_time = time.time() - start_time
            
            metrics["baseline"] = {
                "success": baseline_path is not None,
                "time": baseline_time,
                "path_length": PlannerMetrics.calculate_path_length(baseline_path) if baseline_path else 0,
                "path_smoothness": PlannerMetrics.calculate_path_smoothness(baseline_path) if baseline_path else 0,
                "nodes_expanded": baseline_stats.get("nodes_expanded", 0),
                "search_efficiency": PlannerMetrics.calculate_search_efficiency(
                    baseline_stats.get("nodes_expanded", 0),
                    PlannerMetrics.calculate_path_length(baseline_path) if baseline_path else 0
                )
            }
            
            # If constraints provided, evaluate baseline compliance for comparison
            if constraints and baseline_path:
                metrics["baseline"]["compliance"] = PlannerMetrics.calculate_constraint_compliance(
                    baseline_path, environment, constraints
                )
            
            # Calculate improvement metrics
            if constrained_path and baseline_path:
                metrics["improvement"] = {
                    "time_ratio": baseline_time / constrained_time if constrained_time > 0 else float('inf'),
                    "path_length_ratio": (metrics["baseline"]["path_length"] / 
                                         metrics["constrained"]["path_length"] 
                                         if metrics["constrained"]["path_length"] > 0 else float('inf')),
                    "nodes_expanded_ratio": (metrics["baseline"]["nodes_expanded"] / 
                                           metrics["constrained"]["nodes_expanded"] 
                                           if metrics["constrained"]["nodes_expanded"] > 0 else float('inf')),
                    "search_efficiency_ratio": (metrics["constrained"]["search_efficiency"] / 
                                              metrics["baseline"]["search_efficiency"] 
                                              if metrics["baseline"]["search_efficiency"] > 0 else float('inf'))
                }
                
                if "compliance" in metrics["constrained"] and "compliance" in metrics["baseline"]:
                    metrics["improvement"]["compliance_improvement"] = (
                        metrics["constrained"]["compliance"]["overall"] - 
                        metrics["baseline"]["compliance"]["overall"]
                    )
        
        return metrics