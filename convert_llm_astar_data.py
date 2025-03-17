import os
import json
import glob

def convert_llm_astar_to_constrained(experiment_dir):
    """
    Convert LLM A* data to match the format expected by performance.py
    
    Args:
        experiment_dir: Directory containing LLM A* experiment data
    """
    # Find all scenario data files
    scenario_data_files = glob.glob(os.path.join(experiment_dir, "scenario_*/data/scenario_data.json"))
    metrics_files = glob.glob(os.path.join(experiment_dir, "scenario_*/metrics/metrics.json"))
    
    print(f"Found {len(scenario_data_files)} scenario data files and {len(metrics_files)} metrics files")
    
    # Convert scenario data files
    for file_path in scenario_data_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is LLM A* data
            if "llm_astar" in data:
                # Convert llm_astar to constrained
                data["constrained"] = data.pop("llm_astar")
                
                # Ensure annotations are present with all fields
                if "annotations" not in data:
                    data["annotations"] = {
                        "preference": [],
                        "avoidance": [],
                        "proximity": []
                    }
                elif "proximity" not in data["annotations"]:
                    data["annotations"]["proximity"] = []
                
                # Save the updated data
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Converted {file_path}")
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
    
    # Convert metrics files
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is LLM A* data
            if "llm_astar" in data:
                # Convert llm_astar to constrained
                data["constrained"] = data.pop("llm_astar")
                
                # Save the updated data
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Converted {file_path}")
        except Exception as e:
            print(f"Error converting {file_path}: {e}")

if __name__ == "__main__":
    # Convert the most recent experiment
    experiment_dirs = sorted(glob.glob("ds_logs/llm_astar/experiment_*"))
    if experiment_dirs:
        most_recent = experiment_dirs[-1]
        print(f"Converting {most_recent}")
        convert_llm_astar_to_constrained(most_recent)
    else:
        print("No LLM A* experiments found") 