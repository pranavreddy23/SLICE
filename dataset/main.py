# main.py
import os
import argparse
from dataset_creator import DatasetCreator
from annotation_tool import AnnotationTool
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='SLICE Dataset Creator and Annotator')
    parser.add_argument('--action', type=str, choices=['create', 'annotate'], 
                        default='create', help='Action to perform')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', 
                        help='Directory to store dataset')
    parser.add_argument('--num_maps', type=int, default=5, 
                        help='Number of maps to generate')
    parser.add_argument('--pairs_per_map', type=int, default=2,
                        help='Number of start-goal pairs per map')
    args = parser.parse_args()
    
    # Create dataset directory if it doesn't exist
    os.makedirs(args.dataset_dir, exist_ok=True)
    
    if args.action == 'create':
        print("Creating dataset...")
        creator = DatasetCreator(output_dir=args.dataset_dir)
        dataset = creator.create_dataset(num_maps=args.num_maps, pairs_per_map=args.pairs_per_map)
        print("Dataset created successfully!")
        print(f"Created {len(dataset['maps'])} maps with {len(dataset['scenarios'])} scenarios")
        print("\nTo annotate the dataset, run:")
        print(f"python main.py --action annotate --dataset_dir {args.dataset_dir}")
        
    elif args.action == 'annotate':
        print("Starting annotation tool...")
        tool = AnnotationTool(dataset_path=f"{args.dataset_dir}/dataset_index.json")
        plt.show()
        print("Annotation complete!")

if __name__ == "__main__":
    main()