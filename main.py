from data_structure.media_pipehands import MediaPipeHandsExplorer
import sys


def main():
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    # Create and run explorer
    explorer = MediaPipeHandsExplorer(config_path)
    explorer.run()


if __name__ == "__main__":
    main()