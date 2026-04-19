from config import parse_args, setup_env
from trainer.run import run_experiment

def main():

    config = parse_args()

    setup_env(config)

    run_experiment(config)

    print("✅ Loaded successfully!")
    print(f"Device: {config.device}")


if __name__ == "__main__":
    main()