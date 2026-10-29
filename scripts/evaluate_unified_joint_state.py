"""Use state-conditioned config with the unchanged real-policy joint evaluator."""
from train_unified_joint_state import install


if __name__ == "__main__":
    install()
    from evaluate_unified_joint import main
    main()
