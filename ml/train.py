"""Entry point for training the AP poll model."""

from ml.train_model import load_rows, main, metrics, train

__all__ = ["load_rows", "main", "metrics", "train"]


if __name__ == "__main__":
    main()
