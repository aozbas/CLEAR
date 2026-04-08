"""Supervised training entry point. TODO: wire dataset + loop."""

from ml.models.classifier import build_model


def main():
    model = build_model()
    # TODO: dataloaders, optimizer, loss, epoch loop, checkpoint save
    print(model)


if __name__ == "__main__":
    main()
