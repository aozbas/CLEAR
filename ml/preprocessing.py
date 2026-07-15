from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_UFES_AUGMENTATION_PROFILES = ("baseline", "regularized_v2")


def get_transforms(split: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_pad_ufes_transforms(
    split: str,
    *,
    augmentation_profile: str = "baseline",
) -> transforms.Compose:
    if augmentation_profile not in PAD_UFES_AUGMENTATION_PROFILES:
        raise ValueError(f"Unknown PAD-UFES augmentation profile: {augmentation_profile!r}")
    if split != "train" or augmentation_profile == "baseline":
        return get_transforms("train" if split == "train" else "val")

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                    )
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.1),
                ratio=(0.5, 2.0),
                value="random",
            ),
            normalize,
        ]
    )
