import imgaug as ia
from imgaug import augmenters as iaa


def get():
    def sometimes(aug): return iaa.Sometimes(1.0, aug)

    return iaa.Sequential(
        [

            sometimes(iaa.Affine(

                rotate=(-15, 15),  # rotate by -45 to +45 degrees

                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),

            iaa.SomeOf((0, 5),
                       [

                           iaa.OneOf([
                               iaa.GaussianBlur((1.0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),

                           
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images

            ],
                random_order=True
            )
        ],
        random_order=True
    )
