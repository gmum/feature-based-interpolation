# Feature-Based Interpolation in the Latent Space of Pre-Trained Generative Models - official  implementation

## Example 1 - interpolation visualization in the space of measures between densities given as mixtures of Gaussian
```commandline
python interpolation_visualization.py
```

## Example 2 - interpolation in latent space of generative model
In order to run interpolation on example Mnist DCGAN with 2 dimensional latent (and favour class 8 on interpolation path) run:
```
python main.py --favored_class 8
```

You can also specify your generative model and classifier and swap `mnist_model.classifier_mnist` and `mnist_model.dcgan` for it.

## Example 3 - search for geodesics on manifolds

In script [geodesics_manifolds.ipynb](https://github.com/gmum/feature-based-interpolation/blob/main/geodesics_manifolds.ipynb), we present solutions to examples 1, 2, 3, and 4 from our paper.
