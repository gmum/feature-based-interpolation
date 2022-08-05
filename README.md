# Feature-Based Interpolation and Geodesics in the Latent Spaces of Generative Models

Interpolating between points is a problem connected simultaneously with finding geodesics and study of generative models. In the case of geodesics, we search for the curves with shortest length, while in the case of generative models we typically apply linear interpolation in the latent space.
However, this interpolation uses implicitly the fact that Gaussian is unimodal. Thus the problem of interpolating in the case when the latent density is non-Gaussian is an open problem.

In this paper we present a general and unified approach to interpolation, which simultaneously allows to search for geodesics and interpolating curves in latent space in the case of arbitrary density.
Our results have a strong theoretical background based on the introduced quality measure of an interpolating curve. In particular, we show that maximising the quality measure of the curve can be equivalently understood as a search of geodesic for a certain redefinition of the Riemannian metric on the space. 

We provide examples in three important cases. First, we show that our approach can be easily applied to finding geodesics on manifolds. Next, 
we focus our attention in finding interpolations in pre-trained generative models. We show that our model effectively works in the case of arbitrary density. Moreover, we can interpolate in the subset of the space consisting of data possessing a given feature. The last case is focused on finding interpolation in the space of chemical compounds.

---

&nbsp;
&nbsp;
&nbsp;
&nbsp;


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
