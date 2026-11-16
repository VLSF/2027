---
layout: distill
title: Hearing the shape of a drum with deep generative models
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text.
  Do not include math/latex or hyperlinks.
date: 2027-04-30
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2027-04-30-deep-drums.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Can one hear the shape of a drum?
  - name: Parametrizations, symmetries and architectures
  - name: "Warm-up: hearing triangular drums"
  - name: Isospectral drums of Buser, Conway, Doyle, Semmler
  - name: Closing remarks

---

<style>
  ul.tight-list li {
    margin-bottom: 5px;
    line-height: 1.5;
  }

  d-article h2 {
    font-weight: 400 !important;
    font-size: 1.9rem !important;
    line-height: 1.25 !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
  }

  .question-box {
      margin-left: 5.0rem;
      padding: 0.75rem 1.25rem;
      background-color: rgba(120, 140, 220, 0.12);
      border-left: none !important;
      border-radius: 6px;
      font-style: normal;

      /* Width & Height controls */
      width: 74%;
      max-width: 700px;
    }

  .question-box p:last-child {
      margin-bottom: 0 !important;
  }

</style>

## Introduction

Deep generative models for scientific inverse problems are on the rise. Many exciting new benchmarks are being published left and right <d-cite key="zheng2025inversebench,yao2026posteriorbench"></d-cite>. Dr. Katherine Bouman just gave a great ICLR26 talk on [how to take a picture of a black hole](https://iclr.cc/virtual/2026/invited-talk/10020868). Meanwhile, Midjourney put out [a speculative vision of a full-body ultrasound device](https://www.midjourney.com/medical/blogpost), prompting Ben Recht to write [a grumpy reply](https://www.argmin.net/p/all-data-is-good-data). ArenaPhysics released [a foundation model, Marconi-0,](https://www.arenaphysica.com/publications/rf-studio) for the synthesis of electromagnetic devices. And the authors of a recent [guest post on Anthropic's blog](https://www.anthropic.com/research/teaching-ai-to-build-scientific-models) declared simulation-based inference the future of scientific discovery in the age of AI agents.

With all this commotion, we thought it made sense to step back and offer a brief introduction to scientific inverse problems.

As a running example, we use a famous inverse problem from spectral geometry: reconstructing the shape of a domain from the spectrum of the Laplacian, or, as Lipman Bers put it, *hearing the shape of a drum*.

The problem is aesthetically appealing, features a rich geometric structure with challenging symmetries, and, most importantly, provably has no solution. Sure, that won't stop us---data scientists, deep learners, and AI researchers---from actually solving it anyway with deep generative models.

## Can one hear the shape of a drum?

In 1966, Mark Kac published a paper with that exact title <d-cite key="6d1fe8e1-9a0d-30f4-aeb4-baa6cceccb15"></d-cite>. Consider a membrane $\Gamma$ that is clamped at the boundary $\partial \Gamma$, which we call a drum. If we hit the drum, the vertical displacement $F(x, y, t)$ of its surface follows the standard wave equation

$$
\frac{\partial^2 F(x, y, t)}{\partial t^2} = c^2 \left(\frac{\partial^2 F(x, y, t)}{\partial x^2}+\frac{\partial^2 F(x, y, t)}{\partial y^2}\right), (x, y) \in \Gamma, \left.F(x, y, t)\right|_{(x, y) \in \partial\Gamma} = 0,
$$

where $c$ is the wave speed, depending on the material and the tension of the drum. This physical motion creates the drum's sound. A particularly convenient way to analyze all possible vibrations is to study *pure tones*, given by solutions of the form $F(x, y, t) = \phi_{\omega}(x, y) e^{i\omega t}$. For pure tones, the equation reduces to an eigenproblem:

$$
-\frac{\partial^2 \phi_i(x, y)}{\partial x^2} -\frac{\partial^2 \phi_i(x, y)}{\partial y^2} = \lambda_i \phi_i(x, y),(x, y) \in \Gamma, \left.\phi_i(x, y)\right|_{(x,y)\in\partial\Gamma} = 0.
$$

"Can one hear the shape of a drum?" is a poetic summary of a formal mathematical question:
> Can one find a boundary $\partial\Gamma$, given a complete spectrum $\lambda_1 \leq \lambda_2 \leq \dots$ of the Laplacian spectral problem?
{: .question-box}

Can we *actually* hear how the drum will sound if we compute the spectrum $\lambda_1 \leq \lambda_2 \leq \dots$ somehow? Yes, we can! Thanks to a neat demonstration by [Basel Ashraf](https://github.com/BaselAshraf81/eigendrum).
<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/side-by-side-drums.html' | relative_url }}"
    width="100%"
    height="410"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="Circle and triangle">
  </iframe>
</d-figure>
Of course, in the real world, the actual sound will depend on many other messy details, like the material and the tension of the drum's membrane, where and how one hits it, and the state of the ambient air. We will ignore all that, as mathematicians do, and strictly focus on the reconstruction of the boundary shape $\partial\Gamma$ from the spectrum $\lambda_1 \leq \lambda_2 \leq \dots$. This is our inverse problem.

While the inverse problem is notoriously difficult, the forward problem is fairly straightforward. We need it for certain methods (like Markov Chain Monte Carlo), for dataset collection, and for evaluation. We won't use the specifics of how the forward problem is solved in the rest of the blog post, so this part is optional.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Solution of the forward problem: computing the spectrum of the drum from its boundary</summary>

Finite element discretization, eigensolvers and all that.

</details>

So, can one hear the shape of a drum? In full accordance with [the law of headlines](https://en.wikipedia.org/wiki/Betteridge%27s_law_of_headlines), the 1992 article by Gordon, Webb, and Wolpert answered the question with a resounding 'no' <d-cite key="gordon1992isospectral"></d-cite>. There are distinct drums with exactly the same spectra, and a pair of them can be found below.<d-footnote>Why, yes, we know that they sound different!</d-footnote>

<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/side-by-side-drums_gww.html' | relative_url }}"
    width="100%"
    height="410"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="Two GWW drums">
  </iframe>
</d-figure>

Perfect reconstruction is mathematically impossible. Now what? Should we abandon all hope? Can we still solve the inverse problem somehow? Before addressing this subtle issue, we will need to discuss a few technicalities. In particular, we should settle what we mean when we say that two drums have different shapes.

## Parametrizations, symmetries and architectures

To keep things simple, we define a drum's shape as a polygon, a list of points $\left(\boldsymbol{r}_1,\dots,\boldsymbol{r}_N\right)$ where neighboring points, as well as the first and the last ones, are joined by an edge. Think of it exactly like a game of connect-the-dots.

<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/connect-the-dots.html' | relative_url }}"
      width="100%"
      height="370"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="connect-the-dots parametrisation">
  </iframe>
</d-figure>

The parametrization is non-unique: one can apply a cyclic permutation to the points $\left(\boldsymbol{r}_1,\dots,\boldsymbol{r}_N\right)$ or reverse their order and get the same drum. For a drum with $N$ points, this gives $2N$ equivalent representations. Besides that, the spectrum computed from the Laplace eigenproblem does not change if the drum is shifted, rotated, or reflected.<d-footnote>The cyclic permutations and order reversals form <a href="https://en.wikipedia.org/wiki/Dihedral_group">the dihedral group $D_{N}$</a>, while the shifts, rotations, and reflections form <a href="https://en.wikipedia.org/wiki/Euclidean_group">the Euclidean group $E(2)$</a>.</d-footnote>

<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/connect-the-dots-moons.html' | relative_url }}"
      width="100%"
      height="370"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="connect-the-dots parametrisation">
  </iframe>
</d-figure>

Symmetries complicate the evaluations and design of the architectures. We omit the details in the main text where appropriate and simply write $\left\\|\text{drum}_1 - \text{drum}_2\right\\|$ for the distance between drums, $\mathcal{N}_2: \text{drum}\rightarrow \text{spectrum}$ for the architectures, and so on. The actual operations are slightly more involved, but you won't need them to understand the inverse problem.

That being said, the actual implementation requires welding the symmetries into the metrics and architectures. So, for the readers interested in the nuts and bolts, we included a few technical drop-downs on geometric deep learning.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Symmetry groups, invariant and equivariant maps, the distance between drums</summary>

General explanation of the $D_N$ and $E(N)$. Definition of equivariant and invariant maps. Definition of invariant distance between the drums.

</details>

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Ring transformers: invariant and equivariant architectures</summary>

A description of a particular architecture (with both invariant and equivariant branches) that we use.

</details>

## Warm-up: hearing triangular drums

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Transformer for the regression problem</summary>

That is what we use for regression.

</details>

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Solution by optimization</summary>

Details on how the problem is solved by the optimization.

</details>


## Isospectral drums of Buser, Conway, Doyle, Semmler


## Closing remarks
