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
  - name: Isospectral drums of Buser, Conway, Doyle, and Semmler
  - name: Triangular drums with incomplete information
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
Of course, in the real world, the sound will depend on many other messy details, like the material and the tension of the drum's membrane, where and how one hits it, and the state of the ambient air. We will ignore all that, as mathematicians do, and strictly focus on the reconstruction of the boundary shape $\partial\Gamma$ from the spectrum $\lambda_1 \leq \lambda_2 \leq \dots$. This is our inverse problem.

While the inverse problem is notoriously difficult, the forward problem is fairly straightforward. We need it for certain methods (like Markov Chain Monte Carlo), for dataset collection, and for evaluation. We won't use the specifics of how the forward problem is solved in the rest of the blog post, so this part is optional.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Solution of the forward problem: computing the spectrum of the drum from its boundary</summary>

Let $$\Gamma$$ be the polygon enclosed by the drum's boundary. We start with $$-\Delta u = \lambda u$$ in $$\Gamma$$ and $$u=0$$ on $$\partial \Gamma$$. $$\Gamma$$ is divided into triangles. For each interior mesh vertex $$p_i$$, we define a basis function $$\psi_i$$ by setting its value to one at $$p_i$$, zero at every other vertex, and linearly interpolating those values over each triangle. Function $$\psi_i$$ is continuous across the mesh, since along the shared edge both triangles interpolate the same two endpoint values. Along the boundary it vanishes because the values at both endpoints of every boundary edge are zero. 

We approximate a vibration mode $$u$$ by a linear combination of these basis functions,

$$
u_h(x,y)=\sum_i u_i\,\psi_i(x,y),
\qquad u_i=u_h(p_i).
$$

Coefficients $$u_i$$ are the unknown we solve for.

Multiplying by a test function $$v$$ that also vanishes on the boundary and integrating by parts gives the weak formulation

$$
\int_\Gamma \nabla u\cdot\nabla v\,\mathrm dx
=
\lambda\int_\Gamma uv\,\mathrm dx.
$$

The boundary term vanishes since $$v=0$$ on $$\partial \Gamma$$. We substitute $$u_h$$ for $$u$$ and require this equation to hold for every basis function:

$$
\int_\Gamma \nabla u_h\cdot\nabla\psi_i\,\mathrm dx
=
\lambda_h\int_\Gamma u_h\psi_i\,\mathrm dx.
$$

Expanding $$u_h$$ in the basis gives one equation for each interior vertex $$i$$:

$$
\sum_j u_j\underbrace{\int_\Gamma\nabla\psi_i\cdot\nabla\psi_j\,\mathrm dx}_{K_{ij}}
=
\lambda_h\sum_j u_j\underbrace{\int_\Gamma\psi_i\psi_j\,\mathrm dx}_{M_{ij}}.
$$

Collecting these equations gives $$K\mathbf u=\lambda_h M\mathbf u$$, where $$\mathbf u=(u_1,\ldots,u_n)^\mathsf T$$ holds the coefficients that determine $$u_h$$. The eigenvalue $$\lambda_h$$ from this matrix problem approximates an eigenvalue $$\lambda$$ of the original drum.

We compute the integrals triangle by triangle. On a triangle $$T$$ of area $$A_T$$, let $$N_i$$ be the linear interpolant with value one at local vertex $$i$$ and zero at the other two. Since its gradient is constant

$$
K^{(T)}_{ij}=A_T\,\nabla N_i\cdot\nabla N_j.
$$

And integrating the products $$N_i N_j$$ gives

$$
M^{(T)}=\frac{A_T}{12}
\begin{pmatrix}
2&1&1\\
1&2&1\\
1&1&2
\end{pmatrix}.
$$

We add each triangle's contribution at the corresponding mesh vertex indices. The matrices are sparse because an entry can receive a contribution only when its two vertices belong to the same triangle.

Solving $$K\mathbf u=\lambda_h M\mathbf u$$ gives one eigenvalue per interior mesh vertex, counting multiplicities. We order them as $$\lambda_{1,h}\leq\lambda_{2,h}\leq\cdots$$. The first $$m$$ entries approximate the first $$m$$ eigenvalues of the drum.

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

The parametrization is non-unique: one can apply a cyclic permutation to the points $\left(\boldsymbol{r}_1,\dots,\boldsymbol{r}_N\right)$ or reverse their order and get the same drum. For a drum with $N$ points, this gives $2N$ equivalent representations. Besides that, the spectrum computed from the Laplace eigenproblem does not change if the drum is shifted, rotated, or reflected.<d-footnote>The cyclic permutations and order reversals form <a href="https://en.wikipedia.org/wiki/Dihedral_group">the dihedral group $D_{N}$</a>, while the shifts, rotations, and reflections form <a href="https://en.wikipedia.org/wiki/Euclidean_group">the Euclidean group $E(2)$</a>.</d-footnote> We consider two drums to have the same shape if they differ only by these symmetries. By that logic, the two mirrored moons below are the exact same drum.

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

Symmetries complicate the evaluations and design of the architectures. We omit the details in the main text where appropriate and simply write $\left\\|\text{drum}_1 - \text{drum}_2\right\\|$ for the distance between drums, $\mathcal{N}_1: \text{drum}\rightarrow \text{spectrum}$ for the architectures, and so on. The underlying operations are slightly more involved, but you won't need them to understand the inverse problem.

That being said, a working implementation requires welding the symmetries into the metrics and architectures. So, for the readers interested in the nuts and bolts, we included a few technical drop-downs on geometric deep learning.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Symmetry groups, invariant and equivariant maps, the distance between drums</summary>

General explanation of the $D_N$ and $E(N)$. Definition of equivariant and invariant maps. Definition of invariant distance between the drums.

</details>

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Ring transformers: invariant and equivariant architectures</summary>

The Ring Transformer takes a closed polygon $$X=(x_1, \cdots, x_N)$$ and a vector $$c$$ containing additional information, such as the measured spectrum. Each model uses a fixed number of vertices $$N$$. The architecture has two outputs. The **invariant** output describes the polygon as a whole, for example with a score that stays the same under symmetries described above. The **equivariant** output assigns a displacement vector to each vertex. These vectors rotate and reflect with the polygon, follow their vertices when we relabel them, and stay the same when we translate the polygon.

**Removing dependence on the drum's position and orientation.**

We subtract the mean vertex position from each vertex and build the Gram matrix $$G$$, whose entries are the dot product of the centered vertices.

$$
\overline{\mathbf x}=\frac1N\sum_{j=1}^N\mathbf x_j,
\qquad
\mathbf y_i=\mathbf x_i-\overline{\mathbf x},
\qquad
G_{ij}=\mathbf y_i^\mathsf T\mathbf y_j.
$$

If we transform every vertex by $$\mathbf x'_i=Q\mathbf x_i+\mathbf t$$, where Q is a rotation or reflection, then the mean transforms by the same rule. Subtracting it cancels $$\mathbf t$$ and gives $$\mathbf y'_i=Q\mathbf y_i$$. Taking dot products removes the dependence on Q as well, since

$$
G'_{ij}=(Q\mathbf y_i)^\mathsf T(Q\mathbf y_j)
=\mathbf y_i^\mathsf TQ^\mathsf TQ\mathbf y_j
=G_{ij}.
$$

Thus $$G$$ is invariant under the translations, rotations, and reflections in $$O(2)$$. It retains the distances between vertices, since $$\|\mathbf x_i-\mathbf x_j\|^2=G_{ii}+G_{jj}-2G_{ij}$$. The entries of $$G$$ are still indexed by vertex labels. Changing the starting vertex shifts both its rows and its columns.

**Constructing the ring tokens.**

A token is a feature vector passed to the Transformer. For vertex $$i$$, we take row $$i$$ of $$G$$ and read it in boundary order starting at $$G_{ii}$$ wrapping around at the end.

$$
\mathbf r_i=(G_{ii},G_{i,i+1},\ldots,G_{i,i+N-1}).
$$

For a triangle (i.e. when $$N=3$$), the three tokens are $$(G_{11},G_{12},G_{13})$$, $$(G_{22},G_{23},G_{21})$$, and $$(G_{33},G_{31},G_{32})$$. The first entry always refers to the vertex itself, the second to its next neighbor, and so on. If we change the starting vertex by $$s$$ positions, $$G'_{i,i+k}=G_{i+s, i+s+k}$$ and hence $$\mathbf g'_i = \mathbf g_{i+s}$$. The whole sequence of tokens shifts, and each token keeps its entries in the same order. A shared learned projection to these $$N$$ tokens is applied and a separate projection to create one more token from the context $$c$$ is used.

**Attention with a ciculant bias.**

Each attention head computes a score for every pair of tokens. The score for query token $$i$$ and key token $$j$$ is

$$
A_{ij}=\frac{\mathbf q_i^\mathsf T\mathbf k_j}{\sqrt d}+B_{ij},
$$

where $$\mathbf q_i$$ and $$\mathbf k_j$$ are learned projections of the tokens and $$d$$ is their dimension. The scores for a given query are passed through a softmax over the keys to obtain the attention weights. The bias $$B_{ij}$$ gives the head access to the relative positions of the tokens before this softmax.

For two vertex tokens, we use $$B_{ij} = b_{(j-i) \bmod N}$$. Offset $$0$$ means the same vertex, $$1$$ means the next vertex along the boundary, and $$N-1$$ means the previous vertex. The vertex block of $$B$$ is circulant, meaning each row is the preceding row shifted by one position with wraparound. For $$N=3$$, the full bias matrix is

$$
B=
\begin{pmatrix}
\gamma&\alpha&\alpha&\alpha\\
\beta&b_0&b_1&b_2\\
\beta&b_2&b_0&b_1\\
\beta&b_1&b_2&b_0
\end{pmatrix}.
$$

The first row and column belong to the context token. It has a bias $$\gamma$$ for attending to itself, a shared bias $$\alpha$$ for attending to any vertex, and a shared bias $$\beta$$ for receiving attention from any vertex. These parameters are learned separately for each attention head and layer.

Under a cyclic relabeling, the indices $$i$$ and $$j$$ both shift by the same amount, so their difference stays the same. Each pair of vertex tokens therefore keeps its bias, and the context biases are unchanged because they are shared across vertices. The attention weights and their weighted sums follow the relabeling. The remaining operations in each layer act on each token in the same way, so the final vertex tokens shift together and the final context token stays fixed.

**Reading out the two outputs.**

A learned linear layer maps the final context token to the polygon-level output $$z$$. The context token depends on the vertex coordinates through $$G$$, so translating, rotating, or reflecting the polygon leaves $$z$$ unchanged.

A second linear layer is shared across the vertex tokens. It reads $$N$$ coefficients $$a_{i,k}$$ from token $$i$$, with one coefficient for each step $$k$$ around the boundary. The output vector at vertex $$i$$ is

$$
\boldsymbol\delta_i=\sum_{k=0}^{N-1}a_{i,k}\,\mathbf y_{i+k},
$$

where the vertex indices wrap around. For a triangle, this gives

$$
\begin{aligned}
\boldsymbol\delta_1 &= a_{1,0}\mathbf y_1+a_{1,1}\mathbf y_2+a_{1,2}\mathbf y_3,\\
\boldsymbol\delta_2 &= a_{2,0}\mathbf y_2+a_{2,1}\mathbf y_3+a_{2,2}\mathbf y_1,\\
\boldsymbol\delta_3 &= a_{3,0}\mathbf y_3+a_{3,1}\mathbf y_1+a_{3,2}\mathbf y_2.
\end{aligned}
$$

In each row, the first coefficient multiplies the centered position of the current vertex, the second multiplies that of the next vertex, and the third completes the loop.

If the same triangle is written in the order $$(\mathbf r_2,\mathbf r_3,\mathbf r_1)$$. The old second vertex token is now first, so it supplies the old second row of coefficients. Those coefficients still multiply $$\mathbf y_2,\mathbf y_3,\mathbf y_1$$ in that order. The first output in the new numbering is therefore $$\boldsymbol\delta'_1=\boldsymbol\delta_2$$. The full vector output becomes $$(\boldsymbol\delta_2,\boldsymbol\delta_3,\boldsymbol\delta_1)$$, and the polygon-level output $$z$$ stays the same.

A rotation or reflection $$Q$$ leaves the coefficients $$a_{i,k}$$ unchanged and sends each centered vertex vector $$\mathbf y_j$$ to $$Q\mathbf y_j$$. Applying this to the sum gives $$Q\boldsymbol\delta_i$$. A translation leaves the centered vertices unchanged, so it leaves the output vectors unchanged as well.

**Accounting for traversal in both directions.**

Reversing the traversal changes which vertex comes next. For example, in the order $$(\mathbf r_1,\mathbf r_2,\mathbf r_3)$$, going from $$\mathbf r_2$$ to $$\mathbf r_3$$ takes one step along the boundary. In the reversed order $$(\mathbf r_3,\mathbf r_2,\mathbf r_1)$$, it takes two steps, through $$\mathbf r_1$$. The attention bias for this pair therefore changes from $$b_1$$ to $$b_2$$. In general, if another vertex is $$k$$ steps ahead in one direction, it is $$N-k$$ steps ahead in the other.

Reversing the traversal also changes the order of the dot products within each vertex token. Since the biases for different offsets are learned separately, a single pass can depend on the traversal direction. We account for this by running the same network on both vertex orders and averaging its outputs.

Let $$R$$ reverse a sequence, so $$RX=(\mathbf r_N,\ldots,\mathbf r_1)$$. With $$c$$ held fixed, write $$z(X)$$ and $$\Delta(X)$$ for the polygon-level output and the list of vertex vectors from one pass. We combine the results as

$$
\begin{aligned}
z_{\mathrm{avg}}(X)
&=\tfrac12\bigl(z(X)+z(RX)\bigr),\\
\Delta_{\mathrm{avg}}(X)
&=\tfrac12\bigl(\Delta(X)+R\Delta(RX)\bigr).
\end{aligned}
$$

The vectors in $$\Delta(RX)$$ are listed in reverse vertex order. Applying $$R$$ puts them back in the original order, so each average combines two vectors for the same vertex.

Reversing the input exchanges the two passes because reversing twice returns the original order. The polygon-level average therefore stays the same, and the list of averaged vectors reverses along with the vertices.

</details>

## Warm-up: hearing triangular drums

The simplest possible drum is a triangle, so we start by building a dataset of (triangle, the first 30 eigenvalues) pairs.<d-footnote>The setup is a standard one for statistical learning problems: drums are drawn from a distribution $p$, and the task is to reconstruct the drum shape from its spectrum <span style="font-style: italic;">within</span> this distribution. The setup is certainly a much weaker version of Kac's question, but it better corresponds to practical situations anyway. We usually have at least some idea about objects encountered in a real setting, so we encode this knowledge in the distribution $p$.</d-footnote>


A portion of the dataset is shown below.

<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/triangles-dataset.html' | relative_url }}"
      width="100%"
      height="390"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="dataset with triangular drums">
  </iframe>
</d-figure>

The problem is to reconstruct the coordinates of three points, given the spectrum. How do we solve it? We start with the first options that come to mind: regression and optimization.

**Regression**

Once we select a model $U_{\theta}$ that maps ${\sf spectrum}\equiv(\lambda_1, \dots, \lambda_p)$ to ${\sf drum}_N\equiv\boldsymbol{r}_1, \dots, \boldsymbol{r}_N$, the regression problem becomes optimizing a loss function:

$$
\theta^\star = \arg\min_{\theta} \mathbb{E}_{\sf drum}\left[\left\|U_{\theta} ({\sf spectrum}) - {\sf drum}\right\|\right].
$$

Once the distance between drums is properly defined (see above), standard first-order methods can be used to find good parameters. As an architecture, we used a transformer, described in the drop-down that follows.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Transformer for the regression problem</summary>

That is what we use for regression.

</details>

**Optimization with the exact forward model**

If $F(\boldsymbol{r}_1, \dots, \boldsymbol{r}_N) = (\lambda_1, \dots, \lambda_p)$ is a classical algorithm that solves the forward problem, one can naturally try to find

$$
\begin{split}
    {\sf drum} = &\arg\min_{\boldsymbol{r}_1, \dots, \boldsymbol{r}_N}\left[\sum_{i=1}^{p}\left(\log \widetilde{\lambda}_i - \log \lambda_i \right)^2\right],\\
    &\text{subject to }F(\boldsymbol{r}_1, \dots, \boldsymbol{r}_N) = (\widetilde{\lambda}_1, \dots, \widetilde{\lambda}_p),
\end{split}
$$

given an observed spectrum $(\lambda_1, \dots, \lambda_p)$.<d-footnote>Why the $\log$-loss? It is expected that $\lambda_i \approx i^2$, so without the logarithm, the higher frequencies get a disproportionally large weight.</d-footnote> A familiar gradient-based optimization will do as long as we can estimate the Jacobian of the forward problem $F'$. Two standard approaches are the solution of the adjoint equation and automatic differentiation.<d-footnote>Truth be told, automatic differentiation is often an adjoint equation in disguise <d-cite key="lecun1988theoretical"></d-cite>.</d-footnote>

Computation of $F'$ often becomes the most unpleasant part. For drum shape optimization, the main obstacles are geometric. See the drop-down below for the full story.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">How we compute the gradient; optimization trajectory example</summary>

<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/drum_optimization_animation_semilog_1.html' | relative_url }}"
      width="100%"
      height="685"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="triangle drum optimization example">
  </iframe>
</d-figure>

Short explanation.

</details>

**Optimization with the learned forward model**

When $F'$ is hard to compute for a classical algorithm, a common approach is to replace the forward model with a neural network $G$, since in this case $G'$ is readily available from backpropagation. The inverse problem is solved in two steps:
+ Approximate the forward problem by regression $G \approx F$.
+ Solve the optimization problem $\arg\min_{\boldsymbol{r}\_1, \dots, \boldsymbol{r}\_N}\sum\_{i=1}^{p}\left(\log \widetilde{\lambda}\_i - \log \lambda\_i \right)^2$, where $G(\boldsymbol{r}\_1, \dots, \boldsymbol{r}\_N) = (\widetilde{\lambda}\_1, \dots, \widetilde{\lambda}\_p)$.

Learned models, also called surrogate models or operators, are often much cheaper to evaluate than the exact solver. The caveat is, one needs to know in advance *where* $G$ should approximate $F$. The optimization process will explore the space of inputs, and if for some portion of the space the model is unfaithful ($G \not\approx F$), the results of the optimization can't be trusted. See below for an example.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">How we train the surrogate model; optimization trajectory examples</summary>

Optimization for the surrogate model trained on $900$ samples.
<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/drum_optimization_animation_surrogate_1000.html' | relative_url }}"
      width="100%"
      height="685"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="triangle drum optimization example">
  </iframe>
</d-figure>


Optimization for the surrogate model trained on $5000$ samples.
<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/drum_optimization_animation_surrogate_5000.html' | relative_url }}"
      width="100%"
      height="685"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="triangle drum optimization example">
  </iframe>
</d-figure>

</details>

Let's compare the three methods in an obligatory conference table. The metrics we care about are relative reconstruction error (in percent) and inference time.

| | min (%) | median (%) | mean (%) | max (%) | time per sample |
| :---: | :---: | :---: | :---: | :---: | :---: |
| regression | $\boldsymbol{0.100}$ | $\boldsymbol{0.574}$ | $\boldsymbol{0.663}$ | $\boldsymbol{2.477}$ | $\boldsymbol{3 \textbf{ ms}}$ |
| optimization, exact | $0.340$ | $1.626$ | $11.426$ | $269.291$ | $12.561 \text{ s}$ |
| optim., learned, $M=900$ | $0.180$ | $3.679$ | $5.233$ | $77.987$ | $0.673 \text{ s}$ |
| optim., learned, $M=5000$ | $\underline{0.121}$ | $\underline{0.723}$ | $\underline{2.669}$ | $\underline{23.268}$ | $\underline{0.672 \text{ s}}$|

Frustratingly enough, both regression and optimization work well! The difficulty of inverse problems is nowhere to be seen. The reason is that for triangles, the inverse problem is well-posed.

<details class="tech-details" style="background: rgba(120, 140, 220, 0.12) !important; background-color: rgba(120, 140, 220, 0.12) !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 0.75rem !important;" markdown="1">
<summary style="background: transparent !important; background-color: transparent !important; cursor: pointer; font-weight: 400;">Hearing the shape of a triangle is a well-posed problem</summary>

This is how we know that the problem is well-posed.

</details>

Even though this task is well-posed, deep learning is still useful here. The main gain comes from computational asymmetry: the forward solver is typically a fast, well-understood numerical method, while going backward requires complicated optimization. It can be much easier to generate data with the forward solver and fit a neural network than to run direct optimization every time. As our results show, when this works, it is dramatically faster.

## Isospectral drums of Buser, Conway, Doyle, and Semmler

<d-cite key="gordon1996you"></d-cite>

<d-cite key="buser2010some"></d-cite>

<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/gww_animation.html' | relative_url }}"
      width="100%"
      height="450"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="smooth parametric family of GWW drums">
  </iframe>
</d-figure>

<d-figure>
  <iframe
    src="{{ 'assets/html/2027-04-30-deep-drums/BCDS-dataset.html' | relative_url }}"
      width="100%"
      height="390"
    style="border: none; overflow: hidden; border-radius: 8px;"
    loading="lazy"
    title="dataset with BCDS drums">
  </iframe>
</d-figure>

## Triangular drums with incomplete information

## Closing remarks
