
- Setup visualization
    - Latent space representation

- Check Gan Latent Space Gives modes of environments

- SPend some tume understanding why gpu utils so low (parallel?)

- Make sure test set works on small experiments


- Plot KL divergence
- Plot ALP-GMM Visualization
- Look failed mazes (0 average reward)
- Check GPU
- Difference maze looks (make gif sequential mazes)
- Why random better than alp
- Train test set optimal policy, number of steps required (ex bfs) + order in difficulty
- Exploration -> sparse reward




Main components:
- Visualization:
    - Latent Space GMM
    - Mazes Sampled with training
    - Plot KL Divergence

- Debugging:
    - Computing max reward per mazes to normalize
    - Compute Average reward for 
    - Why ALP-GMM worse than random?
    - Why policy does not learn anything?

- Hypothesis testing:
    - GAN multimodal

TODO Now:
make all sampled maze solvable


- Z dimension not hard coded
- SIze test set not hard coded



- Make code cleaner:
    - Make log simpler to understand + config for place
        - ALP-GMM visualization
- Code for plotting mazes from numpy arrays:
    - Plot
    - Make emplacement clean
