# Autoencoders for Montezuma's Revenge

### Blog post: [https://siliconsloth.com/posts/autoencoders-for-montezumas-revenge/](https://siliconsloth.com/posts/autoencoders-for-montezumas-revenge/)

Implementations of several different autoencoders for frames from Montezuma's Revenge.

Running `train.py` will download the dataset of game frames and train three different autoencoders on them,
then generate figures comparing their accuracy.

Running `interactive.py` will start a playable version of the game in the Arcade Learning Environment,
with reconstructions from one of the trained autoencoders shown alongside the original frames.
This requires that you have Montezuma's Revenge installed in your ALE, as explained [here](https://github.com/mgbellemare/Arcade-Learning-Environment).

This project is explained in more detail in the corresponding blog post at [siliconsloth.com](https://siliconsloth.com/posts/autoencoders-for-montezumas-revenge/).

## Acknowledgements

The training code is partially based on the Keras variational autoencoder tutorial at
[https://keras.io/examples/generative/vae/](https://keras.io/examples/generative/vae/).

This software downloads and uses a small subset of the Google Research DQN Replay Dataset, which can be found at
[https://research.google/resources/datasets/dqn-replay/](https://research.google/resources/datasets/dqn-replay/).