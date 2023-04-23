import numpy as np
import gzip, os
import urllib.request


def download_file(variable):
    url = f"https://storage.googleapis.com/atari-replay-datasets/dqn/MontezumaRevenge/1/replay_logs/%24store%24_{variable}_ckpt.0.gz"
    with urllib.request.urlopen(url) as raw:
        with gzip.open(raw, "rb") as f:
            return np.load(f)


def generate_dataset():
    print("Downloading dataset...")
    observations = download_file("observation")
    rewards = download_file("reward")

    # Get all the frames until the first reward is received,
    # skipping some invalid frames at the start.
    reward_time = np.where(rewards != 0)[0][0]
    selected = observations[3 : reward_time]

    print("Saving dataset...")
    with gzip.open("frames.gz", "wb") as file:
        np.save(file, selected)
    
    return selected


def get_dataset(n_train, n_test):
    if os.path.isfile("frames.gz"):
        print("Loading dataset...")
        with gzip.open("frames.gz", "rb") as file:
            data = np.load(file)
    else:
        data = generate_dataset()

    data = data.astype(float)
    data /= np.max(data)
    np.random.shuffle(data)

    train_data = data[:n_train]
    test_data = data[n_train : n_train + n_test]
    return train_data, test_data
