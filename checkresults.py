import pickle
import matplotlib.pyplot as plt
from PIL import Image

d = pickle.load(open("teachDRL/data/experiments/config_GMM/training_metrics/env_params_save.pkl", "rb"))

print(d.keys())

plt.plot(d["env_test_rewards"])
plt.savefig("Env test reward")



plt.plot(d["env_train_rewards"])
plt.savefig("Env train reward")


