import numpy as np
from game import Paddle
from tensorflow.keras.models import load_model

model = load_model('models/avg_4951_e_490.h5')

env = Paddle(render=True)
episodes = 10

for e in range(1, episodes + 1):
    obs = env.reset()
    ep_reward = 0
    done = False

    while not done:
        env.clock.tick(60)
        action = np.argmax(model.predict(np.array(obs).reshape(1, 5)))
        obs, reward, done = env.step(action)

    print(f'Epsiode: {e}/{episodes} Reward: {ep_reward}')
