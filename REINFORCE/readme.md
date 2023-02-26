# REINFORCE

---

## cartpole_reinforce.py
basic reinforce algorithm implement with CartPole-v1 [Gym CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
<br>
CartPole has...
- 4 Observation space (Cart position, Cart velocity, Pole Angle, Pole Angular velocity)
- Discrete action space (move left or move right, action space size is 2)
<br>
<img src="./img/Cartpole.png"></img>

---

## cartpole_improved_reinforce.py
improved reinforce algorithm implement with CartPole-v1 [Gym CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
<br>
What is improved reinforce algorithm?
In basic reinforce, we update policy network with Gt
But in improved reinforce, we update policy network with Gt normalization. This will help to Gt has lower variance
<br>
Implement
Average all Gt history. This is b (b has no effect to Rs)
When update Policy Network, we replace Gt to Gt - b (check cartpole_improved_reinforce.py line: 50-54)


## Refer this
code is very short (under 200 lines), no need to seperate file (like PolicyNet)