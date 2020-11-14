from game import *
from neural_network_RL import *
from reward_counter import *
import matplotlib.pyplot as plt
from statistics import stdev


def make_a_choice(choice_probability):
    return np.random.choice(2, 1, p=[choice_probability, 1 - choice_probability])[0]


gamma = 0.99  # discount factor for reward
alpha = 0.05  # learning rate


ball1 = Object(Vec2d(10, 10), Vec2d(3, 5), Vec2d(0, -9.8))
platform1 = Object(Vec2d(10, 0), Vec2d(10, 0), Vec2d(0, 0), 3)
step1 = 0.05  # sec
game_size = 20

neural_net = NeuralNetwork(7, 1, 10, alpha)
reward_counter = RewardCounter(gamma)

img_size = 200  # square
canvas1 = 255 * np.ones((img_size, img_size, 3), np.uint8)
red_canvas = np.ones((img_size, img_size, 3), np.uint8)
red_canvas[:, :] = [0, 0, 200]


inputs = [np.array([0., 0., 0., 0., 0., 0., 0.]) for n in range(50000)]
a, b = [], []
n = 0
min_loss = 1
fig = plt.figure()

while True:
    for x in range(10):
        n += 1
        ball1.set_motion(Vec2d(10, 10), Vec2d(3, 5), Vec2d(0, -9.8))
        # ball1.set_motion(Vec2d(10, 10), Vec2d((np.random.rand(1)[0] - 0.5) * 12, (np.random.rand(1)[0] - 0.5) * 20), Vec2d(0, -9.8))
        platform1.set_motion(Vec2d(10, 0), Vec2d(10, 0), Vec2d(0, 0))
        for l in range(500):
            ball1.move(step1)
            platform1.move(step1)
            inputs[l] = np.array([ball1.xy.x, ball1.xy.y, ball1.v.x, ball1.v.y, platform1.xy.x, platform1.xy.y, platform1.v.x])
            out = neural_net.get_result(inputs[l])[0]  # probability to turn right
            action_choice = make_a_choice(out)

            if action_choice:
                if platform1.v.x < 0:
                    push_off_horizontally(platform1)
                else:
                    pass
            else:
                if platform1.v.x > 0:
                    push_off_horizontally(platform1)
                else:
                    pass

            if obj_crosses_the_lower_border(ball1, step1):
                if platform1.left <= ball1.xy.x <= platform1.right:
                    push_off_vertically(ball1)
                    reward_counter.reward(1.)
                else:
                    push_off_vertically(ball1)
                    reward_counter.reward(-1)
            if obj_crosses_the_upper_border(ball1, step1, game_size):
                push_off_vertically(ball1)
            if obj_crosses_the_left_or_right_borders(ball1, step1, game_size):
                push_off_horizontally(ball1)
            if obj_crosses_the_left_or_right_borders(platform1, step1, game_size):
                push_off_horizontally(platform1)
            if not n % 100:
                cv2.imshow(f'1', print_ball_and_platform(canvas1, ball1, platform1, img_size, game_size))
                cv2.waitKey(1)
            reward_counter.reward(0)
        a.append(n)
        b.append(reward_counter.loss())
        if not n % 10:
            # neural_net.alpha = neural_net.alpha / 1.01
            ssum = 0
            for z in range(len(b) - 10, len(b)):
                ssum += b[z]
            for z in range(len(b) - 10, len(b)):
                b[z] = ssum / 10
            # if b[len(b) - 1] < min_loss:
            #    neural_net.alpha = neural_net.alpha * b[len(b) - 1] / min_loss
            #   min_loss = b[len(b) - 1]
        # print(neural_net.alpha)
        if not n % 100:
            plt.plot(a, b)
            plt.draw()
            plt.pause(0.01)
            fig.clear()
    m = sum(reward_counter.rewards) / len(reward_counter.rewards)
    for i in range(5000):
        neural_net.train(np.array([inputs[i]]), np.array([[(reward_counter.rewards[i] - m) * reward_counter.loss()]]))
    reward_counter.reset(gamma)