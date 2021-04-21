import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces


class snake_game(gym.Env):

    N = 10

    board = np.zeros((0,0))         # main data structure to store all information.
                                    # 0 - empty, 1 - occupied by snake, 2 - reward

    snake_pos = np.empty((0,2))     # position of snake
    snake_dir = 0                   # direction of snake

    rng = None                      # random number generator
    target_pos = np.array([0,0])    # reward position

    control_codec = None            # parse control into change in coordinate
    score = 0

    game_ended = True

    def __init__(self, N=10):  

        super().__init__()      

        self.N = N

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 3, (self.N+2,self.N+2))

        self.rng = np.random.default_rng()

        self.control_codec = {
            0   :   np.array([0,1]),
            1   :   np.array([-1,0]),
            2   :   np.array([0,-1]),
            3   :   np.array([1,0]),
        }

    def reset(self):        
        # initialize meta-data
        self.snake_pos = np.array([[self.N/2 + 1,self.N/2 + 1]], dtype=np.int64)

        # initialize board
        self.board = np.ones((self.N + 2,self.N + 2))
        self.board[1:1+self.N,1:1+self.N] = 0

        self.score = 0
        self.game_ended = False

        # initialize snake
        self.board[self.snake_pos[0,0],self.snake_pos[0,1]] = 1

        # sample, initialize reward
        self.target_pos = self.sample_target()
        self.board[self.target_pos[0],self.target_pos[1]] = 2

        return self.extract_state()

    def step(self, action):

        assert self.action_space.contains(action)

        if not (self.snake_dir - action % 4 == 2):
            new_head_pos = self.snake_pos[0,:] + self.control_codec[action]
            self.snake_dir = action
        else:
            action = self.snake_dir
            new_head_pos = self.snake_pos[0,:] + self.control_codec[action]
        
        if (self.board[new_head_pos[0],new_head_pos[1]] == 1):
            # new snake pos is wall or snake
            s =  self.extract_state()
            return s, -100, True, None                          # reward for stuck    

        elif (self.board[new_head_pos[0],new_head_pos[1]] == 2):
            # new snake pos is reward
            self.snake_pos = np.vstack((new_head_pos, self.snake_pos))
            self.board[new_head_pos[0],new_head_pos[1]] = 1

            self.score += 1

            self.target_pos = self.sample_target()
            self.board[self.target_pos[0],self.target_pos[1]] = 2

            return self.extract_state(), 10, False, None         # reward for successful target reach

        else:
            # new snake pos is empty
            old_head_dis = np.sum(np.abs(self.snake_pos[0,:] - self.target_pos))

            self.board[self.snake_pos[-1,0],self.snake_pos[-1,1]] = 0 
            self.snake_pos = np.vstack((new_head_pos, self.snake_pos[:-1,:]))
            self.board[new_head_pos[0],new_head_pos[1]] = 1

            new_head_dis = np.sum(np.abs(self.snake_pos[0,:] - self.target_pos))

            if (new_head_dis < old_head_dis):
                return self.extract_state(), 1, False, None     
            elif (new_head_dis > old_head_dis):
                return self.extract_state(), -1, False, None   
            else:
                return self.extract_state(), 0, False, None   

    def render(self, mode='human', block=True):

        if (mode == 'human'):
            plt.imshow((self.extract_state() * 64).astype(np.uint8))
            plt.show(block=block)
        else:
            return (self.extract_state() * 64).astype(np.uint8)

    def sample_target(self):
        available_inds = np.argwhere(self.board == 0)
        i = self.rng.integers(0, available_inds.shape[0])

        return available_inds[i,:]

    def extract_state(self):
        state = self.board.copy()
        state[self.snake_pos[0,0],self.snake_pos[0,1]] = 3
        return state

if (__name__ == "__main__"):

    env = snake_game(10)

    env.reset()

    while True:
        
        env.render()

        next_state, reward, done, _ = env.step(int(input("Input control: \n")))

        print(next_state)
        print(reward)        

        if (done):
            break        

