import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import random

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/Predict', methods=['POST', 'GET'])
def predict():
    # numpy is in python library to support large multidimensional matrices and arrays
    # R matrix
    R = np.array([[-1, -1, -1, -1, 0, -1],
                  [-1, -1, -1, 0, -1, 100],
                  [-1, -1, -1, 0, -1, -1],
                  [-1, 0, 0, -1, 0, -1],
                  [0, -1, -1, 0, -1, 100],
                  [-1, 0, -1, -1, 0, 100]])

    # Q matrix
    Q = np.zeros_like(R)

    # Gamma (learning parameter)
    gamma = 0.8

    # Initial state(Usually to be chosen at random)

    initial_state = random.choice([0, 1, 2, 3, 4])

    # This function returns all available actions in the state given as an argument
    def available_actions(state):
        current_state_row = R[state, :]
        av_act = np.where(current_state_row >= 0)[0]
        return av_act

    # This function chooses at random which to be performed within the range
    # of all available actions.

    def sample_next_actions(available_actions_range):
        next_action = int(np.random.choice(available_act, 1))
        return next_action

    # sample next action to be performed

    # This function updates the Q matrix according to the path selected and the q
    # learning algorithm
    def update(current_state, action, gamma):
        max_index = np.where(Q[action, :] == np.max(Q[action, :]))[0]

        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)
        max_value = Q[action, max_index]

        Q[current_state, action] = R[current_state, action] + gamma * max_value

        # Get available actions in the current state
        available_act = available_actions(initial_state)

        action = sample_next_actions(available_act)

    # _______________________________________________________

    # Training

    # Train over 10 000 iterations.(Re-iterate the process above)

    for i in range(100):
        current_state = np.random.randint(0, int(Q.shape[0]))
        available_act = available_actions(current_state)
        action = sample_next_actions(available_act)
        update(current_state, action, gamma)

    # Normalize the "trained" Q matrix

    print(Q / np.max(Q) * 100)

    # -----------------------------------------------------
    # Testing
    # Goal state = 5
    # Best sequence path starting from 2 -> 2,3,1,5

    if request.method == "POST":
        req = request.form.get("Room_no")
        req = int(req, 10)

    if req > 5:
        return render_template('result1.html')
    else:
        current_state = req
    steps = [current_state]

    while current_state != 5:
        next_step_index = np.where(Q[current_state, :] == np.max(Q[current_state, :]))[0]

        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size=1))

        else:
            next_step_index = int(next_step_index)

            steps.append(next_step_index)
            current_state = next_step_index

    # print selected sequence of steps

    return render_template('result.html', pred=steps)


if __name__ == '__main__':
    app.run(host="https://path-prediction-bot.herokuapp.com/")
