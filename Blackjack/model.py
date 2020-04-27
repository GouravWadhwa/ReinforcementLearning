import numpy as np
import matplotlib.pyplot as plt

from Blackjack import Blackjack

def action (state, policy) :
    usable_ace = 0

    if state['usable_ace'] > 0 :
        usable_ace = 1

    action = policy[usable_ace, state['player_total']-1, state['dealer_total']-1]
    return action

def generate_episode (state, policy) :
    rewards = []
    actions = []
    states = [[1 if state['usable_ace'] > 0 else 0, state['player_total']-1, state['dealer_total']-1]]
    while True :
        if state['player_total'] > 21 :
            rewards.append (-1)
            break
        else :
            rewards.append (0)

        new_action = action(state, policy)
        actions.append (new_action)

        if new_action == 1 :
            state = blackjack.hit ()
            states.append ([1 if state['usable_ace'] > 0 else 0, state['player_total']-1, state['dealer_total']-1])
        else :
            state = blackjack.stick ()
            #print (state)
            states.append ([1 if state['usable_ace'] > 0 else 0, state['player_total']-1, state['dealer_total']-1])
            
            if state['dealer_total'] > 21 or state['dealer_total'] < state['player_total'] :
                rewards.append (1)
            elif state['dealer_total'] == state['player_total'] :
                rewards.append (0)
            else :
                rewards.append (-1)
            break

    return states, actions, rewards

blackjack = Blackjack ()

average_policy = np.zeros ((2, 21, 11), dtype=np.float)

for k in range (1001) :
    policy = np.ones ((2, 21, 11))
    policy[:, 18:21, :] = 0

    q_values = np.zeros ((2, 21, 11, 2))

    for i in range (5000) :
        initial_state = blackjack.new_state ()
        states, actions, rewards = generate_episode (initial_state, policy)

        G = 0
        for j in range (len(actions)-1, -1, -1) :
            G = G + rewards[j+1]

            q_values[states[j][0], states[j][1], states[j][2], int (actions[j])] = 0.1 * q_values[states[j][0], states[j][1], states[j][2], int (actions[j])] + 0.9 * G
        
        policy = np.argmax (q_values, axis=-1)

    average_policy += policy

average_policy /= 1001
average_policy = np.round (average_policy, decimals=2)

print ("BEST POLICY OF PLAYING A GAME OF BLACKJACK")
print ("The probabilty is of doing a hit in a state")

print ("x-axis Dealer's Card")
print ("y-axis Players Card")

print ("Without Usable Ace")
print ("     2     3     4     5     6     7     8     9     10    A")
for i in range (11, 21) :
    print (i+1, end="   ")
    for j in range (10) :
        print ('%.2f'%average_policy[0, i, j+1], end='  ')
    print ()

print ()

print ("With Usable Ace")
print ("     2     3     4     5     6     7     8     9     10    A")
for i in range (11, 21) :
    print (i+1, end="   ")
    for j in range (10) :
        if i == 11 :
            print ("1.00  ", end="")
            continue
        print ('%.2f'%average_policy[1, i, j+1], end='  ')
    print ()