import numpy as np

class Blackjack () :
    def new_state (self) :
        player_card_1 = np.random.randint (2, 12)
        player_card_2 = np.random.randint (2, 12)

        dealer_card = np.random.randint (2, 12)

        usable_ace = 0

        if player_card_1 == 11 :
            usable_ace += 1
        if player_card_2 == 11 :
            usable_ace += 1

        player_total = player_card_1 + player_card_2

        self.current_state =  {
            'usable_ace': usable_ace,
            'player_total' : player_total,
            'dealer_total' : dealer_card,
            'dealer_ace' : 1 if dealer_card == 1 else 0
        }

        return self.current_state

    def hit (self) :
        player_card = np.random.randint (2, 12)

        if player_card == 11 :
            self.current_state['usable_ace'] += 1

        self.current_state['player_total'] += player_card

        if self.current_state['player_total'] > 21 and self.current_state['usable_ace'] > 0 :
            self.current_state['player_total'] -= 10
            self.current_state['usable_ace'] -= 1

        return self.current_state

    def stick (self) :
        dealer_ace = 0

        if self.current_state['dealer_ace'] == 1 :
            self.current_state['dealer_total'] = 11

        while self.current_state['dealer_total'] <= self.current_state['player_total'] :
            dealer_card = np.random.randint (2, 12) 

            if dealer_card == 11 :
                self.current_state['dealer_ace'] += 1

            self.current_state['dealer_total'] += dealer_card

            if self.current_state['dealer_total'] > 21 and self.current_state['dealer_ace'] > 0 :
                self.current_state['dealer_total'] -= 10
                self.current_state['dealer_ace'] -= 1

        return self.current_state
