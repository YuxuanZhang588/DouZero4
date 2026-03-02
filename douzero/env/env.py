"""
Environment wrapper for 4-player Doudizhu.
Handles card dealing, observation encoding, and reward calculation.
"""
from collections import Counter
import numpy as np

from douzero.env.game import GameEnv, POSITIONS, FARMER_POSITIONS

# Card to column mapping for feature encoding (13 ranks, no jokers)
Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

# 52-card deck (no jokers) for 4-player Doudizhu
deck = []
for i in range(3, 15):  # 3-A (13 ranks)
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])  # 4 twos
# Total: 52 cards


class Env:
    """
    4-player Doudizhu environment wrapper.
    """
    def __init__(self, objective):
        """
        Initialize the environment.
        
        Args:
            objective: 'wp' (win probability), 'adp' (average doubled points), 
                      or 'logadp' for reward calculation.
        """
        self.objective = objective

        # Initialize 4 dummy players
        self.players = {}
        for position in POSITIONS:
            self.players[position] = DummyAgent(position)

        # Initialize the game environment
        self._env = GameEnv(self.players)
        self.infoset = None

    def reset(self):
        """
        Reset the environment with a new shuffled deck.
        
        4-player dealing:
        - Landlord: 16 cards (12 + 4 bottom cards)
        - Each farmer: 12 cards
        """
        self._env.reset()

        # Randomly shuffle the 52-card deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        
        # Deal cards: 12 each + 4 bottom cards for landlord
        card_play_data = {
            'landlord': _deck[:12] + _deck[48:52],  # 12 + 4 bottom = 16 cards
            'landlord_next': _deck[12:24],          # 12 cards
            'landlord_across': _deck[24:36],        # 12 cards
            'landlord_prev': _deck[36:48],          # 12 cards
            'bottom_cards': _deck[48:52],           # 4 bottom cards (visible to landlord)
        }
        
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the game
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        Execute one step with the given action.
        
        Returns:
            obs: Next observation (None if game over)
            reward: Reward for this step
            done: Whether game is finished
            info: Additional information (empty dict)
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action

def get_obs(infoset):
    """
    Obtain observations for 4-player Doudizhu.
    Routes to position-specific observation functions.
    
    Returns a dict with:
    - 'position': player position string
    - 'x_batch': features batch with action encoding
    - 'z_batch': historical action features batch
    - 'legal_actions': list of legal moves
    - 'x_no_action': features without action encoding
    - 'z': historical features (not batched)
    """
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    elif infoset.player_position == 'landlord_next':
        return _get_obs_farmer(infoset, 'landlord_next')
    elif infoset.player_position == 'landlord_across':
        return _get_obs_farmer(infoset, 'landlord_across')
    elif infoset.player_position == 'landlord_prev':
        return _get_obs_farmer(infoset, 'landlord_prev')
    else:
        raise ValueError(f'Unknown position: {infoset.player_position}')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    Utility function for one-hot encoding of card counts.
    """
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[num_left_cards - 1] = 1
    return one_hot

def _cards2array(list_cards):
    """
    Transform a list of card integers into a 52-dimensional array.
    4 rows (suits) x 13 columns (ranks), flattened.
    No jokers in 4-player Doudizhu.
    """
    if len(list_cards) == 0:
        return np.zeros(52, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card in Card2Column:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
    return matrix.flatten('F')

def _action_seq_list2array(action_seq_list):
    """
    Encode historical moves for ResNet input.
    4-player: encode 20 actions (5 rounds x 4 players).
    Output shape: (20, 52) — 20 individual moves, each a 52-dim card encoding.
    This is used as Conv1D input with in_channels=20, length=52.
    """
    action_seq_array = np.zeros((len(action_seq_list), 52))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    return action_seq_array  # Shape: (20, 52)

def _process_action_seq(sequence, length=20):
    """
    Process action sequence for LSTM encoding.
    4-player uses 20 moves (5 rounds x 4 players).
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    One-hot encode the number of bombs played.
    """
    one_hot = np.zeros(15)
    one_hot[min(bomb_num, 14)] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obtain the landlord features for 4-player Doudizhu.
    
    Features:
    - my_handcards: 52 (current hand)
    - other_handcards: 52 (all unknown cards)
    - last_action: 52 (last played cards)
    - farmer_next_played: 52 (cards played by landlord_next)
    - farmer_across_played: 52 (cards played by landlord_across)
    - farmer_prev_played: 52 (cards played by landlord_prev)
    - farmer_next_num_cards: 13 (one-hot, max 12 cards)
    - farmer_across_num_cards: 13 (one-hot, max 12 cards)
    - farmer_prev_num_cards: 13 (one-hot, max 12 cards)
    - bomb_num: 15 (one-hot)
    - my_action: 52 (action encoding)
    
    Total x_batch: 52*6 + 13*3 + 15 + 52 = 418 dims (6*52=312 + 3*13=39 + 15 + 52)
    Total x_no_action: 52*6 + 13*3 + 15 = 366 dims
    z shape: (20, 52) — 20 moves × 52-dim card encoding
    """
    num_legal_actions = len(infoset.legal_actions)
    
    # My hand cards
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    # Other hand cards (all cards not in my hand and not played)
    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    # Last action
    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    # Action encoding
    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    # Farmer next played cards
    farmer_next_played = _cards2array(
        infoset.played_cards['landlord_next'])
    farmer_next_played_batch = np.repeat(
        farmer_next_played[np.newaxis, :],
        num_legal_actions, axis=0)

    # Farmer across played cards  
    farmer_across_played = _cards2array(
        infoset.played_cards['landlord_across'])
    farmer_across_played_batch = np.repeat(
        farmer_across_played[np.newaxis, :],
        num_legal_actions, axis=0)

    # Farmer prev played cards
    farmer_prev_played = _cards2array(
        infoset.played_cards['landlord_prev'])
    farmer_prev_played_batch = np.repeat(
        farmer_prev_played[np.newaxis, :],
        num_legal_actions, axis=0)

    # Farmer num cards left (max 12 each)
    farmer_next_num_cards = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_next'], 13)
    farmer_next_num_cards_batch = np.repeat(
        farmer_next_num_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    farmer_across_num_cards = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_across'], 13)
    farmer_across_num_cards_batch = np.repeat(
        farmer_across_num_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    farmer_prev_num_cards = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_prev'], 13)
    farmer_prev_num_cards_batch = np.repeat(
        farmer_prev_num_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # Bomb count
    bomb_num = _get_one_hot_bomb(infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    # Stack features
    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         farmer_next_played_batch,
                         farmer_across_played_batch,
                         farmer_prev_played_batch,
                         farmer_next_num_cards_batch,
                         farmer_across_num_cards_batch,
                         farmer_prev_num_cards_batch,
                         bomb_num_batch,
                         my_action_batch))
    
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             farmer_next_played,
                             farmer_across_played,
                             farmer_prev_played,
                             farmer_next_num_cards,
                             farmer_across_num_cards,
                             farmer_prev_num_cards,
                             bomb_num))
    
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    
    obs = {
        'position': 'landlord',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_obs_farmer(infoset, position):
    """
    Obtain farmer features for 4-player Doudizhu.
    Unified function for landlord_next, landlord_across, landlord_prev.
    
    Features:
    - my_handcards: 52
    - other_handcards: 52
    - last_action: 52
    - landlord_played: 52
    - teammate1_played: 52 (first teammate in order)
    - teammate2_played: 52 (second teammate in order)
    - landlord_num_cards: 17 (one-hot, max 16 cards)
    - teammate1_num_cards: 13 (one-hot, max 12 cards)
    - teammate2_num_cards: 13 (one-hot, max 12 cards)
    - bomb_num: 15
    - my_action: 52
    
    Total x_batch: 52*6 + 17 + 13*2 + 15 + 52 = 422 dims (6*52=312 + 17+26+15 + 52)
    Total x_no_action: 52*6 + 17 + 13*2 + 15 = 370 dims
    z shape: (20, 52) — 20 moves × 52-dim card encoding
    """
    # Define teammate order based on position
    # Order: landlord -> landlord_next -> landlord_across -> landlord_prev -> landlord ...
    if position == 'landlord_next':
        teammates = ['landlord_across', 'landlord_prev']
    elif position == 'landlord_across':
        teammates = ['landlord_prev', 'landlord_next']
    elif position == 'landlord_prev':
        teammates = ['landlord_next', 'landlord_across']
    else:
        raise ValueError(f'Unknown farmer position: {position}')
    
    num_legal_actions = len(infoset.legal_actions)
    
    # My hand cards
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    # Other hand cards
    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    # Last action
    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    # Action encoding
    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    # Landlord played cards
    landlord_played = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_batch = np.repeat(
        landlord_played[np.newaxis, :],
        num_legal_actions, axis=0)

    # Teammate 1 played cards
    teammate1_played = _cards2array(
        infoset.played_cards[teammates[0]])
    teammate1_played_batch = np.repeat(
        teammate1_played[np.newaxis, :],
        num_legal_actions, axis=0)

    # Teammate 2 played cards
    teammate2_played = _cards2array(
        infoset.played_cards[teammates[1]])
    teammate2_played_batch = np.repeat(
        teammate2_played[np.newaxis, :],
        num_legal_actions, axis=0)

    # Landlord num cards left (max 16)
    landlord_num_cards = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 17)
    landlord_num_cards_batch = np.repeat(
        landlord_num_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # Teammate 1 num cards left (max 12)
    teammate1_num_cards = _get_one_hot_array(
        infoset.num_cards_left_dict[teammates[0]], 13)
    teammate1_num_cards_batch = np.repeat(
        teammate1_num_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # Teammate 2 num cards left (max 12)
    teammate2_num_cards = _get_one_hot_array(
        infoset.num_cards_left_dict[teammates[1]], 13)
    teammate2_num_cards_batch = np.repeat(
        teammate2_num_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    # Bomb count
    bomb_num = _get_one_hot_bomb(infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    # Stack features
    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         landlord_played_batch,
                         teammate1_played_batch,
                         teammate2_played_batch,
                         landlord_num_cards_batch,
                         teammate1_num_cards_batch,
                         teammate2_num_cards_batch,
                         bomb_num_batch,
                         my_action_batch))
    
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_played,
                             teammate1_played,
                             teammate2_played,
                             landlord_num_cards,
                             teammate1_num_cards,
                             teammate2_num_cards,
                             bomb_num))
    
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    
    obs = {
        'position': position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs
