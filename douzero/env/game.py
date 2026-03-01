"""
Game logic for 4-player Doudizhu.
1 landlord vs 3 farmers, 52 cards (no jokers), '3' is wildcard.
"""
from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener

# Card mappings (no jokers in 4-player)
EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17}

# All possible bombs (for bomb counting) - no king bomb in 4-player
bombs = [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17]]

# Player positions (4-player)
POSITIONS = ['landlord', 'landlord_next', 'landlord_across', 'landlord_prev']
FARMER_POSITIONS = ['landlord_next', 'landlord_across', 'landlord_prev']


class GameEnv(object):
    """
    4-player Doudizhu game environment.
    """
    def __init__(self, players):
        self.card_play_action_seq = []
        self.bottom_cards = None  # 4 bottom cards for landlord
        self.game_over = False
        self.acting_player_position = None
        self.player_utility_dict = None
        self.players = players
        
        # Track last move for each position
        self.last_move_dict = {pos: [] for pos in POSITIONS}
        
        # Track played cards for each position
        self.played_cards = {pos: [] for pos in POSITIONS}
        
        self.last_move = []
        self.last_three_moves = []  # Track last 3 moves (for 4-player pass detection)
        
        self.num_wins = {'landlord': 0, 'farmer': 0}
        self.num_scores = {'landlord': 0, 'farmer': 0}
        
        # Info sets for each position
        self.info_sets = {pos: InfoSet(pos) for pos in POSITIONS}
        
        self.bomb_num = 0
        self.last_pid = 'landlord'  # Last player who played non-pass
        self.consecutive_passes = 0  # Track consecutive passes

    def card_play_init(self, card_play_data):
        """Initialize game with dealt cards."""
        for pos in POSITIONS:
            self.info_sets[pos].player_hand_cards = card_play_data[pos]
        self.bottom_cards = card_play_data['bottom_cards']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        """Check if game is over (any player has no cards left)."""
        for pos in POSITIONS:
            if len(self.info_sets[pos].player_hand_cards) == 0:
                self.compute_player_utility()
                self.update_num_wins_scores()
                self.game_over = True
                return

    def compute_player_utility(self):
        """Compute utility for each side based on winner."""
        if len(self.info_sets['landlord'].player_hand_cards) == 0:
            # Landlord wins
            self.player_utility_dict = {'landlord': 3, 'farmer': -1}
        else:
            # Farmers win
            self.player_utility_dict = {'landlord': -3, 'farmer': 1}

    def update_num_wins_scores(self):
        """Update win counts and scores."""
        for pos, utility in self.player_utility_dict.items():
            base_score = 3 if pos == 'landlord' else 1
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                self.num_scores[pos] += base_score * (2 ** self.bomb_num)
            else:
                self.num_scores[pos] -= base_score * (2 ** self.bomb_num)

    def get_winner(self):
        return self.winner

    def get_bomb_num(self):
        return self.bomb_num

    def _is_bomb(self, action):
        """Check if action is a bomb (considering wildcards)."""
        if len(action) != 4:
            return False
        move_type = md.get_move_type(action)
        return move_type['type'] == md.TYPE_4_BOMB

    def step(self):
        """Execute one step of the game."""
        action = self.players[self.acting_player_position].act(self.game_infoset)
        assert action in self.game_infoset.legal_actions

        # Track consecutive passes
        if len(action) == 0:  # Pass
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
            self.last_pid = self.acting_player_position

        # Check if action is a bomb
        if self._is_bomb(action):
            self.bomb_num += 1

        # Update last move for this position
        self.last_move_dict[self.acting_player_position] = action.copy()
        
        # Add to action sequence
        self.card_play_action_seq.append(action)
        
        # Remove played cards from hand
        self.update_acting_player_hand_cards(action)
        
        # Track played cards
        self.played_cards[self.acting_player_position] += action

        # Track bottom cards usage (for landlord)
        if self.acting_player_position == 'landlord' and \
                len(action) > 0 and \
                len(self.bottom_cards) > 0:
            for card in action:
                if len(self.bottom_cards) > 0 and card in self.bottom_cards:
                    self.bottom_cards.remove(card)

        # Check if game is over
        self.game_done()
        
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        """Get the last non-pass move."""
        last_move = []
        if len(self.card_play_action_seq) != 0:
            # Find last non-pass move
            for i in range(len(self.card_play_action_seq) - 1, -1, -1):
                if len(self.card_play_action_seq[i]) > 0:
                    last_move = self.card_play_action_seq[i]
                    break
        return last_move

    def get_last_three_moves(self):
        """Get the last three moves (for 4-player pass tracking)."""
        last_three = [[], [], []]
        for i, card in enumerate(self.card_play_action_seq[-3:]):
            last_three[2-i] = card
        return last_three

    def get_acting_player_position(self):
        """
        Get next acting player (4-player rotation).
        Order: landlord -> landlord_next -> landlord_across -> landlord_prev -> landlord
        
        Also handles 3-consecutive-pass trick reset.
        """
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'
        else:
            # Find current position index and rotate
            current_idx = POSITIONS.index(self.acting_player_position)
            next_idx = (current_idx + 1) % 4
            self.acting_player_position = POSITIONS[next_idx]

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        """Remove played cards from player's hand."""
        if action != []:
            for card in action:
                self.info_sets[self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        """
        Get all legal actions for current player.
        Handles 3-consecutive-pass trick reset for 4-player.
        """
        mg = MovesGener(self.info_sets[self.acting_player_position].player_hand_cards)
        action_sequence = self.card_play_action_seq

        # Determine the rival move to beat
        rival_move = []
        
        # In 4-player: 3 consecutive passes reset the trick
        if self.consecutive_passes >= 3:
            # Trick reset - can play anything
            rival_move = []
        elif len(action_sequence) != 0:
            # Find last non-pass move
            for i in range(len(action_sequence) - 1, -1, -1):
                if len(action_sequence[i]) > 0:
                    rival_move = action_sequence[i]
                    break

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        moves = list()

        if rival_move_type == md.TYPE_0_PASS:
            # Can play any move
            moves = mg.gen_moves()

        elif rival_move_type == md.TYPE_1_SINGLE:
            all_moves = mg.gen_type_1_single()
            moves = ms.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_2_PAIR:
            all_moves = mg.gen_type_2_pair()
            moves = ms.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            all_moves = mg.gen_type_3_triple()
            moves = ms.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_4_BOMB:
            all_moves = mg.gen_type_4_bomb()
            moves = ms.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == md.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = ms.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

        # Add bombs if not already facing a bomb (bombs beat everything except bigger bombs)
        if rival_move_type not in [md.TYPE_0_PASS, md.TYPE_4_BOMB]:
            moves = moves + mg.gen_type_4_bomb()

        # Add pass option if there's a rival move to beat
        if len(rival_move) != 0:
            moves = moves + [[]]

        # Sort each move
        for m in moves:
            m.sort()

        return moves

    def reset(self):
        """Reset game state for new game."""
        self.card_play_action_seq = []
        self.bottom_cards = None
        self.game_over = False
        self.acting_player_position = None
        self.player_utility_dict = None
        
        self.last_move_dict = {pos: [] for pos in POSITIONS}
        self.played_cards = {pos: [] for pos in POSITIONS}
        
        self.last_move = []
        self.last_three_moves = []
        
        self.info_sets = {pos: InfoSet(pos) for pos in POSITIONS}
        
        self.bomb_num = 0
        self.last_pid = 'landlord'
        self.consecutive_passes = 0

    def get_infoset(self):
        """Build and return the current info set for the acting player."""
        pos = self.acting_player_position
        
        self.info_sets[pos].last_pid = self.last_pid
        self.info_sets[pos].legal_actions = self.get_legal_card_play_actions()
        self.info_sets[pos].bomb_num = self.bomb_num
        self.info_sets[pos].last_move = self.get_last_move()
        self.info_sets[pos].last_three_moves = self.get_last_three_moves()
        self.info_sets[pos].last_move_dict = self.last_move_dict
        self.info_sets[pos].consecutive_passes = self.consecutive_passes

        # Number of cards left for each player
        self.info_sets[pos].num_cards_left_dict = {
            p: len(self.info_sets[p].player_hand_cards) for p in POSITIONS
        }

        # Combine other players' hand cards
        self.info_sets[pos].other_hand_cards = []
        for p in POSITIONS:
            if p != pos:
                self.info_sets[pos].other_hand_cards += self.info_sets[p].player_hand_cards

        self.info_sets[pos].played_cards = self.played_cards
        self.info_sets[pos].bottom_cards = self.bottom_cards
        self.info_sets[pos].card_play_action_seq = self.card_play_action_seq

        # All hand cards (for training with perfect information)
        self.info_sets[pos].all_handcards = {
            p: self.info_sets[p].player_hand_cards for p in POSITIONS
        }

        return deepcopy(self.info_sets[pos])


class InfoSet(object):
    """
    Information set containing all game state information for a player.
    """
    def __init__(self, player_position):
        self.player_position = player_position
        self.player_hand_cards = None
        self.num_cards_left_dict = None
        self.bottom_cards = None  # The 4 bottom cards (shown to landlord)
        self.card_play_action_seq = None
        self.other_hand_cards = None
        self.legal_actions = None
        self.last_move = None
        self.last_three_moves = None
        self.last_move_dict = None
        self.played_cards = None
        self.all_handcards = None
        self.last_pid = None
        self.bomb_num = None
        self.consecutive_passes = None
