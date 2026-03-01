"""
Move generator for 4-player Doudizhu with wildcard '3'.
Generates all legal moves considering that '3' can substitute for any rank.
"""
from douzero.env.utils import MIN_SINGLE_CARDS, MIN_PAIRS, MIN_TRIPLES, select, WILDCARD_RANK
import collections
import itertools

WILD = WILDCARD_RANK  # 3


class MovesGener(object):
    """
    Generate all possible moves from a hand of cards.
    Supports wildcard '3' that can substitute for any rank.
    """
    def __init__(self, cards_list):
        self.cards_list = cards_list
        self.cards_dict = collections.defaultdict(int)

        for i in self.cards_list:
            self.cards_dict[i] += 1

        self.num_wilds = self.cards_dict.get(WILD, 0)
        
        self.single_card_moves = []
        self.gen_type_1_single()
        self.pair_moves = []
        self.gen_type_2_pair()
        self.triple_cards_moves = []
        self.gen_type_3_triple()
        self.bomb_moves = []
        self.gen_type_4_bomb()

    def _get_non_wild_ranks(self):
        """Get list of non-wild ranks in hand."""
        return [k for k in self.cards_dict.keys() if k != WILD]

    def gen_type_1_single(self):
        """Generate all single card moves."""
        self.single_card_moves = []
        for i in set(self.cards_list):
            self.single_card_moves.append([i])
        return self.single_card_moves

    def gen_type_2_pair(self):
        """
        Generate all pair moves.
        With wildcards, pairs can be:
        - Two of same non-wild rank
        - One non-wild + one wild (forming pair of non-wild rank)
        - Two wilds (forming pair of 3s)
        """
        self.pair_moves = []
        seen = set()
        
        # Natural pairs (2 of same rank)
        for k, v in self.cards_dict.items():
            if v >= 2:
                move = tuple(sorted([k, k]))
                if move not in seen:
                    self.pair_moves.append(list(move))
                    seen.add(move)
        
        # Pairs using wildcard
        if self.num_wilds >= 1:
            for k, v in self.cards_dict.items():
                if k != WILD and v >= 1:
                    # One non-wild + one wild
                    move = tuple(sorted([k, WILD]))
                    if move not in seen:
                        self.pair_moves.append(list(move))
                        seen.add(move)
        
        return self.pair_moves

    def gen_type_3_triple(self):
        """
        Generate all triple moves.
        With wildcards, triples can be formed with 0-3 wilds.
        """
        self.triple_cards_moves = []
        seen = set()
        
        # Natural triples
        for k, v in self.cards_dict.items():
            if v >= 3:
                move = tuple([k, k, k])
                if move not in seen:
                    self.triple_cards_moves.append(list(move))
                    seen.add(move)
        
        # Triples with 1 wild
        if self.num_wilds >= 1:
            for k, v in self.cards_dict.items():
                if k != WILD and v >= 2:
                    move = tuple(sorted([k, k, WILD]))
                    if move not in seen:
                        self.triple_cards_moves.append(list(move))
                        seen.add(move)
        
        # Triples with 2 wilds
        if self.num_wilds >= 2:
            for k, v in self.cards_dict.items():
                if k != WILD and v >= 1:
                    move = tuple(sorted([k, WILD, WILD]))
                    if move not in seen:
                        self.triple_cards_moves.append(list(move))
                        seen.add(move)
        
        # Pure wild triple (3 wilds = triple of 3s)
        if self.num_wilds >= 3:
            move = tuple([WILD, WILD, WILD])
            if move not in seen:
                self.triple_cards_moves.append(list(move))
                seen.add(move)
        
        return self.triple_cards_moves

    def gen_type_4_bomb(self):
        """
        Generate all bomb moves (4 of same rank).
        With wildcards, bombs can be formed with 0-4 wilds.
        """
        self.bomb_moves = []
        seen = set()
        
        # Natural bombs
        for k, v in self.cards_dict.items():
            if v == 4:
                move = tuple([k, k, k, k])
                if move not in seen:
                    self.bomb_moves.append(list(move))
                    seen.add(move)
        
        # Bombs with 1 wild
        if self.num_wilds >= 1:
            for k, v in self.cards_dict.items():
                if k != WILD and v >= 3:
                    move = tuple(sorted([k, k, k, WILD]))
                    if move not in seen:
                        self.bomb_moves.append(list(move))
                        seen.add(move)
        
        # Bombs with 2 wilds
        if self.num_wilds >= 2:
            for k, v in self.cards_dict.items():
                if k != WILD and v >= 2:
                    move = tuple(sorted([k, k, WILD, WILD]))
                    if move not in seen:
                        self.bomb_moves.append(list(move))
                        seen.add(move)
        
        # Bombs with 3 wilds
        if self.num_wilds >= 3:
            for k, v in self.cards_dict.items():
                if k != WILD and v >= 1:
                    move = tuple(sorted([k, WILD, WILD, WILD]))
                    if move not in seen:
                        self.bomb_moves.append(list(move))
                        seen.add(move)
        
        # Pure wild bomb (4 wilds = bomb of 3s)
        if self.num_wilds >= 4:
            move = tuple([WILD, WILD, WILD, WILD])
            if move not in seen:
                self.bomb_moves.append(list(move))
                seen.add(move)
        
        return self.bomb_moves

    def gen_type_6_3_1(self):
        """
        Generate all triple with single kicker moves.
        """
        result = []
        seen = set()
        
        for triple in self.triple_cards_moves:
            triple_set = set(triple)
            # Count cards used by triple
            triple_counter = collections.Counter(triple)
            
            # Find available kickers
            for kicker_rank in self.cards_dict:
                # Calculate remaining cards of kicker_rank after triple
                remaining = self.cards_dict[kicker_rank] - triple_counter.get(kicker_rank, 0)
                if remaining >= 1:
                    move = tuple(sorted(triple + [kicker_rank]))
                    if move not in seen:
                        result.append(list(move))
                        seen.add(move)
        
        return result

    def _gen_serial_moves_with_wilds(self, min_serial, repeat, repeat_num=0):
        """
        Generate serial moves (straights, consecutive pairs, consecutive triples)
        with wildcard support.
        
        Args:
            min_serial: Minimum number of consecutive ranks
            repeat: Cards per rank (1 for straight, 2 for pairs, 3 for triples)
            repeat_num: If > 0, generate only sequences of exactly this length
        """
        moves = []
        seen = set()
        
        # Valid ranks for sequences (3-14, no 2/17)
        valid_ranks = [r for r in range(3, 15)]  # 3, 4, ..., 14 (A)
        
        if repeat_num > 0:
            lengths = [repeat_num]
        else:
            lengths = range(min_serial, len(valid_ranks) + 1)
        
        for seq_len in lengths:
            # Try each starting position
            for start_idx in range(len(valid_ranks) - seq_len + 1):
                target_ranks = valid_ranks[start_idx:start_idx + seq_len]
                
                # Calculate cards needed and available
                wilds_needed = 0
                can_form = True
                cards_to_use = []
                
                for rank in target_ranks:
                    have = self.cards_dict.get(rank, 0)
                    if rank == WILD:
                        # Wild cards will be allocated later
                        continue
                    
                    if have >= repeat:
                        # Have enough of this rank
                        cards_to_use.extend([rank] * repeat)
                    else:
                        # Need wilds to fill
                        cards_to_use.extend([rank] * have)
                        wilds_needed += repeat - have
                
                # Check if we have enough wilds
                available_wilds = self.num_wilds - cards_to_use.count(WILD)
                if wilds_needed > available_wilds:
                    continue
                
                # Add wilds to complete the sequence
                cards_to_use.extend([WILD] * wilds_needed)
                
                # Verify total cards
                if len(cards_to_use) == seq_len * repeat:
                    move = tuple(sorted(cards_to_use))
                    if move not in seen:
                        moves.append(list(move))
                        seen.add(move)
        
        return moves

    def gen_type_8_serial_single(self, repeat_num=0):
        """Generate straight moves (5+ consecutive singles)."""
        return self._gen_serial_moves_with_wilds(MIN_SINGLE_CARDS, 1, repeat_num)

    def gen_type_9_serial_pair(self, repeat_num=0):
        """Generate consecutive pair moves (3+ consecutive pairs)."""
        return self._gen_serial_moves_with_wilds(MIN_PAIRS, 2, repeat_num)

    def gen_type_10_serial_triple(self, repeat_num=0):
        """Generate consecutive triple moves (2+ consecutive triples, plane without wings)."""
        return self._gen_serial_moves_with_wilds(MIN_TRIPLES, 3, repeat_num)

    def gen_type_11_serial_3_1(self, repeat_num=0):
        """
        Generate plane with single wings moves.
        Each triple needs one single kicker.
        """
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_1_moves = []
        seen = set()

        for s3 in serial_3_moves:
            s3_counter = collections.Counter(s3)
            num_triples = len(s3) // 3
            
            # Calculate remaining cards for kickers
            remaining = collections.defaultdict(int)
            for rank, count in self.cards_dict.items():
                remaining[rank] = count - s3_counter.get(rank, 0)
            
            # Get available kicker cards
            kicker_cards = []
            for rank, count in remaining.items():
                kicker_cards.extend([rank] * count)
            
            if len(kicker_cards) < num_triples:
                continue
            
            # Select kickers
            for kickers in itertools.combinations(kicker_cards, num_triples):
                move = tuple(sorted(list(s3) + list(kickers)))
                if move not in seen:
                    serial_3_1_moves.append(list(move))
                    seen.add(move)
        
        return serial_3_1_moves

    def gen_moves(self):
        """Generate all possible moves from the hand."""
        moves = []
        moves.extend(self.gen_type_1_single())
        moves.extend(self.gen_type_2_pair())
        moves.extend(self.gen_type_3_triple())
        moves.extend(self.gen_type_4_bomb())
        moves.extend(self.gen_type_6_3_1())
        moves.extend(self.gen_type_8_serial_single())
        moves.extend(self.gen_type_9_serial_pair())
        moves.extend(self.gen_type_10_serial_triple())
        moves.extend(self.gen_type_11_serial_3_1())
        return moves
