"""
Move detector for 4-player Doudizhu with wildcard '3'.
Detects the type of a move considering that '3' can substitute for any rank.
"""
from douzero.env.utils import *
import collections

# Wildcard is rank 3
WILD = WILDCARD_RANK  # 3


def _num_wild(move):
    """Count how many wildcards (3s) are in the move."""
    return sum(1 for c in move if c == WILD)


def _non_wild_counts(move):
    """Return a Counter of non-wild cards only."""
    return collections.Counter(c for c in move if c != WILD)


def _best_set_rank(move, set_size):
    """
    Determine the best (highest) effective rank for a set of 'set_size' cards.
    If the move is ALL wildcards, treat as natural 3 (lowest).
    Otherwise, use wildcards to complete the set at the highest non-wild rank.
    Returns (effective_rank, is_valid).
    """
    if len(move) != set_size:
        return (None, False)
        
    wilds = _num_wild(move)
    non_wild = _non_wild_counts(move)
    
    # All wildcards -> natural set of 3s (lowest)
    if len(move) == wilds:
        return (WILD, True)
    
    # For a valid set, all non-wild cards must be the same rank
    if len(non_wild) != 1:
        return (None, False)
    
    # The single non-wild rank forms the set with wilds
    rank = list(non_wild.keys())[0]
    count = non_wild[rank]
    
    # Check if we can complete the set
    if count + wilds >= set_size:
        return (rank, True)
    
    return (None, False)


def _best_bomb_rank(move):
    """
    Determine if move is a valid bomb (4 of same rank with wildcards).
    Returns (effective_rank, is_valid).
    """
    if len(move) != 4:
        return (None, False)
    return _best_set_rank(move, 4)


def _straight_check(move, required_len):
    """
    Check if move forms a valid straight (consecutive singles).
    Straights cannot include '2' (rank 17). Minimum length is 5.
    Returns (start_rank, length, is_valid).
    
    Wildcards can fill gaps in the sequence.
    """
    if len(move) != required_len or required_len < MIN_SINGLE_CARDS:
        return (None, 0, False)
    
    wilds = _num_wild(move)
    non_wild = [c for c in move if c != WILD]
    non_wild_set = set(non_wild)
    
    # Straights cannot contain 2 (rank 17)
    if 17 in non_wild_set:
        return (None, 0, False)
    
    # Check for duplicates in non-wild (each rank appears once in straight)
    if len(non_wild) != len(non_wild_set):
        return (None, 0, False)
    
    # All wildcards -> form lowest straight starting at 4 (using 3 as wild for position 3)
    # Actually straights go from rank 3 to 14, but 3 is wild
    # All-wild straight: positions 3,4,5,6,7 -> effective start is 3 (lowest)
    if not non_wild:
        return (3, required_len, True)
    
    min_rank = min(non_wild_set)
    max_rank = max(non_wild_set)
    
    # The span of non-wild cards
    span = max_rank - min_rank + 1
    
    if span > required_len:
        # Non-wild cards span too wide for the straight length
        return (None, 0, False)
    
    # Find the best (highest) starting position for the straight
    # Constraints:
    # - start >= 3 (minimum card in straight)
    # - start + required_len - 1 <= 14 (maximum is A, 2 not allowed)
    # - Window must contain all non-wild cards
    
    # The window [start, start+required_len-1] must contain min_rank and max_rank
    earliest_start = max(3, max_rank - required_len + 1)
    latest_start = min(min_rank, 14 - required_len + 1)
    
    if earliest_start > latest_start:
        return (None, 0, False)
    
    # Find the highest valid start (gives highest straight)
    for start in range(latest_start, earliest_start - 1, -1):
        # Count how many wilds needed to fill this window
        needed_wilds = 0
        for r in range(start, start + required_len):
            if r not in non_wild_set:
                needed_wilds += 1
        
        if needed_wilds <= wilds:
            return (start, required_len, True)
    
    return (None, 0, False)


def _serial_pair_check(move, required_pairs):
    """
    Check if move forms consecutive pairs (min 3 pairs = 6 cards).
    Cannot include '2' (rank 17).
    Returns (start_rank, num_pairs, is_valid).
    """
    if len(move) != required_pairs * 2 or required_pairs < MIN_PAIRS:
        return (None, 0, False)
    
    wilds = _num_wild(move)
    non_wild_counter = collections.Counter(c for c in move if c != WILD)
    
    # Cannot include 2
    if 17 in non_wild_counter:
        return (None, 0, False)
    
    # Check each non-wild rank has at most 2 cards
    for r, cnt in non_wild_counter.items():
        if cnt > 2:
            return (None, 0, False)
    
    # All wilds -> lowest consecutive pairs starting at 3
    if not non_wild_counter:
        return (WILD, required_pairs, True)
    
    non_wild_ranks = sorted(non_wild_counter.keys())
    min_rank = min(non_wild_ranks)
    max_rank = max(non_wild_ranks)
    
    span = max_rank - min_rank + 1
    if span > required_pairs:
        return (None, 0, False)
    
    earliest_start = max(3, max_rank - required_pairs + 1)
    latest_start = min(min_rank, 14 - required_pairs + 1)
    
    if earliest_start > latest_start:
        return (None, 0, False)
    
    # Find highest valid start
    for start in range(latest_start, earliest_start - 1, -1):
        needed_wilds = 0
        for r in range(start, start + required_pairs):
            cnt = non_wild_counter.get(r, 0)
            needed_wilds += (2 - cnt)
        
        if needed_wilds <= wilds:
            return (start, required_pairs, True)
    
    return (None, 0, False)


def _serial_triple_check(move, required_triples):
    """
    Check if move forms consecutive triples (min 2 triples = 6 cards).
    Cannot include '2' (rank 17).
    Returns (start_rank, num_triples, is_valid).
    """
    if len(move) != required_triples * 3 or required_triples < MIN_TRIPLES:
        return (None, 0, False)
    
    wilds = _num_wild(move)
    non_wild_counter = collections.Counter(c for c in move if c != WILD)
    
    if 17 in non_wild_counter:
        return (None, 0, False)
    
    for r, cnt in non_wild_counter.items():
        if cnt > 3:
            return (None, 0, False)
    
    if not non_wild_counter:
        return (WILD, required_triples, True)
    
    non_wild_ranks = sorted(non_wild_counter.keys())
    min_rank = min(non_wild_ranks)
    max_rank = max(non_wild_ranks)
    
    span = max_rank - min_rank + 1
    if span > required_triples:
        return (None, 0, False)
    
    earliest_start = max(3, max_rank - required_triples + 1)
    latest_start = min(min_rank, 14 - required_triples + 1)
    
    if earliest_start > latest_start:
        return (None, 0, False)
    
    for start in range(latest_start, earliest_start - 1, -1):
        needed_wilds = 0
        for r in range(start, start + required_triples):
            cnt = non_wild_counter.get(r, 0)
            needed_wilds += (3 - cnt)
        
        if needed_wilds <= wilds:
            return (start, required_triples, True)
    
    return (None, 0, False)


def _classify_3_1(move):
    """
    Check if move is triple with single kicker (4 cards).
    Returns (triple_rank, is_valid).
    """
    if len(move) != 4:
        return (None, False)
    
    wilds = _num_wild(move)
    non_wild_counter = collections.Counter(c for c in move if c != WILD)
    
    # 4 wilds = bomb, not 3+1
    if wilds == 4:
        return (None, False)
    
    if wilds == 3:
        # 3 wilds + 1 non-wild: wilds form triple of 3, non-wild is kicker
        if len(non_wild_counter) == 1:
            return (WILD, True)
        return (None, False)
    
    if wilds == 2:
        # 2 wilds + 2 non-wilds
        if len(non_wild_counter) == 1:
            # Both non-wilds same rank: triple of that rank + wild kicker
            rank = list(non_wild_counter.keys())[0]
            return (rank, True)
        elif len(non_wild_counter) == 2:
            # Two different ranks: highest forms triple with wilds
            ranks = sorted(non_wild_counter.keys(), reverse=True)
            return (ranks[0], True)
        return (None, False)
    
    if wilds == 1:
        # 1 wild + 3 non-wilds
        if len(non_wild_counter) == 1:
            # All same rank = triple, wild is kicker
            rank = list(non_wild_counter.keys())[0]
            return (rank, True)
        elif len(non_wild_counter) == 2:
            # Find rank with 2 cards (can form triple with wild)
            for r, cnt in non_wild_counter.items():
                if cnt == 2:
                    return (r, True)
            return (None, False)
        return (None, False)
    
    # No wilds: need exactly 3 of one rank and 1 of another
    if len(non_wild_counter) == 2:
        for r, cnt in non_wild_counter.items():
            if cnt == 3:
                return (r, True)
    
    return (None, False)


def _classify_serial_3_1(move, num_triples):
    """
    Check if move is consecutive triples with single kickers (plane with wings).
    Total cards = num_triples * 4
    Returns (start_rank, is_valid).
    """
    expected_len = num_triples * 4
    if len(move) != expected_len or num_triples < MIN_TRIPLES:
        return (None, False)
    
    wilds = _num_wild(move)
    non_wild_counter = collections.Counter(c for c in move if c != WILD)
    
    # All wilds
    if not non_wild_counter:
        return (WILD, True)
    
    # Try different starting positions for the triple sequence
    # The triple ranks cannot include 2 (17), but kickers can
    for start in range(14, 2, -1):
        end = start + num_triples - 1
        if end > 14:
            continue
        
        # Calculate wilds needed for triples in [start, end]
        wilds_for_triples = 0
        triple_cards_used = {}
        valid = True
        
        for r in range(start, start + num_triples):
            cnt = non_wild_counter.get(r, 0)
            use = min(cnt, 3)
            need = 3 - use
            wilds_for_triples += need
            triple_cards_used[r] = use
            # Check if this rank has too many cards (more than 3 would be a problem)
        
        if wilds_for_triples > wilds:
            continue
        
        # Count remaining cards for kickers
        remaining_wilds = wilds - wilds_for_triples
        remaining_non_wild = 0
        for r, cnt in non_wild_counter.items():
            if r in triple_cards_used:
                remaining_non_wild += cnt - triple_cards_used[r]
            else:
                remaining_non_wild += cnt
        
        total_kickers = remaining_wilds + remaining_non_wild
        if total_kickers == num_triples:
            return (start, True)
    
    return (None, False)


def get_move_type(move):
    """
    Detect the type of a move in 4-player Doudizhu with wildcard '3'.
    Returns a dict with 'type' and optionally 'rank' and 'len'.
    """
    move = sorted(move)
    move_size = len(move)
    
    if move_size == 0:
        return {'type': TYPE_0_PASS}
    
    # Single
    if move_size == 1:
        return {'type': TYPE_1_SINGLE, 'rank': move[0]}
    
    # Pair
    if move_size == 2:
        rank, valid = _best_set_rank(move, 2)
        if valid:
            return {'type': TYPE_2_PAIR, 'rank': rank}
        return {'type': TYPE_15_WRONG}
    
    # Triple
    if move_size == 3:
        rank, valid = _best_set_rank(move, 3)
        if valid:
            return {'type': TYPE_3_TRIPLE, 'rank': rank}
        return {'type': TYPE_15_WRONG}
    
    # 4 cards: Bomb or Triple+Single
    if move_size == 4:
        # Check bomb first (higher priority)
        rank, valid = _best_bomb_rank(move)
        if valid:
            return {'type': TYPE_4_BOMB, 'rank': rank}
        
        # Check triple with single
        rank, valid = _classify_3_1(move)
        if valid:
            return {'type': TYPE_6_3_1, 'rank': rank}
        
        return {'type': TYPE_15_WRONG}
    
    # 5+ cards: check various types
    
    # Check straight (5+ consecutive singles)
    if move_size >= MIN_SINGLE_CARDS:
        start, length, valid = _straight_check(move, move_size)
        if valid:
            return {'type': TYPE_8_SERIAL_SINGLE, 'rank': start, 'len': length}
    
    # Check serial pairs (3+ consecutive pairs)
    if move_size >= MIN_PAIRS * 2 and move_size % 2 == 0:
        num_pairs = move_size // 2
        start, _, valid = _serial_pair_check(move, num_pairs)
        if valid and num_pairs >= MIN_PAIRS:
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': start, 'len': num_pairs}
    
    # Check serial triples / plane (2+ consecutive triples)
    if move_size >= MIN_TRIPLES * 3 and move_size % 3 == 0:
        num_triples = move_size // 3
        start, _, valid = _serial_triple_check(move, num_triples)
        if valid and num_triples >= MIN_TRIPLES:
            return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': start, 'len': num_triples}
    
    # Check serial 3+1 / plane with single wings (2+ triples with kickers)
    if move_size >= MIN_TRIPLES * 4 and move_size % 4 == 0:
        num_triples = move_size // 4
        if num_triples >= MIN_TRIPLES:
            start, valid = _classify_serial_3_1(move, num_triples)
            if valid:
                return {'type': TYPE_11_SERIAL_3_1, 'rank': start, 'len': num_triples}
    
    return {'type': TYPE_15_WRONG}


def get_move_effective_rank(move, move_type_info=None):
    """
    Get the effective rank of a move for comparison purposes.
    This accounts for wildcards.
    """
    if move_type_info is None:
        move_type_info = get_move_type(move)
    
    return move_type_info.get('rank', 0)


# Legacy function for compatibility
def is_continuous_seq(move):
    """Check if move is a continuous sequence (without considering wildcards)."""
    if len(move) < 2:
        return True
    move = sorted(move)
    for i in range(len(move) - 1):
        if move[i+1] - move[i] != 1:
            return False
    return True
