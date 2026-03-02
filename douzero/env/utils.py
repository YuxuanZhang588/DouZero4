import itertools

# global parameters
MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2

# Wildcard: '3' can substitute for any rank
WILDCARD_RANK = 3

# action types (4-player Doudizhu)
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4
# TYPE_5_KING_BOMB removed (no jokers in 4-player)
TYPE_6_3_1 = 6  # Triple with single
# TYPE_7_3_2 removed (not supported in 4-player)
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10
TYPE_11_SERIAL_3_1 = 11
# TYPE_12_SERIAL_3_2 removed (not supported in 4-player)
TYPE_13_4_2 = 13       # Four of a kind with two kicker cards (四带二)
# TYPE_14_4_22 removed (not supported in 4-player)
TYPE_15_WRONG = 15

# Player positions (4-player)
POSITIONS = ['landlord', 'landlord_next', 'landlord_across', 'landlord_prev']

# betting round action
PASS = 0
CALL = 1
RAISE = 2

# Rank ordering for comparisons (3 is lowest as natural, 2 is highest)
# 3=3, 4=4, ..., 14=A, 17=2
RANK_ORDER = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]

def rank_value(rank):
    """Get comparison value for a rank (higher = better)"""
    if rank == 17:  # '2' is highest
        return 15
    return rank  # 3-14 (3-A)

# return all possible results of selecting num cards from cards list
def select(cards, num):
    return [list(i) for i in itertools.combinations(cards, num)]
