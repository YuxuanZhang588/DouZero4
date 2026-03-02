"""
Move selector for 4-player Doudizhu with wildcard '3'.
Filters moves that can beat a rival's move, considering wildcard effective ranks.
"""
import collections
from douzero.env.move_detector import get_move_type, WILD


def get_effective_rank(move, move_type_info=None):
    """
    Get the effective rank of a move for comparison.
    This properly handles wildcards.
    """
    if move_type_info is None:
        move_type_info = get_move_type(move)
    return move_type_info.get('rank', 0)


def common_handle(moves, rival_move):
    """
    Filter moves that have higher effective rank than rival.
    Works for all move types.
    """
    rival_info = get_move_type(rival_move)
    rival_rank = get_effective_rank(rival_move, rival_info)
    
    new_moves = []
    for move in moves:
        move_info = get_move_type(move)
        move_rank = get_effective_rank(move, move_info)
        if move_rank > rival_rank:
            new_moves.append(move)
    return new_moves


def filter_type_1_single(moves, rival_move):
    """Filter singles that beat rival single."""
    return common_handle(moves, rival_move)


def filter_type_2_pair(moves, rival_move):
    """Filter pairs that beat rival pair."""
    return common_handle(moves, rival_move)


def filter_type_3_triple(moves, rival_move):
    """Filter triples that beat rival triple."""
    return common_handle(moves, rival_move)


def filter_type_4_bomb(moves, rival_move):
    """Filter bombs that beat rival bomb."""
    return common_handle(moves, rival_move)


def filter_type_6_3_1(moves, rival_move):
    """Filter triple+single that beat rival triple+single."""
    return common_handle(moves, rival_move)


def filter_type_8_serial_single(moves, rival_move):
    """Filter straights that beat rival straight."""
    return common_handle(moves, rival_move)


def filter_type_9_serial_pair(moves, rival_move):
    """Filter consecutive pairs that beat rival consecutive pairs."""
    return common_handle(moves, rival_move)


def filter_type_10_serial_triple(moves, rival_move):
    """Filter consecutive triples that beat rival consecutive triples."""
    return common_handle(moves, rival_move)


def filter_type_11_serial_3_1(moves, rival_move):
    """Filter plane with wings that beat rival plane with wings."""
    return common_handle(moves, rival_move)


def filter_type_13_4_2(moves, rival_move):
    """Filter four-with-two (四带二) that beat rival four-with-two."""
    return common_handle(moves, rival_move)
