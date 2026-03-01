"""
Generate evaluation data for 4-player Doudizhu.
"""
import argparse
import pickle
import numpy as np

# 52-card deck (no jokers for 4-player)
deck = []
for i in range(3, 15):  # 3-A (13 ranks)
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])  # 4 twos
# Total: 52 cards

def get_parser():
    parser = argparse.ArgumentParser(description='4-Player Doudizhu: random data generator')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=10000, type=int)
    return parser
    
def generate():
    """
    Generate a random game configuration for 4-player Doudizhu.
    
    Card distribution:
    - Landlord: 16 cards (12 + 4 bottom cards)
    - Each farmer: 12 cards
    """
    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_data = {
        'landlord': _deck[:12] + _deck[48:52],  # 12 + 4 bottom = 16 cards
        'landlord_next': _deck[12:24],           # 12 cards
        'landlord_across': _deck[24:36],         # 12 cards
        'landlord_prev': _deck[36:48],           # 12 cards
        'bottom_cards': _deck[48:52],            # 4 bottom cards (visible to landlord)
    }
    for key in card_play_data:
        card_play_data[key].sort()
    return card_play_data


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'

    print("output_pickle:", output_pickle)
    print("generating data...")

    data = []
    for _ in range(flags.num_games):
        data.append(generate())

    print("saving pickle file...")
    with open(output_pickle,'wb') as g:
        pickle.dump(data,g,pickle.HIGHEST_PROTOCOL)




