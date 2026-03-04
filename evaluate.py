"""
Evaluation script for 4-player Doudizhu.
"""
import os 
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu 4-Player Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_next', type=str,
            default='baselines/sl/landlord_next.ckpt')
    parser.add_argument('--landlord_across', type=str,
            default='baselines/sl/landlord_across.ckpt')
    parser.add_argument('--landlord_prev', type=str,
            default='baselines/sl/landlord_prev.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    parser.add_argument('--legacy', action='store_true',
            help='Load ALL checkpoints as legacy LSTM models (pre-ResNet)')
    parser.add_argument('--legacy_positions', type=str, default='',
            help='Comma-separated positions to load as legacy LSTM, e.g. landlord,landlord_next')
    parser.add_argument('--z_encoder', type=str, default='resnet',
            choices=['resnet', 'transformer'],
            help='z-history encoder backend used when training the checkpoint (default: resnet)')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    from douzero.env.game import POSITIONS
    if args.legacy:
        legacy_positions = set(POSITIONS)
    elif args.legacy_positions:
        legacy_positions = {p.strip() for p in args.legacy_positions.split(',') if p.strip()}
    else:
        legacy_positions = set()

    evaluate(args.landlord,
             args.landlord_next,
             args.landlord_across,
             args.landlord_prev,
             args.eval_data,
             args.num_workers,
             legacy_positions=legacy_positions,
             z_encoder=args.z_encoder)
