import os

fields_path = 'douzero_checkpoints/attn_v1/fields.csv'
logs_path   = 'douzero_checkpoints/attn_v1/logs.csv'

if not os.path.exists(logs_path):
    print('logs.csv not found at', logs_path)
else:
    header = open(fields_path).read().strip().split(',')
    cols   = {h: i for i, h in enumerate(header)}
    rows   = [l.strip().split(',') for l in open(logs_path) if l.strip()]

    latest_frames = rows[-1][cols['frames']]
    print(f'Total rows: {len(rows)}, latest frames: {latest_frames}')
    print(f'{"frames":>10} | ll_return |  ll_loss | fn_return |  fn_loss')
    print('-' * 65)

    step = max(1, len(rows) // 20)
    shown = rows[::step]
    if rows[-1] not in shown:
        shown.append(rows[-1])

    for r in shown:
        try:
            fr  = int(r[cols['frames']])
            ll  = float(r[cols['mean_episode_return_landlord']])
            lll = float(r[cols['loss_landlord']])
            fn  = float(r[cols['mean_episode_return_landlord_next']])
            fnl = float(r[cols['loss_landlord_next']])
            marker = '  <-- latest' if r == rows[-1] else ''
            print(f'{fr:>10} | {ll:>+8.3f} | {lll:>8.3f} | {fn:>+8.3f} | {fnl:>8.3f}{marker}')
        except Exception:
            pass
