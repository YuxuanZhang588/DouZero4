header = open('douzero_checkpoints/resnet_v2/fields.csv').read().strip().split(',')
cols = {h: i for i, h in enumerate(header)}
rows = [l.strip().split(',') for l in open('douzero_checkpoints/resnet_v2/logs.csv') if l.strip()]
print(f'Total rows: {len(rows)}, latest frames: {rows[-1][cols["frames"]]}')
print(f"{'frames':>10} | ll_return | ll_loss  | fn_return | fn_loss")
print("-" * 65)
for r in rows[::500]:
    try:
        fr  = int(r[cols['frames']])
        ll  = float(r[cols['mean_episode_return_landlord']])
        ll_l= float(r[cols['loss_landlord']])
        fn  = float(r[cols['mean_episode_return_landlord_next']])
        fn_l= float(r[cols['loss_landlord_next']])
        print(f"{fr:>10} | {ll:>+8.3f} | {ll_l:>8.3f} | {fn:>+8.3f} | {fn_l:>8.3f}")
    except Exception as e:
        pass
# Always print last row
r = rows[-1]
fr=int(r[cols['frames']]); ll=float(r[cols['mean_episode_return_landlord']]); ll_l=float(r[cols['loss_landlord']]); fn=float(r[cols['mean_episode_return_landlord_next']]); fn_l=float(r[cols['loss_landlord_next']])
print(f"{fr:>10} | {ll:>+8.3f} | {ll_l:>8.3f} | {fn:>+8.3f} | {fn_l:>8.3f}  <-- latest")
