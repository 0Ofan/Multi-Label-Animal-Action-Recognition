"""Controlled old/new-loss fitting check on 16 fixed Train videos, never validation."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .action_detection import (
    SCHEMA, ActionDetector, action_targets, decode_actions, detection_loss,
    evaluate_detections, load_records,
)


def reference_loss(logits, presence, target, mask):
    """Frozen pre-change objective, retained only for this controlled comparison."""
    valid = mask[:, :, None, None].expand_as(target)
    def balanced(prediction, truth):
        return (-(truth * nn.functional.logsigmoid(prediction)).sum() / truth.sum().clamp_min(1)
                - ((1 - truth) * nn.functional.logsigmoid(-prediction)).sum() / (1 - truth).sum().clamp_min(1))
    return balanced(logits[valid], target[valid]) + balanced(presence, target[:, :, :, 0].amax(1).gt(0).float())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('proposal/boundary_results/features'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    torch.set_num_threads(1)
    data, classes = load_records(args.data, splits=('train',))
    # Exclude sub-bin events here to isolate learning from sampling resolution.
    eligible = [r for r in sorted(data['train'], key=lambda r: r['video_id'])
                if 1.5 <= r['duration'] <= 20.
                and all(e['interval'][1] - e['interval'][0] >= .5 for e in r['events'])
                and any(e['interval'][0] >= .5 and e['interval'][1] <= r['duration'] - .5 for e in r['events'])]
    random.Random(6116).shuffle(eligible)
    rows, groups = [], set()
    for row in eligible:
        if row['split_group_id'] in groups:
            continue
        rows.append(row)
        groups.add(row['split_group_id'])
        if len(rows) == 16:
            break
    if len(rows) != 16:
        raise ValueError('Need 16 eligible distinct Train groups')
    cached = {}
    for row in rows:
        value = torch.load(args.cache / (row['video_id'] + '.pt'), weights_only=True, map_location='cpu')
        if value['rate'] != 4. or value['source_sha256'] != sha(args.data / row['video_path']):
            raise ValueError('Feature cache provenance mismatch')
        cached[row['video_id']] = (value['features'], action_targets(row['events'], value['edges']), value['edges'])
    def batch(part):
        values = [cached[r['video_id']] for r in part]
        x = pad_sequence([v[0] for v in values], batch_first=True)
        y = pad_sequence([v[1] for v in values], batch_first=True)
        lengths = torch.tensor([len(v[0]) for v in values])
        return x, y, lengths, torch.arange(x.shape[1])[None] < lengths[:, None]
    @torch.inference_mode()
    def evaluate(model):
        model.eval()
        predictions = []
        for offset in range(0, 16, 8):
            part = rows[offset:offset + 8]
            x, _, lengths, mask = batch(part)
            logits, presence = model(x, lengths, mask)
            for i, row in enumerate(part):
                predictions.append(dict(video_id=row['video_id'], references=row['events'],
                    segments=decode_actions(logits[i, :lengths[i]], presence[i], cached[row['video_id']][2], classes)))
        return evaluate_detections(predictions), predictions
    protected = Path('proposal/action_detection_results/detector.pt')
    original_sha = sha(protected) if protected.exists() else None
    args.output.mkdir(parents=True)
    report = dict(diagnostic_only=True, completed=False, seed=6116, batch_size=8, threads=1,
        training_split='train', evaluation_split='same_16_train_videos', validation_read=False, test_read=False,
        selection='distinct groups; duration 1.5–20s; all events >=0.5s; at least one interior event',
        success_criterion='boundary_precision >= 0.9 and boundary_recall >= 0.9',
        videos=[dict(video_id=r['video_id'], annotation_sha256=r['annotation_sha256']) for r in rows],
        source_sha256=sha(Path(__file__).with_name('action_detection.py')), variants={})
    for name, objective in [('reference_loss', reference_loss), ('balanced_loss', detection_loss)]:
        # Reset both model initialization and minibatch permutation RNG for a fair comparison.
        torch.manual_seed(6116)
        model = ActionDetector()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        history, best, best_epoch, success = [], -1., 0, False
        for epoch in range(args.epochs + 1):
            if epoch:
                model.train()
                order = torch.randperm(16).tolist()
                for offset in range(0, 16, 8):
                    part = [rows[i] for i in order[offset:offset + 8]]
                    x, y, lengths, mask = batch(part)
                    logits, presence = model(x, lengths, mask)
                    loss = objective(logits, presence, y, mask)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            if epoch % 10 and epoch != args.epochs:
                continue
            metrics, predictions = evaluate(model)
            history.append(dict(epoch=epoch, train_subset=metrics))
            print(json.dumps(dict(variant=name, **history[-1])), flush=True)
            if metrics['boundary_f1'] > best:
                best, best_epoch = metrics['boundary_f1'], epoch
                torch.save(dict(schema=SCHEMA, state_dict=model.state_dict(), classes=classes,
                    rate=4., threshold=.5, smoke=True, diagnostic_only=True, epoch=epoch, loss=name),
                    args.output / (name + '.pt'))
                (args.output / (name + '_predictions.json')).write_text(json.dumps(predictions, indent=2), encoding='utf-8')
            success = metrics['boundary_precision'] >= .9 and metrics['boundary_recall'] >= .9
            if success:
                break
        report['variants'][name] = dict(history=history, best_epoch=best_epoch, best_boundary_f1=best,
            epochs_completed=epoch, fitting_criterion_met=success)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['completed'] = True
    report['historical_checkpoint_unchanged'] = original_sha == (sha(protected) if protected.exists() else None)
    if not report['historical_checkpoint_unchanged']:
        raise RuntimeError('Historical checkpoint changed during diagnostic run')
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
