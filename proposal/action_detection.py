"""Automatic multi-label temporal detection using shared dense ResNet features.

No animal/action query is required. Development reads Train/Valid only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from animalact2.phase1_5_1.common import PackedBiGRU
from .boundary_grounding import dense_features, targets

SCHEMA = "automatic-actions-v1"


class ActionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = PackedBiGRU(128)
        self.boundaries = nn.Linear(256, 119 * 3)
        self.presence = nn.Linear(256, 119)

    def forward(self, features, lengths, mask):
        encoded = self.temporal(features, lengths, mask)
        pooled = (encoded * mask[:, :, None]).sum(1) / lengths.to(encoded.device)[:, None]
        return self.boundaries(encoded).reshape(*encoded.shape[:2], 119, 3), self.presence(pooled)


def action_targets(events, edges):
    result = torch.zeros(len(edges) - 1, 119, 3)
    for class_id in sorted({event['class_id'] for event in events}):
        if not 0 <= class_id < 119:
            raise ValueError("Invalid action class ID")
        result[:, class_id] = targets([e['interval'] for e in events if e['class_id'] == class_id], edges)
    return result


def detection_loss(logits, presence, target, mask):
    present = target[:, :, :, 0].masked_fill(~mask[:, :, None], 0).amax(1).gt(0)
    valid = mask[:, :, None].expand_as(target[:, :, :, 0])
    temporal = valid & present[:, None, :]
    absent = valid & ~present[:, None, :]
    def balanced_bce(prediction, truth):
        positive = -(truth * nn.functional.logsigmoid(prediction)).sum() / truth.sum().clamp_min(1)
        negative = -((1 - truth) * nn.functional.logsigmoid(-prediction)).sum() / (1 - truth).sum().clamp_min(1)
        return positive + negative
    channels = []
    for channel in range(3):
        prediction = logits[:, :, :, channel]
        truth = target[:, :, :, channel]
        # Present-class temporal background gets its own weight, independent of 119-class sparsity.
        temporal_loss = balanced_bce(prediction[temporal], truth[temporal])
        absent_logits = prediction[absent]
        absent_loss = nn.functional.softplus(absent_logits).sum() / max(absent_logits.numel(), 1)
        channels.append(temporal_loss + absent_loss)
    return torch.stack(channels).mean() + balanced_bce(presence, present.float())


def interval_iou(first, second):
    overlap = max(0., min(first[1], second[1]) - max(first[0], second[0]))
    return overlap / max(first[1] - first[0] + second[1] - second[0] - overlap, 1e-8)


def decode_actions(logits, presence, edges, classes, threshold=.5):
    scores = torch.as_tensor(logits).detach().sigmoid().cpu().numpy()
    present = torch.as_tensor(presence).detach().sigmoid().cpu().numpy()
    edges = np.asarray(edges, dtype=float)
    if (len(edges) < 2 or len(classes) != 119 or scores.shape != (len(edges) - 1, 119, 3)
            or present.shape != (119,) or not np.isfinite(edges).all()
            or not np.isfinite(scores).all() or not np.isfinite(present).all() or np.any(np.diff(edges) <= 0)
            or not 0 < threshold < 1):
        raise ValueError("Invalid action prediction geometry or threshold")
    segments = []
    for cid in np.flatnonzero(present >= threshold):
        activity, start, end = scores[:, cid].T
        # Start/end peaks independently generate candidates anywhere in the video.
        starts = np.flatnonzero((start >= threshold) & (start >= np.r_[-1., start[:-1]]) & (start > np.r_[start[1:], -1.]))
        ends = np.flatnonzero((end >= threshold) & (end > np.r_[-1., end[:-1]]) & (end >= np.r_[end[1:], -1.]))
        if not len(starts) or not len(ends):
            continue
        begin, finish = np.meshgrid(starts, ends, indexing='ij')
        cumulative = np.r_[0., np.cumsum(activity * np.diff(edges))]
        means = (cumulative[finish + 1] - cumulative[begin]) / np.maximum(edges[finish + 1] - edges[begin], 1e-8)
        means = np.clip(means, 0, 1)
        valid = (finish >= begin) & (means >= threshold)
        confidence = means * np.sqrt(start[begin] * end[finish]) * present[cid]
        candidates = [(float(confidence[a, b]), [float(edges[begin[a, b]]), float(edges[finish[a, b] + 1])])
                      for a, b in zip(*np.where(valid))]
        chosen = []
        for score, span in sorted(candidates, key=lambda row: (-row[0], row[1][0], row[1][1])):
            if any(max(0., min(span[1], previous[1]) - max(span[0], previous[0]))
                   > .5 * min(span[1] - span[0], previous[1] - previous[0]) for previous in chosen):
                continue
            chosen.append(span)
            segments.append(dict(class_id=int(cid), action=classes[cid], start=span[0], end=span[1], score=score))
    return sorted(segments, key=lambda row: (row['start'], row['end'], row['class_id']))


def load_records(root, splits=('train', 'valid')):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    entries = sorted(json.loads((root / 'action_classes.json').read_text(encoding='utf-8'))['classes'], key=lambda row: row['class_id'])
    if [row['class_id'] for row in entries] != list(range(119)):
        raise ValueError("Invalid action vocabulary")
    classes = [row['name'] for row in entries]
    ids = {name: cid for cid, name in enumerate(classes)}
    records = {split: [] for split in splits}
    for row in manifest['samples']:
        if row['split'] not in records:
            continue
        raw = (root / row['annotation_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['annotation_sha256']:
            raise ValueError("Annotation checksum mismatch")
        annotation = json.loads(raw)
        grouped = {}
        for animal in annotation['annotations']:
            for action in animal['actions']:
                span = [float(action['start']), min(float(action['end']), row['duration'])]
                if not np.isfinite(span).all() or not 0 <= span[0] < span[1]:
                    raise ValueError("Invalid annotation interval")
                grouped.setdefault(ids[action['action']], []).append(span)
        events = []
        for cid, spans in grouped.items():
            merged = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end])
            events.extend(dict(class_id=cid, interval=span) for span in merged)
        records[row['split']].append(dict(row, events=events))
    return records, classes


def evaluate_detections(records):
    counts = dict(predictions=0, references=0, iou_hits=0, boundary_hits=0)
    query_ious = []
    for item in records:
        expected = item['references']
        predicted = sorted(item['segments'], key=lambda row: -row['score'])
        counts['predictions'] += len(predicted)
        counts['references'] += len(expected)
        for cid in sorted({r['class_id'] for r in expected}):
            references = [r['interval'] for r in expected if r['class_id'] == cid]
            candidates = [s for s in predicted if s['class_id'] == cid]
            query_ious.append(max((interval_iou([candidates[0]['start'], candidates[0]['end']], r)
                                   for r in references), default=0.) if candidates else 0.)
        for kind in ('iou', 'boundary'):
            used = set()
            for segment in predicted:
                span = [segment['start'], segment['end']]
                choices = [(i, reference) for i, reference in enumerate(expected)
                           if i not in used and reference['class_id'] == segment['class_id']]
                if kind == 'iou':
                    choices = [(i, r) for i, r in choices if interval_iou(span, r['interval']) >= .5]
                else:
                    choices = [(i, r) for i, r in choices if max(abs(np.asarray(span) - r['interval'])) <= .5]
                if choices:
                    index, _ = max(choices, key=lambda pair: interval_iou(span, pair[1]['interval']))
                    used.add(index)
                    counts[kind + '_hits'] += 1
    result = dict(counts)
    result.update(positive_action_queries=len(query_ious),
                  miou=float(np.mean(query_ious)) if query_ious else None,
                  r1_iou_05=float(np.mean(np.asarray(query_ious) >= .5)) if query_ious else None,
                  r1_iou_07=float(np.mean(np.asarray(query_ious) >= .7)) if query_ious else None)
    for kind in ('iou', 'boundary'):
        hits = counts[kind + '_hits']
        result[kind + '_precision'] = hits / max(counts['predictions'], 1)
        result[kind + '_recall'] = hits / max(counts['references'], 1)
        result[kind + '_f1'] = 2 * hits / max(counts['predictions'] + counts['references'], 1)
    return result


@torch.inference_mode()
def predict_actions(predictor, video, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != SCHEMA or checkpoint.get('smoke', True):
        raise ValueError("A full automatic action detector is required; smoke weights cannot be deployed")
    from animalact2.dashboard.registry import classes as vocabulary
    if checkpoint['classes'] != [row['name'] for row in vocabulary()]:
        raise ValueError("Detector vocabulary mismatch")
    value = dense_features(predictor, video, checkpoint['rate'])
    model = ActionDetector().to(predictor.device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    features = value['features'][None].to(predictor.device)
    logits, presence = model(features, torch.tensor([features.shape[1]]), torch.ones(features.shape[:2], dtype=torch.bool, device=predictor.device))
    return dict(status='ready', model=SCHEMA, experimental=True, duration=value['duration'],
                segments=decode_actions(logits[0], presence[0], value['edges'], checkpoint['classes'], checkpoint['threshold']),
                max_time_bin_seconds=float(np.diff(value['edges']).max()), calibrated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('proposal/boundary_results/features'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error('Training settings must be positive')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; prior runs are protected')
    torch.set_num_threads(args.threads)
    torch.manual_seed(6116)
    records, classes = load_records(args.data.resolve())
    if args.mode == 'smoke':
        records = {split: rows[:2] for split, rows in records.items()}
    cached = {}
    for rows in records.values():
        for row in rows:
            value = torch.load(args.cache / (row['video_id'] + '.pt'), map_location='cpu', weights_only=True)
            with (args.data / row['video_path']).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != value['source_sha256']:
                    raise ValueError('Video feature cache mismatch')
            if value['rate'] != 4. or abs(value['duration'] - row['duration']) > max(1 / row['fps'], .05):
                raise ValueError('Dense feature sampling or duration mismatch')
            cached[row['video_id']] = (value['features'], action_targets(row['events'], value['edges']), value['edges'])
    def batch(rows):
        items = [cached[row['video_id']] for row in rows]
        x = pad_sequence([v[0] for v in items], batch_first=True)
        y = pad_sequence([v[1] for v in items], batch_first=True)
        lengths = torch.tensor([len(v[0]) for v in items])
        mask = torch.arange(x.shape[1])[None] < lengths[:, None]
        return x.to(device), y.to(device), lengths, mask.to(device)
    args.output.mkdir(parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ActionDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    best, stale, history = -1., 0, []
    for epoch in range(1, (1 if args.mode == 'smoke' else args.epochs) + 1):
        model.train()
        order = torch.randperm(len(records['train'])).tolist()
        losses = []
        for offset in range(0, len(order), args.batch_size):
            rows = [records['train'][i] for i in order[offset:offset + args.batch_size]]
            x, y, lengths, mask = batch(rows)
            logits, presence = model(x, lengths, mask)
            loss = detection_loss(logits, presence, y, mask)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        validation = []
        with torch.inference_mode():
            for offset in range(0, len(records['valid']), args.batch_size):
                rows = records['valid'][offset:offset + args.batch_size]
                x, y, lengths, mask = batch(rows)
                logits, presence = model(x, lengths, mask)
                for i, row in enumerate(rows):
                    edges = cached[row['video_id']][2]
                    validation.append(dict(video_id=row['video_id'], references=row['events'],
                        segments=decode_actions(logits[i, :lengths[i]], presence[i], edges, classes)))
        metrics = evaluate_detections(validation)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics))
        print(json.dumps(history[-1]), flush=True)
        if metrics['boundary_f1'] > best:
            best, stale = metrics['boundary_f1'], 0
            torch.save(dict(schema=SCHEMA, state_dict=model.state_dict(), classes=classes, rate=4.,
                epoch=epoch, threshold=.5, smoke=args.mode == 'smoke', validation=metrics,
                selection_split='valid'), args.output / 'detector.pt')
            (args.output / 'validation_records.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
        else:
            stale += 1
        (args.output / 'results.json').write_text(json.dumps(dict(history=history,
            schema=SCHEMA, completed=False, smoke=args.mode == 'smoke', test_read=False,
            selection_metric='boundary_f1', seed=6116, batch_size=args.batch_size,
            threads=args.threads, patience=args.patience), indent=2), encoding='utf-8')
        if stale >= args.patience:
            break
    path = args.output / 'results.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['completed'] = True
    report['enabled_for_inference'] = False
    report['epochs_completed'] = len(history)
    report['negative_policy'] = 'Unannotated actions are treated as background; annotation completeness is assumed.'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
