"""Episode-preserving truncated BPTT with exportable per-epoch checkpoints."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .common import read, write, append, sha
from .networks import Actor, initial_spec, policy


def dataset(directories):
    examples = []
    for directory in directories:
        directory = Path(directory)
        manifest = read(directory/'manifest.json')
        for row in manifest['episodes']:
            if not row['accepted']:
                continue
            path = directory/row['file']
            if sha(path) != row['sha256']:
                raise ValueError(f'Dataset checksum mismatch: {path}')
            with np.load(path) as arrays:
                examples.append((arrays['obs'].astype(np.float64), arrays['action'].astype(np.float64)))
    if not examples:
        raise ValueError('No accepted demonstrations')
    return examples


def offline(spec, episodes):
    square = np.zeros(3)
    count = 0
    for observations, actions in episodes:
        controller = policy(spec)
        for observation, target in zip(observations, actions):
            prediction = controller.act(observation)
            square += (prediction-target)**2
            count += 1
    return np.sqrt(square/count).tolist()


def fit(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    examples = dataset(args.data)
    validation = dataset([args.validation])
    spec = read(args.initial) if args.initial else initial_spec(args.kind, args.seed, args.hidden)
    actor = Actor(spec)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    start_epoch = 0
    seen = 0
    checkpoint = out/'checkpoint.pt'
    if checkpoint.exists():
        stored = torch.load(checkpoint, weights_only=False)
        actor.load_state_dict(stored['actor'])
        optimizer.load_state_dict(stored['optimizer'])
        rng.bit_generator.state = stored['rng']
        start_epoch = stored['epoch']
        seen = stored['samples_seen']
    elif not args.initial:
        # Supervised bias initialization uses only the training labels.
        target = np.mean(np.concatenate([y for _, y in examples]), axis=0)
        with torch.no_grad():
            actor.bias[actor.out] = torch.tensor(np.arctanh(np.clip(target, -.95, .95)))
    write(out/'arguments.json', dict(kind=args.kind, seed=args.seed, hidden=args.hidden,
          epochs=args.epochs, window_steps=args.window, batch=args.batch, lr=args.lr,
          data=[str(p) for p in args.data], validation=str(args.validation),
          initial=str(args.initial) if args.initial else None,
          demonstration_samples=sum(len(x) for x, _ in examples)))
    start = time.monotonic()
    if start_epoch == 0:
        write(out/'policy_epoch000.json', actor.export())
    for epoch in range(start_epoch, args.epochs):
        order = rng.permutation(len(examples))
        total_loss = 0.
        updates = 0
        for offset in range(0, len(order), args.batch):
            batch = [examples[i] for i in order[offset:offset+args.batch]]
            length = max(len(x) for x, _ in batch)
            x = np.zeros((length, len(batch), 11))
            y = np.zeros((length, len(batch), 3))
            valid = np.zeros((length, len(batch)))
            for b, (observations, actions) in enumerate(batch):
                x[:len(observations), b] = observations
                y[:len(actions), b] = actions
                valid[:len(actions), b] = 1.
            x, y, valid = (torch.from_numpy(a) for a in (x, y, valid))
            state = actor.initial_state(len(batch))
            previous_prediction = None
            for begin in range(0, length, args.window):
                state = state.detach()
                predictions = []
                end = min(begin+args.window, length)
                for step in range(begin, end):
                    mean, state = actor(x[step], state)
                    predictions.append(mean.tanh())
                prediction = torch.stack(predictions)
                live = valid[begin:end]
                denominator = live.sum().clamp_min(1)*3
                error = (prediction-y[begin:end]).square()
                loss = (error*live[..., None]).sum()/denominator
                if end-begin > 1:
                    differences = (prediction[1:]-prediction[:-1])-(y[begin+1:end]-y[begin:end-1])
                    loss = loss + .05*(differences.square()*live[1:, :, None]).sum()/denominator
                if previous_prediction is not None:
                    continuity = (prediction[0]-previous_prediction)-(y[begin]-y[begin-1])
                    loss = loss + .05*(continuity.square()*live[0, :, None]).sum()/denominator
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                optimizer.step()
                previous_prediction = prediction[-1].detach()
                total_loss += float(loss.detach())
                updates += 1
                seen += int(live.sum())
        epoch_number = epoch+1
        exported = actor.export()
        write(out/f'policy_epoch{epoch_number:03d}.json', exported)
        record = dict(epoch=epoch_number, loss=total_loss/max(1, updates),
                      samples_seen=seen, wall_seconds=time.monotonic()-start,
                      status='training', kind=args.kind, seed=args.seed)
        if epoch_number % args.validate_every == 0 or epoch_number == args.epochs:
            record['offline_rmse'] = offline(exported, validation)
        temporary = out/'checkpoint.tmp.pt'
        torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch_number, samples_seen=seen, rng=rng.bit_generator.state), temporary)
        temporary.replace(checkpoint)
        append(out/'training.jsonl', record)
        write(out/'status.json', record)
        print(record, flush=True)
    write(out/'last_policy.json', actor.export())
    write(out/'completed.json', dict(status='completed', epochs=args.epochs, samples_seen=seen))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['snn', 'ann'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--initial')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--window', type=int, default=500)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--validate-every', type=int, default=5)
    fit(parser.parse_args())


if __name__ == '__main__':
    main()
