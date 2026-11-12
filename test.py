#!/usr/bin/env python3
"""Run CT-OBCR image prediction or labeled-list evaluation on CPU or CUDA."""

import argparse
import csv
import os
from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parse_args(argv=None, print_parameters=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=PACKAGE_DIR / 'checkpoint/final.pt')
    images = parser.add_mutually_exclusive_group()
    images.add_argument('--test-list', type=Path,
                        help='ordered image-path/label list; default: data/scan_test.txt')
    images.add_argument('--image', type=Path, help='predict one image')
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='root for relative paths inside the test list')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--batch-size', type=positive_int,
                        help='default: evaluation batch size saved in the checkpoint')
    parser.add_argument('--num-threads', type=positive_int, default=4)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--save-logits', action='store_true')
    if print_parameters:
        parser.add_argument('--params-only', action='store_true',
                            help='print parameter counts without evaluating images')
    return parser.parse_args(argv)


def resolve_device(requested, torch):
    if requested == 'cpu':
        return 'cpu'
    available = bool(torch.cuda.is_available())
    if requested == 'cuda' and not available:
        raise RuntimeError('CUDA is unavailable; use --device cpu or --device auto')
    return 'cuda' if available else 'cpu'


def read_test_list(path, data_root, num_classes):
    records, seen = [], set()
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.rsplit('\t', 1) if '\t' in line else line.rsplit(None, 1)
            if len(fields) != 2:
                raise ValueError('Expected image path and label at line %d' % number)
            try:
                label = int(fields[1])
            except ValueError:
                raise ValueError('Invalid integer label at line %d' % number)
            if not 0 <= label < num_classes:
                raise ValueError('Label outside [0, %d] at line %d' % (num_classes - 1, number))
            image = Path(fields[0]).expanduser()
            if not image.is_absolute():
                image = data_root / image
            image = image.resolve()
            if not image.is_file():
                raise FileNotFoundError('Image missing at line %d: %s' % (number, image))
            canonical = os.path.normcase(str(image))
            if canonical in seen:
                raise ValueError('Duplicate image at line %d: %s' % (number, image))
            seen.add(canonical)
            records.append((fields[0], image, label))
    if not records:
        raise ValueError('Test list is empty: ' + str(path))
    return records


def load_image(path, image_size, np, Image):
    bilinear = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
    with Image.open(path) as source:
        image = source.convert('RGB').resize((image_size, image_size), resample=bilinear)
        pixels = np.asarray(image, dtype=np.uint8)
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)
    tensor /= np.float32(255.0)
    tensor -= np.float32(0.5)
    tensor /= np.float32(0.5)
    return tensor


def display_parameter_counts(counts):
    print('Inference parameters (includes the local discriminator; excludes the training decoder):')
    learnable = int(counts['inference_learnable_before_freeze'])
    total = int(counts['inference_registered'])
    print('  Learnable before inference freezing: {:,} ({:.6f} M; {:.1f} M rounded)'.format(
        learnable, learnable / 1e6, learnable / 1e6))
    print('  Fixed parameters: {:,}'.format(int(counts['fixed_parameters'])))
    print('  Total registered inference parameters: {:,} ({:.6f} M)'.format(total, total / 1e6))
    print('  Training decoder excluded: {:,}'.format(int(counts['training_decoder_excluded'])))


def _run(argv=None, print_parameters=False):
    args = parse_args(argv, print_parameters=print_parameters)
    # Keep --help usable before installing the inference dependencies.
    import numpy as np
    from PIL import Image
    import torch
    from inference_runtime import load_model

    torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device, torch)
    wrapper, metadata = load_model(args.checkpoint, device=device)
    print('Device: %s; PyTorch: %s' % (device, torch.__version__))
    if print_parameters:
        display_parameter_counts(metadata['parameter_counts'])
        if args.params_only:
            return 0

    classes = int(metadata['num_classes'])
    image_size = int(metadata['image_size'])
    saved_batch = int(metadata['eval_batch_size'])
    if min(classes, image_size, saved_batch) < 1:
        raise ValueError('Invalid image, class, or batch setting in checkpoint')
    if args.image is not None:
        image = args.image.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError('Image not found: ' + str(image))
        if args.batch_size not in (None, 1):
            raise ValueError('--image uses a batch of one; omit --batch-size or set it to 1')
        records, batch_size = [(str(args.image), image, None)], 1
    else:
        test_list = args.test_list or PACKAGE_DIR / 'data/scan_test.txt'
        records = read_test_list(test_list.expanduser().resolve(),
                                 args.data_root.expanduser().resolve(), classes)
        batch_size = args.batch_size or saved_batch
    print('Images: %d; batch size: %d (saved: %d)' % (len(records), batch_size, saved_batch))
    if metadata.get('optimal'):
        print('NOTICE: predictions can depend on other images in the batch. '
              'Keep the original test-list order and saved batch size for comparison.', file=sys.stderr)
    if batch_size != saved_batch:
        print('NOTICE: batch grouping differs from the saved evaluation. '
              'This model can depend on batch composition.', file=sys.stderr)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / 'predictions.csv'
    logits_path = output / 'logits.npy'
    if predictions_path.exists() or (args.save_logits and logits_path.exists()):
        raise FileExistsError('Results already exist; choose another --output-dir')
    logits_parts = []
    correct = labeled = 0
    with predictions_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index', 'path', 'label', 'prediction', 'correct',
                         'max_probability', 'batch_id', 'position_in_batch'])
        for offset in range(0, len(records), batch_size):
            chunk = records[offset:offset + batch_size]
            images = np.stack([load_image(record[1], image_size, np, Image) for record in chunk])
            with torch.no_grad():
                logits = wrapper(torch.from_numpy(images).to(device)).detach().cpu().numpy()
            if logits.shape != (len(chunk), classes) or not np.isfinite(logits).all():
                raise ValueError('Model returned nonfinite or incorrectly shaped logits')
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            predicted = logits.argmax(axis=1)
            for position, (record, prediction) in enumerate(zip(chunk, predicted)):
                listed_path, _, label = record
                prediction = int(prediction)
                ok = None if label is None else int(label == prediction)
                confidence = float(probabilities[position, prediction])
                if ok is not None:
                    labeled += 1
                    correct += ok
                writer.writerow([offset + position, listed_path, label, prediction, ok,
                                 confidence, offset // batch_size, position])
                if args.image is not None:
                    print('Prediction: %d; confidence: %.6f' % (prediction, confidence))
            if args.save_logits:
                logits_parts.append(logits)
            wrapper.assert_frozen_state()
            print('Processed %d/%d images' % (offset + len(chunk), len(records)), file=sys.stderr)
    if args.save_logits:
        np.save(str(logits_path), np.concatenate(logits_parts, axis=0), allow_pickle=False)
    if labeled:
        print('Accuracy: %d/%d (%.6f%%)' % (correct, labeled, 100.0 * correct / labeled))
    print('Predictions: ' + str(predictions_path))
    if args.save_logits:
        print('Logits: ' + str(logits_path))
    return 0


def main(argv=None, print_parameters=False):
    try:
        return _run(argv, print_parameters=print_parameters)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
