from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
from enum import Enum

import torchvision.models as models

import os
import json
import socket
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from PIL import Image


# --- This code is an XPU translation of https://github.com/argonne-lcf/dl_scaling/blob/main/resnet50/resnet_ddp.py, translated by Claude ---
# --- This is a second translation, by ChatGPT, to remove validation requirements, allowing me to use places365 dataset, which does not include val set ---
# --- Third Translation, from claude, to make places365 dataset work. ---


# Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    # Aurora: local rank comes from the launcher's PALS_LOCAL_RANKID,
    # not computed manually as rank % N
    local_rank = int(os.environ.get('PALS_LOCAL_RANKID', 0))

    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    # Aurora: append the high-speed network (HSN) suffix so nodes reach
    # each other over the correct fabric
    os.environ["MASTER_ADDR"] = f"{master_addr}.hsn.cm.aurora.alcf.anl.gov"
    os.environ["MASTER_PORT"] = str(2345)
    print("DDP: I am worker %s of %s. My local rank is %s" %(rank, size, local_rank))
    # MPI.COMM_WORLD.Barrier()

except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)

# ---------------------------------------------------------------------------
# Dataset handling: support both standard ImageFolder layouts (ImageNet,
# Imagenette) and hierarchical layouts (Places365) via auto-detection.
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp')
CACHE_FILENAME = '.dataset_index.json'


def _is_imagefolder_layout(root):
    """
    True if `root` looks like a standard ImageFolder tree: the immediate
    child directories directly contain image files (train/class/img.jpg).
    False if classes are nested deeper (Places365: train/a/arena/hockey/...).
    We sample the first-level subdirs and check whether any of THEM contain
    images directly. If they contain only subdirs, it's hierarchical.
    """
    subdirs = [d for d in os.scandir(root) if d.is_dir()]
    if not subdirs:
        return False
    for d in subdirs:
        for entry in os.scandir(d.path):
            if entry.is_file() and entry.name.lower().endswith(IMG_EXTENSIONS):
                return True   # a first-level dir holds images -> ImageFolder
    return False              # first-level dirs hold only subdirs -> recursive


class RecursiveImageFolder(Dataset):
    """
    Treats every leaf directory (a directory that directly contains image
    files) as its own class. The class NAME is the path relative to the
    dataset root, so Places365's arena/hockey and arena/rodeo become two
    distinct classes with no renaming or file movement.

    Uses a JSON cache (.dataset_index.json at the root) to avoid rescanning
    ~1.8M files on every run.

    Image paths are stored in the cache RELATIVE to the dataset root, so the
    same cache works whether the dataset is accessed via raw Lustre
    (/flare/.../data_256) or via the Copper mount
    (/tmp/USER/copper_mount/lus/flare/.../data_256). Absolute paths are
    reconstructed at read time by joining with self.root. This keeps the
    Copper-vs-Lustre comparison clean: both conditions load the identical
    manifest instead of rebuilding it because the root path differs.
    """

    CACHE_VERSION = 2  # bump if the on-disk format changes

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []          # list of (relative_image_path, class_index)
        self.classes = []          # list of relative class names
        cache_status = self._load_or_build_index()
        self.cache_status = cache_status

    def _cache_path(self):
        return os.path.join(self.root, CACHE_FILENAME)

    def _load_or_build_index(self):
        cache = self._cache_path()
        # Try cache first; rebuild only on a genuine problem (missing,
        # unreadable, or wrong format version). We deliberately do NOT
        # validate the absolute root path, because the cache is portable:
        # raw-Lustre and Copper-mount runs share it.
        if os.path.isfile(cache):
            try:
                with open(cache, 'r') as f:
                    data = json.load(f)
                if data.get('version') == self.CACHE_VERSION:
                    self.classes = data['classes']
                    self.samples = [(p, c) for p, c in data['samples']]
                    return 'loaded'
            except Exception:
                pass  # fall through to rebuild
        self._build_index()
        self._save_index()
        return 'created'

    def _build_index(self):
        class_to_idx = {}
        samples = []
        # Walk the tree; any directory that directly holds image files is a class.
        for dirpath, dirnames, filenames in os.walk(self.root):
            imgs = [f for f in filenames if f.lower().endswith(IMG_EXTENSIONS)]
            if not imgs:
                continue
            rel_dir = os.path.relpath(dirpath, self.root)
            if rel_dir == '.':
                continue
            if rel_dir not in class_to_idx:
                class_to_idx[rel_dir] = len(class_to_idx)
            idx = class_to_idx[rel_dir]
            for fn in imgs:
                # store path RELATIVE to root (portable across raw/mount)
                samples.append((os.path.join(rel_dir, fn), idx))
        # Stable ordering: sort classes by name, remap indices accordingly.
        ordered = sorted(class_to_idx.keys())
        remap = {class_to_idx[name]: i for i, name in enumerate(ordered)}
        self.classes = ordered
        self.samples = [(p, remap[c]) for (p, c) in samples]

    def _save_index(self):
        data = {
            'version': self.CACHE_VERSION,
            'classes': self.classes,
            'samples': self.samples,   # relative paths
        }
        tmp = self._cache_path() + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, self._cache_path())
        except Exception:
            # Cache is an optimization; never fail the run if it can't be written.
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, target = self.samples[index]
        # reconstruct the absolute path against THIS run's root (raw or mount)
        path = os.path.join(self.root, rel_path)
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def dataset_needs_cache(traindir):
    """
    True only when the dataset is hierarchical (Recursive) AND no valid cache
    exists yet -- i.e. an expensive scan is about to happen that should be done
    once by rank 0, not simultaneously by every rank. ImageFolder layouts and
    already-cached recursive datasets return False (cheap, no coordination).
    The cache is portable (relative paths), so we validate only by format
    version, not by absolute root path.
    """
    if _is_imagefolder_layout(traindir):
        return False
    cache = os.path.join(traindir, CACHE_FILENAME)
    if os.path.isfile(cache):
        try:
            with open(cache, 'r') as f:
                data = json.load(f)
            if data.get('version') == RecursiveImageFolder.CACHE_VERSION:
                return False   # valid cache already present
        except Exception:
            pass
    return True


def build_train_dataset(traindir, transform):
    """
    Auto-detect layout and return (dataset, format_str, cache_status).
    cache_status is None for ImageFolder (no cache involved).
    """
    if _is_imagefolder_layout(traindir):
        ds = datasets.ImageFolder(traindir, transform)
        return ds, 'ImageFolder', None
    ds = RecursiveImageFolder(traindir, transform)
    return ds, 'Recursive', ds.cache_status


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    print("start training")
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i==0:
            first_batch_time = time.time() - end
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0 and rank==0:
        #     progress.display(i + 1)
        if i >= args.steps and args.steps > 0:
            if rank==0:
                print('Throughput: {:.3f} images/s,'.format((args.steps-1) * args.batch_size * size / (batch_time.sum-first_batch_time)),
                    'Batch size: {},'.format(args.batch_size),
                    'Num of GPUs: {},'.format(size),
                    'Total time: {:.3f} s,'.format(batch_time.sum),
                    'Average batch time: {:.3f} s,'.format(batch_time.avg),
                    'First batch time: {:.3f} s'.format(first_batch_time))
            return


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if rank==0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('data', metavar='DIR', nargs='?', default='../../../datasets/imagenet/images',
                    help='path to dataset (default: imagenet)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPU (XPU) training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--steps', default=100, type=int,
                    metavar='N', help='number of iterations to measure throughput, -1 for disable')

    args = parser.parse_args()
    use_xpu = not args.no_cuda and torch.xpu.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # DDP: Initialize library.
    if with_ddp:
        torch.distributed.init_process_group(
            backend='xccl', init_method='env://',
            world_size=size, rank=rank)

    torch.manual_seed(args.seed)

    # DDP: Pin GPU (tile) to local rank.
    if use_xpu:
        torch.xpu.set_device(int(local_rank))

    if use_xpu:
        device = torch.device("xpu")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_xpu:
        xpu_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(xpu_kwargs)

    # Data loading code
    dataset_format = 'Dummy'
    cache_status = None
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # Datasets like ImageNet/Imagenette nest images under a `train/`
        # subdir; Places365's data_256 does not. Use `<data>/train` only if it
        # actually exists, otherwise use `<data>` directly.
        candidate = os.path.join(args.data, 'train')
        traindir = candidate if os.path.isdir(candidate) else args.data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        # Auto-detect ImageFolder (ImageNet/Imagenette) vs hierarchical
        # (Places365) layout. Only rank 0 builds the cache to avoid all
        # ranks writing .dataset_index.json at once; other ranks wait, then
        # load it.
        if with_ddp and dataset_needs_cache(traindir):
            if rank == 0:
                train_dataset, dataset_format, cache_status = build_train_dataset(traindir, train_transform)
            dist.barrier()
            if rank != 0:
                train_dataset, dataset_format, cache_status = build_train_dataset(traindir, train_transform)
        else:
            train_dataset, dataset_format, cache_status = build_train_dataset(traindir, train_transform)

    # shuffle=False: epoch 2 must re-read epoch 1's exact files so Copper's
    # cache has something to hit. This is a caching benchmark, not a
    # convergence run -- ordering doesn't matter for throughput.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=size, rank=rank, shuffle=False, seed=args.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **train_kwargs)

    if rank == 0:
        images_per_rank = len(train_dataset) // size
        batches_per_rank = images_per_rank // args.batch_size
        print(f"Dataset format: {dataset_format}")
        if not args.dummy:
            num_classes = len(getattr(train_dataset, 'classes', []) or [])
            print(f"Classes: {num_classes}")
        print(f"Training images: {len(train_dataset):,}")
        print(f"Images per rank: {images_per_rank:,}")
        print(f"Batches per rank: {batches_per_rank}")
        if cache_status is not None:
            print(f"Dataset cache: {cache_status}")
        # Early warning: if each rank can't reach --steps batches, the
        # throughput print will never fire (the Imagenette-at-scale failure).
        if args.steps > 0 and batches_per_rank <= args.steps:
            print(f"WARNING: batches_per_rank ({batches_per_rank}) <= steps "
                  f"({args.steps}); throughput may not print. Use a larger "
                  f"dataset, smaller batch size, or fewer steps.")

    model = models.resnet50()
    # DDP: move model to device BEFORE wrapping in DDP (Aurora: required
    # to avoid hangs when using xccl backend)
    model = model.to(device)
    # Wrap the model in DDP:
    if with_ddp:
        model = DDP(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr*size,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # DDP: set epoch on the sampler for proper shuffling across epochs
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # test(model, device, val_loader)
    print(time.time()-t0)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.xpu.is_available():
            device = torch.device("xpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
