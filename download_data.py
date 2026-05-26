#!/usr/bin/env python

import inspect
import os
import tarfile
import urllib.request


def here(f):
    me = inspect.getsourcefile(here)
    return os.path.join(os.path.dirname(os.path.abspath(me)), f)


def download_extract(urlbase, name, into):
    print("Downloading " + name)
    os.makedirs(into, exist_ok=True)
    fname = os.path.join(into, name)
    if not os.path.exists(fname):
        urllib.request.urlretrieve(os.path.join(urlbase, name), fname)
    print("Extracting...")
    with tarfile.open(fname) as f:
        f.extractall(path=into)


if __name__ == '__main__':
    baseurl = 'https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/'
    datadir = here('data')

    # First, download the Tosato datasets.
    download_extract(baseurl, 'CAVIARShoppingCenterFullOccl.tar.bz2', into=datadir)
    download_extract(baseurl, 'CAVIARShoppingCenterFull.tar.bz2', into=datadir)
    download_extract(baseurl, 'HIIT6HeadPose.tar.bz2', into=datadir)
    download_extract(baseurl, 'HOC.tar.bz2', into=datadir)
    download_extract(baseurl, 'HOCoffee.tar.bz2', into=datadir)
    download_extract(baseurl, 'IHDPHeadPose.tar.bz2', into=datadir)
    download_extract(baseurl, 'QMULPoseHeads.tar.bz2', into=datadir)

    # Second, Benfold's TownCentre dataset.
    download_extract(baseurl, 'TownCentreHeadImages.tar.bz2', into=datadir)

    print("Done.")
