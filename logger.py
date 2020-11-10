# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import sys

import numpy as np
import scipy.misc
import tensorflow as tf


from io import BytesIO  # Python 3.x
from torch.utils.tensorboard import SummaryWriter



class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        for i, img in enumerate(images):
            # Create a Summary value
            self.writer.add_image(tag='%s/%d' % (tag, i), image=img, global_step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        self.writer.add_histogram(tag, values, global_step=step, bins='tensorflow', max_bins=bins)
        self.writer.flush()
