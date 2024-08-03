from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        with self.writer.as_default():
            self.writer.add_scalar(tag, value, step=step)
            self.writer.flush()
