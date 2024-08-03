from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, global_step=step)
        self.writer.flush()
        
    def add_im(self, tag, im, step):
        self.writer.add_image(tag, im, global_step=step)