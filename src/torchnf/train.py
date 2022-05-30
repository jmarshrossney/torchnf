"""    @property
    def optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @property
    def scheduler(
        self,
    ) -> Union[
        torch.optim.lr_scheduler._LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ]:
        raise NotImplementedError

    def train(self, batch_size: int, n_steps: int) -> None:

        for step in range(n_steps):

            _, log_weights = self.forward(batch_size)
            loss = log_weights.mean().neg()
            self.global_step += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # requires loss if ReduceLROnPlateau
"""
