from training import *

trainer = train()


metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")[["epoch", "train_loss_epoch", "val_loss"]]
metrics.set_index("epoch", inplace=True)

sns.relplot(data=metrics, kind="line", height=5, aspect=1.5)
plt.grid()