import matplotlib.pyplot as plt
import os

def plot_training_history(history, ticker):
    # Extract metrics
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    mae = history.history.get('mae', [])
    val_mae = history.history.get('val_mae', [])
    epochs_range = range(1, len(loss) + 1)

    # Create the plot
    plt.figure(figsize=(10, 4))

    # Plot MAE
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mae, label='Training MAE')
    plt.plot(epochs_range, val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training vs Validation MAE')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)
    plot_path = os.path.join(dir_path, f"training_{ticker}.png")
    plt.savefig(plot_path)
    plt.close()

def plot_balance(df_back, balance, ticker):
    plt.figure(figsize=(12, 6))  # Optional: make it more readable
    plt.plot(df_back['Date'], balance[:])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.xticks(rotation=45)  # Optional: rotate for better readability
    plt.tight_layout()  # Optional: avoids label cut-off

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)
    plot_path = os.path.join(dir_path, f"balance_{ticker}.png")
    plt.savefig(plot_path)
    plt.close()

def plot_metrics(results, ticker):
    metrics = ['return', 'trades', 'no_trades', 'hit_rate']
    for metric in metrics:
        to_show = []
        for r in results:
            to_show.append(r[metric])
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, 100), to_show, marker='o')
        plt.title(f'{metric.capitalize()} vs Quantile Threshold')
        plt.xlabel('Quantile (%)')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.tight_layout()

        dir_path = f"./Infos/{ticker}"
        os.makedirs(dir_path, exist_ok=True)
        plot_path = os.path.join(dir_path, f"threshold_{metric}_{ticker}.png")
        plt.savefig(plot_path)
        plt.close()