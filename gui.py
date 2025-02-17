import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from train import train_model


def start_training():
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()


def run_training():
    try:
        epochs = int(epochs_entry.get())
    except ValueError:
        epochs = 1000
        epochs_entry.delete(0, tk.END)
        epochs_entry.insert(0, str(epochs))

    try:
        lr = float(lr_entry.get())
    except ValueError:
        lr = 0.01
        lr_entry.delete(0, tk.END)
        lr_entry.insert(0, str(lr))

    try:
        hidden_layers = int(hidden_layers_entry.get())
    except ValueError:
        hidden_layers = 1
        hidden_layers_entry.delete(0, tk.END)
        hidden_layers_entry.insert(0, str(hidden_layers))

    try:
        hidden_dim = int(hidden_dim_entry.get())
    except ValueError:
        hidden_dim = 10
        hidden_dim_entry.delete(0, tk.END)
        hidden_dim_entry.insert(0, str(hidden_dim))

    func_choice = func_var.get()

    model, x_vals, y_vals, preds, loss_history, func_name = train_model(
        num_epochs=epochs,
        learning_rate=lr,
        num_hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        func_choice=func_choice
    )

    root.after(0, update_plots, x_vals, y_vals, preds, loss_history, func_name)


def update_plots(x_vals, y_vals, preds, loss_history, func_name):
    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(loss_history, label="Loss")
    axes[0].set_title("Зміна Loss")
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(x_vals, y_vals, label="Real", color="blue", s=20)
    axes[1].scatter(x_vals, preds, label="Predicted", color="red", s=20)
    axes[1].set_title(f"Апроксимація {func_name}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


root = tk.Tk()
root.geometry("850x500")

settings_frame = tk.Frame(root)
settings_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

tk.Label(settings_frame, text="Кількість епох:").grid(row=0, column=0, sticky="w")
epochs_entry = tk.Entry(settings_frame, width=10)
epochs_entry.insert(0, "1000")
epochs_entry.grid(row=0, column=1, padx=5, pady=2)

tk.Label(settings_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
lr_entry = tk.Entry(settings_frame, width=10)
lr_entry.insert(0, "0.01")
lr_entry.grid(row=1, column=1, padx=5, pady=2)

tk.Label(settings_frame, text="Приховані шари:").grid(row=2, column=0, sticky="w")
hidden_layers_entry = tk.Entry(settings_frame, width=10)
hidden_layers_entry.insert(0, "1")
hidden_layers_entry.grid(row=2, column=1, padx=5, pady=2)

tk.Label(settings_frame, text="Нейрони в шарі:").grid(row=3, column=0, sticky="w")
hidden_dim_entry = tk.Entry(settings_frame, width=10)
hidden_dim_entry.insert(0, "10")
hidden_dim_entry.grid(row=3, column=1, padx=5, pady=2)

tk.Label(settings_frame, text="Функція:").grid(row=4, column=0, sticky="w")
func_var = tk.StringVar(value="f(x)=x^2")
func_combo = ttk.Combobox(settings_frame, textvariable=func_var,
                          values=["f(x)=x^2", "f(x)=x^3+2x"], width=15)
func_combo.grid(row=4, column=1, padx=5, pady=2)

start_btn = tk.Button(settings_frame, text="Почати", command=start_training)
start_btn.grid(row=5, column=0, columnspan=2, pady=10)

exit_btn = tk.Button(settings_frame, text="Вихід", command=root.quit)
exit_btn.grid(row=6, column=0, columnspan=2, pady=5)

plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

root.mainloop()
