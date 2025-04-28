import climate_learn as cl
from data_util import get_data_module
import torch

def main():
    print("Loading data...")
    dm = get_data_module()

    print("Loading ClimaX model...")
    model = cl.load_forecasting_module(
        data_module=dm,
        architecture="climax",
        pretrained=False
    )

    trainer = cl.get_trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )

    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")

    model_path = "./climax_model.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
