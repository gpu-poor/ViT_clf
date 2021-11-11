import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from ipynb.fs.full.vit_notebook import ViTfinetune, FireSmokeDataModule


def get_img_aug_obj():
    transform = transforms.Compose([
     transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.CenterCrop((100, 100)),
     transforms.RandomCrop((80, 80)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=(-90, 90)),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor()
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
    return transform






def main():

    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=3,
       verbose=True,
       mode='min'
    )

    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CKPT_PATH,
        monitor='val_loss',
        filename=MODEL_CKPT ,
        save_top_k=3,
        mode='min')
    
    #Dataset creation
    CSV_FILE="./raw_data/fire&safety&smoke_processed_train.csv"
    IMAGE_DIR = "../fnS_images/"
    transform = get_img_aug_obj()
    # Init our data pipeline
    dm = FireSmokeDataModule(
        csv_file_path=CSV_FILE,
        class_list=['fire', 'spark', 'smoke'],
        image_dir=IMAGE_DIR,
        filename_col_name="Filename",
        batch_size=64,
        transform=transform

    )
    # To access the x_dataloader we need to call prepare_data and setup.
    dm.prepare_data()
    dm.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape

    
    #Training code 


    model = ViTfinetune(num_classes=3)
    tb_logger = TensorBoardLogger("logs/")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=50,
                         progress_bar_refresh_rate=20, 
                         gpus=1, 
                         logger=tb_logger,
                         callbacks=[early_stop_callback, checkpoint_callback])

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test()





if __name__ == "__main__":
    main()