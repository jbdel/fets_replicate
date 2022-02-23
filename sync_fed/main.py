import yaml
import torch
import torchio
from dataset.gandlf_data import GANDLFData
from model.PyTorch3DResUNet import PyTorch3DResUNet, load_model
from losses import fets_phase2_validation, mirrored_brats_dice_loss
from dataset.GANDLF.utils import one_hot
import utils
import os
import sys
import numpy as np
import time
from torch.optim import Adam

OUT_DIR = "sync_federation"
NUM_ROUNDS = 99
NUM_EPOCHS_PER_SITE = 1
BATCH_SIZE = 1

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == '__main__':
    # Dataloader
    config = yaml.load(open('conf.yml'), Loader=yaml.FullLoader)
    loaders = []
    for i in range(0, 5):
        g = GANDLFData(data_path="data/{}".format(i),
                       training_batch_size=BATCH_SIZE,
                       **config)
        loaders.append(g)

    # Model
    model = PyTorch3DResUNet()
    # state_dict = load_model("model/initial/")
    # model.set_tensor_dict(state_dict)

    # optim
    optimizers = [Adam(params=model.parameters(), lr=0.00005) for _ in range(len(loaders))]

    f = open(os.path.join(OUT_DIR, 'sync_train_progress.csv'), 'w+')

    # SSH
    ssh_client = utils.open_communication(OUT_DIR)

    # Start
    for num_round in range(NUM_ROUNDS):
        model.train()
        train_loss = []
        val_loss = []
        val_dice = []
        best_val_loss = float("inf")

        for i, g in enumerate(loaders):
            train_loader = g.train_loader
            val_loader = g.val_loader
            optimizer = optimizers[i]
            # train
            for epoch in range(NUM_EPOCHS_PER_SITE):
                for iteration, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    features = torch.cat([batch[key][torchio.DATA] for key in ['1', '2', '3', '4']],
                                         dim=1).float().cuda()
                    label = batch['label'][torchio.DATA].cuda()
                    label = one_hot(label, config["class_list"]).float()
                    output = model(features)
                    loss = mirrored_brats_dice_loss(output=output,
                                                    target=label,
                                                    num_classes=3,
                                                    weights=None,
                                                    class_list=config["class_list"],
                                                    to_scalar=False)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    print("\r[Round %2d][Site %2d][Epoch %2d][Step %4d/%4d] Loss: %.4f" % (
                        num_round + 1,
                        i + 1,
                        epoch + 1,
                        iteration,
                        int(len(train_loader.dataset) / BATCH_SIZE),
                        loss.cpu().data.numpy() / BATCH_SIZE,
                    ), end='          ')

            # eval
            site_val_loss = []
            site_label = []
            site_output = []
            model.eval()
            for batch in val_loader:
                features = torch.cat([batch[key][torchio.DATA] for key in ['1', '2', '3', '4']], dim=1).float()
                output = g.infer_with_crop_and_patches(
                    model_inference_function=[model.infer_batch_with_no_numpy_conversion],
                    features=features)

                label = batch['label'][torchio.DATA]
                label = one_hot(label, g.class_list).float()

                loss = mirrored_brats_dice_loss(output=output,
                                                target=label,
                                                num_classes=3,
                                                weights=None,
                                                class_list=config["class_list"],
                                                to_scalar=False)

                val_loss.append(loss.item())
                site_val_loss.append(loss.item())
                site_label.append(label.cpu().data)
                site_output.append(output.cpu().data)

            metrics = fets_phase2_validation(output=torch.cat(site_output, dim=0),
                                             target=torch.cat(site_label, dim=0),
                                             class_list=config["class_list"],
                                             fine_grained=model.validate_with_fine_grained_dice,
                                             )
            # val_dice_score = (metrics["binary_DICE_ET"] + metrics["binary_DICE_TC"] + metrics["binary_DICE_WT"]) / 3
            val_dice_score = metrics["binary_DICE_WT"]
            val_dice.append(val_dice_score)
            print("\n[Round %2d][Site %2d] Val Loss: %.4f, mean dice: %.4f" % (
                num_round + 1,
                i + 1,
                np.mean(np.array(site_val_loss)),
                val_dice_score
            ))

        mean_train_loss = np.mean(np.array(train_loss))
        mean_val_loss = np.mean(np.array(val_loss))
        mean_val_dice = np.mean(np.array(val_dice))
        best_val_loss = mean_val_loss if mean_val_loss < best_val_loss else best_val_loss
        print("End of round", num_round)
        print("mean_train_loss", mean_train_loss)
        print("mean_val_loss", mean_val_loss)
        print("mean_val_dice (WT)", mean_val_dice)
        print("best_val_loss", best_val_loss)
        print("Uploading...")
        # Put online
        f.write('{},{},{},{},{}\n'.format(num_round,
                                          1,
                                          mean_train_loss,
                                          mean_val_loss,
                                          mean_val_dice))
        f.close()
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
        utils.put_file(ssh_client, OUT_DIR, 'model.pt')
        utils.put_file(ssh_client, OUT_DIR, 'sync_train_progress.csv')
        print("Upload done... Waiting for new line in csv")
        num_lines = -1
        # Wait for new line
        while num_lines != (2 * (num_round + 1)):
            time.sleep(5)
            utils.get_file(ssh_client, OUT_DIR, 'sync_train_progress.csv')
            num_lines = len(open(os.path.join(OUT_DIR, 'sync_train_progress.csv'), 'r').readlines())

        # get model, and start over
        utils.get_file(ssh_client, OUT_DIR, 'model.pt')
        model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'model.pt')))
        f = open(os.path.join(OUT_DIR, 'sync_train_progress.csv'), 'a')
