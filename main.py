import yaml
import torch
import torchio
from dataset.gandlf_data import GANDLFData
from model.PyTorch3DResUNet import PyTorch3DResUNet, load_model
from losses import fets_phase2_validation, mirrored_brats_dice_loss
from dataset.GANDLF.utils import one_hot

if __name__ == '__main__':
    # Dataloader
    config = yaml.load(open('conf.yml'), Loader=yaml.FullLoader)
    g = GANDLFData(data_path="data", **config)
    train_loader = g.train_loader
    val_loader = g.val_loader

    # Model
    model = PyTorch3DResUNet()
    state_dict = load_model("model/initial/")
    model.set_tensor_dict(state_dict)
    loss_fn = mirrored_brats_dice_loss
    model.eval()

    with torch.no_grad():
        for batch in train_loader:
            features = torch.cat([batch[key][torchio.DATA] for key in ['1', '2', '3', '4']], dim=1).float().cuda()
            label = batch['label'][torchio.DATA].cuda()
            label = one_hot(label, g.class_list).float()

            print('features.shape', features.shape)
            print('label.shape', label.shape)

            output = model(features)
            loss = loss_fn(output=output,
                           target=label,
                           num_classes=3,
                           weights=None,
                           class_list=g.class_list,
                           to_scalar=False)
            print(loss)
            break

        for batch in val_loader:
            features = torch.cat([batch[key][torchio.DATA] for key in ['1', '2', '3', '4']], dim=1).float()
            output = g.infer_with_crop_and_patches(
                model_inference_function=[model.infer_batch_with_no_numpy_conversion],
                features=features)

            label = batch['label'][torchio.DATA]
            label = one_hot(label, g.class_list).float()

            current_valscore = fets_phase2_validation(output=output,
                                                      target=label,
                                                      class_list=g.class_list,
                                                      fine_grained=model.validate_with_fine_grained_dice,
                                                      )
            print(current_valscore)
            break