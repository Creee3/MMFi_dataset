import os
import argparse
import math
import json

import yaml
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize

from feeder.mmfi import make_dataset, make_dataloader
from feeder.person_in_wifi_3d import PersonInWif3D, piw3_make_dataloader
from feeder.wipose import WiPose, wp_make_dataloader

from utils import *
from model.model import *
from utils import setup_seed

from torchvision import models, transforms

from model.metafi.mynetwork import metafinet, metafi_weights_init
from model.hpeli.hpeli import hpelinet, hpeli_weights_init

import sys


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pose Decoding Stage")
    parser.add_argument("--config_file", type=str, help="Configuration YAML file", default='config/mmfi/pose_config.yaml')
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--eval_num_workers", type=int, default=None)


    args = parser.parse_args()
    
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.eval_num_workers is not None:
        config['eval_num_workers'] = args.eval_num_workers
    num_workers = int(config.get('num_workers', 8))
    eval_num_workers = int(config.get('eval_num_workers', num_workers))
    if num_workers < 0 or eval_num_workers < 0:
        raise ValueError('num_workers and eval_num_workers must be non-negative.')

    run_root = os.path.abspath(config.get('run_root', '.'))
    os.makedirs(run_root, exist_ok=True)
    save_epoch_features = bool(config.get('save_epoch_features', False))
    log_dir = os.path.join(run_root, 'logs', config['dataset_name'], config['setting'], 'pose_logs', config['experiment_name'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_name = os.path.join(log_dir, f"train_seed_{config['seed']}.log")
    sys.stdout = Logger(log_name)
    print(f"日志将保存至: {log_name}")
    
    setup_seed(config['seed'])
    dataset_root = config['dataset_root']

    batch_size = config['batch_size']
    load_batch_size = min(config['max_device_batch_size'], batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # load dataset
    if config['dataset_name'] == 'mmfi-csi':
        train_dataset, val_dataset = make_dataset(config['training_semi'], dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(
            train_dataset, is_training=True, generator=rng_generator,
            batch_size=load_batch_size, num_workers=num_workers
        )
        val_loader = make_dataloader(
            val_dataset, is_training=False, generator=rng_generator,
            batch_size=load_batch_size, num_workers=eval_num_workers
        )
        print(f'Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}')
        print(f'DataLoader workers: train={num_workers} | validation={eval_num_workers}')
    elif config['dataset_name'] == 'wipose':
        train_dataset = WiPose('training', dataset_root)
        val_dataset = WiPose('validation', dataset_root)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = wp_make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size)
        val_loader = wp_make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=load_batch_size)
    elif config['dataset_name'] == 'person-in-wifi-3d':
        train_dataset = PersonInWif3D('training', dataset_root, config['setting'])
        val_dataset = PersonInWif3D('validation', dataset_root, config['setting'])
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = piw3_make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size)
        val_loader = piw3_make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=load_batch_size)
    else:
        print('No dataset!')

    # TODO: Settings, e.g., your model, optimizer, device, ...
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config['pretrained_model_path'] is not None:
        print('*'*20+'   Load Pretrain Weights   '+'*'*20)
        print('*'*20+'  '+config['dataset_name']+','+config['setting']+','+config['experiment_name']+'   '+'*'*20)
        model = torch.load(config['pretrained_model_path'], map_location='cpu', weights_only=False)
        writer = SummaryWriter(os.path.join(run_root, 'logs', config['dataset_name'], config['setting'], 'pose_pretrain',config['experiment_name']))
        if config['dataset_name'] == 'mmfi-csi':
            model = ViT_Pose_Decoder(model.encoder, keypoints=17, coor_num=3, token_num=114, dataset=config['dataset_name']).to(device)   # 72*5
            optim = torch.optim.SGD(
                filter(lambda p:p.requires_grad, model.parameters()),
                lr=float(config.get('base_learning_rate', 1e-3)),
                weight_decay=0.01,
            )
        elif config['dataset_name'] == 'person-in-wifi-3d':
            model = ViT_Pose_Decoder(model.encoder, keypoints=14, coor_num=3, token_num=90*10, dataset=config['dataset_name'], num_person=config['num_person']).to(device)
            optim = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3) 
        elif config['dataset_name'] == 'wipose':
            model = ViT_Pose_Decoder(model.encoder, keypoints=18, coor_num=2, token_num=45*5, dataset=config['dataset_name']).to(device)
            optim = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3)  
        # save model
        weights_path = os.path.join(config['save_path'], config['dataset_name'], config['setting'], 'pose_pretrain', config['experiment_name'])
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        # save train feature
        train_feature_path = os.path.join(run_root, 'features', config['dataset_name'], config['setting'], 'pose_pretrain', config['experiment_name'])
        if not os.path.exists(train_feature_path):
            os.makedirs(train_feature_path)
        # save test feature
        test_feature_path = os.path.join(run_root, 'features', config['dataset_name'], config['setting'], 'pose_pretrain', config['experiment_name'])
        if not os.path.exists(test_feature_path):
            os.makedirs(test_feature_path)
    else:
        print('*'*20+'   Training from Scratch  '+'*'*20)
        print('*'*20+'  '+config['dataset_name']+','+config['setting']+','+config['experiment_name']+'   '+'*'*20)
        writer = SummaryWriter(os.path.join(run_root, 'logs', config['dataset_name'], config['setting'], 'pose_scratch',config['experiment_name']))
        if config['dataset_name'] == 'mmfi-csi':
            if config['experiment_name'] == 'metafi':
                model = metafinet(num_keypoints=17, num_coor=3, num_person=config['num_person'],dataset=config['dataset_name']).to(device)
                # model.apply(metafi_weights_init)使用默认的kaiming初始化方式
                optim = torch.optim.SGD(
                    model.parameters(),
                    lr=float(config.get('base_learning_rate', 1.85e-3)),
                    momentum=0.9,
                )
            elif config['experiment_name'] == 'hpeli':
                model = hpelinet(num_keypoints=17, num_coor=3, subcarrier_num=114, num_person=config['num_person'],dataset=config['dataset_name']).to(device)
                model.apply(hpeli_weights_init)
                optim = torch.optim.SGD(
                    model.parameters(),
                    lr=float(config.get('base_learning_rate', 1e-3)),
                    momentum=0.9,
                )
            else:
                model = MAE_ViT(image_size=(114, 10),
                        patch_size=(2,2),
                        encoder_layer=4,
                        encoder_head=4,
                        decoder_layer=2,
                        decoder_head=4,
                        emb_dim=256)
                model = ViT_Pose_Decoder(model.encoder, keypoints=17, coor_num=3, token_num=285, dataset=config['dataset_name']).to(device)
                optim = torch.optim.AdamW(
                    model.parameters(), lr=float(config.get('base_learning_rate', 1e-3))
                )
        elif config['dataset_name'] == 'person-in-wifi-3d':
            if config['experiment_name'] == 'metafi':
                model = metafinet(num_keypoints=14, num_coor=3, num_person=config['num_person'],dataset=config['dataset_name']).to(device)
                model.apply(metafi_weights_init)
                optim = torch.optim.AdamW(model.parameters(), lr=1e-2)  
            elif config['experiment_name'] == 'hpeli':
                model = hpelinet(num_keypoints=14, num_coor=3, subcarrier_num=180, num_person=config['num_person'],dataset=config['dataset_name']).to(device)
                model.apply(hpeli_weights_init)
                optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
            else:
                model = MAE_ViT(image_size=(180, 20),
                        patch_size=(2,2),
                        encoder_layer=4,
                        encoder_head=4,
                        decoder_layer=2,
                        decoder_head=4,
                        emb_dim=256,
                        input_dim=3)
                model = ViT_Pose_Decoder(model.encoder, keypoints=14, coor_num=3, token_num=900, dataset=config['dataset_name'], num_person=config['num_person']).to(device)
                optim = torch.optim.SGD(model.parameters(),lr=1e-2, momentum=0.9)
        elif config['dataset_name'] == 'wipose':
            if config['experiment_name'] == 'metafi':
                model = metafinet(num_keypoints=18, num_coor=2, dataset=config['dataset_name']).to(device)
                model.apply(metafi_weights_init)
                optim = torch.optim.SGD(model.parameters(),lr=1e-3, momentum=0.9)
            elif config['experiment_name'] == 'hpeli':
                model = hpelinet(num_keypoints=18, num_coor=2, subcarrier_num=90, dataset=config['dataset_name']).to(device)
                model.apply(hpeli_weights_init)
                optim = torch.optim.SGD(model.parameters(),lr=1e-3, momentum=0.9)   #1e-3
            else:
                model = MAE_ViT(image_size=(90, 5),
                        patch_size=(2,1),
                        encoder_layer=4,
                        encoder_head=4,
                        decoder_layer=2,
                        decoder_head=4,
                        emb_dim=256,
                        input_dim=3)
                model = ViT_Pose_Decoder(model.encoder, keypoints=18, coor_num=2, token_num=45*5, dataset=config['dataset_name']).to(device)
                optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # save model
        weights_path = os.path.join(config['save_path'], config['dataset_name'], config['setting'], 'pose_scratch', config['experiment_name'])
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        # save train feature
        train_feature_path = os.path.join(run_root, 'features', config['dataset_name'], config['setting'], 'pose_scratch', config['experiment_name'])
        if not os.path.exists(train_feature_path):
            os.makedirs(train_feature_path)
        # save test feature
        test_feature_path = os.path.join(run_root, 'features', config['dataset_name'], config['setting'], 'pose_scratch', config['experiment_name'])
        if not os.path.exists(test_feature_path):
            os.makedirs(test_feature_path)


    # TODO: Codes for training (and saving models)
    optim.zero_grad()
    step_count = 0
    best_val_mpjpe = 1
    best_val_pampjpe = 1
    best_val_mpjpe_align = 1
    best_val_pampjpe_align = 1
    best_val_pck = [0 for _ in range(5)]
    best_val_pck_align = [0 for _ in range(5)]
    best_epoch = None
    pck_order = [50, 40, 30, 20, 10]
    checkpoint_selection = config.get('checkpoint_selection', 'pa_mpjpe').lower()
    if checkpoint_selection not in {'pa_mpjpe', 'mpjpe'}:
        raise ValueError(
            "checkpoint_selection must be 'pa_mpjpe' or 'mpjpe', "
            f"got {checkpoint_selection!r}"
        )
    print(f'Checkpoint selection: validation {checkpoint_selection}')
    history = []
  
    for epoch in range(config['total_epoch']):
        model.train()
        losses = []
        mpjpe_list = []
        pampjpe_list = []
        mpjpe_align_list = []
        pampjpe_align_list = []
        feature_list = []
        wifi_list = []
        label_list = []
        pred_list = []
        gt_list = []
       
        pck_iter = [[] for _ in range(5)]
        pck_align_iter = [[] for _ in range(5)]
        attention_list_first = []
        attention_list_second = []
        for batch_idx, batch_data in enumerate(train_loader):
            # Please check the data structure here.
            csi_data = batch_data['input_wifi-csi'].to(device)
            n, _, _, _ = csi_data.size()
            pose_gt = batch_data['output'].to(device)
            
            if config['dataset_name'] == 'mmfi-csi' or config['dataset_name'] == 'wipose':
                label = batch_data['label'].to(device)
                label_list.append(label.data.cpu().numpy())
            predicted_pose, feature = model(csi_data)
            if config['dataset_name'] == 'person-in-wifi-3d':
                person_num = batch_data['person_num'].to(device)
                predicted_pose = torch.cat([predicted_pose[idx,:num,:,:].reshape((-1, 14, 3)) for idx, num in enumerate(person_num)], dim=0)
                pose_gt = torch.cat([pose_gt[idx,:num,:,:].reshape((-1, 14, 3)) for idx, num in enumerate(person_num)], dim=0)

            gt_list.append(pose_gt.data.cpu().numpy())
            pred_list.append(predicted_pose.data.cpu().numpy())
            feature_list.append(feature.data.cpu().numpy())
            wifi_list.append(csi_data.data.cpu().numpy())
            """
            loss_mpjpe = torch.mean(torch.norm(predicted_pose-pose_gt, dim=-1))
            loss = loss_mpjpe
            loss.backward()
            """
            loss_mse = torch.nn.functional.mse_loss(predicted_pose, pose_gt)
            loss = loss_mse
            loss.backward()
            """
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(predicted_pose, pose_gt)
            loss.backward()
            """
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        avg_train_loss = sum(losses) / len(losses)
 
        current_lr = optim.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, global_step=epoch)
        feature_list = np.concatenate(feature_list, axis=0)
        wifi_list = np.concatenate(wifi_list, axis=0)
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        if config['dataset_name'] == 'mmfi-csi' or config['dataset_name'] == 'wipose':
            label_list = np.concatenate(label_list, axis=0)

        if save_epoch_features:
            if config['dataset_name'] == 'mmfi-csi':
                np.savez(os.path.join(train_feature_path, 'train_feature.npz'), fea=feature_list, pred_pose=pred_list, gt_pose=gt_list, att=[attention_list_first, attention_list_second], wifi=wifi_list, label=label_list)
            elif config['dataset_name'] == 'person-in-wifi-3d':
                np.savez(os.path.join(train_feature_path, 'train_feature.npz'), fea=feature_list, pred_pose=pred_list, gt_pose=gt_list, att=[attention_list_first, attention_list_second], wifi=wifi_list)
            elif config['dataset_name'] == 'wipose':
                np.savez(os.path.join(train_feature_path, 'train_feature.npz'), fea=feature_list, pred_pose=pred_list, gt_pose=gt_list, att=[attention_list_first, attention_list_second], wifi=wifi_list, label=label_list)
        print('*'*100)
        print(f'In epoch {epoch}, learning rate: {current_lr}, traning loss:{avg_train_loss}.')


        # # TODO: Codes for test (if)
        model.eval()
        gt_list = []
        pred_list = []
        feature_list = []
        wifi_list = []
        label_list = []
        attention_list_first = []
        attention_list_second = []
        with torch.no_grad():
            losses = []
            mpjpe_list = []
            pampjpe_list = []
            mpjpe_joints_list = []
            pampjpe_joints_list = []
            mpjpe_align_list = []
            pampjpe_align_list = []
            
            pck_iter = [[] for _ in range(5)]
            pck_align_iter = [[] for _ in range(5)]
            subject_mpjpe = {}
            for batch_idx, batch_data in enumerate(val_loader):
                val_csi = batch_data['input_wifi-csi'].to(device)
                val_pose_gt = batch_data['output'].to(device)
                if config['dataset_name'] == 'mmfi-csi' or config['dataset_name'] == 'wipose':
                    label = batch_data['label'].to(device)
                    label_list.append(label.data.cpu().numpy())

                predicted_val_pose, pred_fea = model(val_csi)

                if config['dataset_name'] == 'person-in-wifi-3d':
                    person_num = batch_data['person_num'].to(device)
                    predicted_val_pose = torch.cat([predicted_val_pose[idx,:num,:,:].reshape((-1, 14, 3)) for idx, num in enumerate(person_num)], dim=0)
                    val_pose_gt = torch.cat([val_pose_gt[idx,:num,:,:].reshape((-1, 14, 3)) for idx, num in enumerate(person_num)], dim=0)


                feature_list.append(pred_fea.data.cpu().numpy())
                wifi_list.append(val_csi.data.cpu().numpy())
                gt_list.append(val_pose_gt.data.cpu().numpy())
                pred_list.append(predicted_val_pose.data.cpu().numpy())
                
                # loss = torch.mean(torch.norm(predicted_val_pose-val_pose_gt, dim=-1))
                #修改损失函数为MSE
                loss = torch.nn.functional.mse_loss(predicted_val_pose, val_pose_gt)
                # calculate the pck, mpjpe, pampjpe
                for idx, percentage in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                    pck_iter[idx].append(compute_pck_pckh(predicted_val_pose.permute(0,2,1).data.cpu().numpy(), val_pose_gt.permute(0,2,1).data.cpu().numpy(), percentage, align=False, dataset=config['dataset_name']))
                mpjpe, pampjpe, mpjpe_joints, pampjpe_joints = calulate_error(predicted_val_pose.data.cpu().numpy(), val_pose_gt.data.cpu().numpy(), align=False)
                mpjpe_list += mpjpe.tolist()
                pampjpe_list += pampjpe.tolist()
                losses.append(loss.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
            avg_val_pampjpe = sum(pampjpe_list) / len(pampjpe_list)
            if config['dataset_name'] == 'mmfi-csi':
                pck_overall = [np.mean(pck_value, 0)[17] for pck_value in pck_iter]
            elif config['dataset_name'] == 'person-in-wifi-3d':
                pck_overall = [np.mean(pck_value, 0)[14] for pck_value in pck_iter]
            elif config['dataset_name'] == 'wipose':
                pck_overall = [np.mean(pck_value, 0)[18] for pck_value in pck_iter]
                
            gt_list = np.concatenate(gt_list, axis=0)
            pred_list = np.concatenate(pred_list, axis=0)
            feature_list = np.concatenate(feature_list, axis=0)
            wifi_list = np.concatenate(wifi_list, axis=0)
            if config['dataset_name'] == 'mmfi-csi' or config['dataset_name'] == 'wipose':
                label_list = np.concatenate(label_list, axis=0)
            
        print(f'In epoch {epoch}, test losss: {avg_val_loss}')
        print(f'test mpjpe: {avg_val_mpjpe}, test pa-mpjpe: {avg_val_pampjpe}, test pck50: {pck_overall[0]}, test pck40: {pck_overall[1]}, test pck30: {pck_overall[2]}, test pck20: {pck_overall[3]}, test pck10: {pck_overall[4]}.')
        if save_epoch_features:
            if config['dataset_name'] == 'mmfi-csi':
                np.savez(os.path.join(test_feature_path, 'test_feature.npz'), fea=feature_list, pred_pose=pred_list, gt_pose=gt_list, att=[attention_list_first, attention_list_second], wifi=wifi_list, label=label_list)
            elif config['dataset_name'] == 'person-in-wifi-3d':
                np.savez(os.path.join(test_feature_path, 'test_feature.npz'), fea=feature_list, pred_pose=pred_list, gt_pose=gt_list, att=[attention_list_first, attention_list_second], wifi=wifi_list)
            elif config['dataset_name'] == 'wipose':
                np.savez(os.path.join(test_feature_path, 'test_feature.npz'), fea=feature_list, pred_pose=pred_list, gt_pose=gt_list, att=[attention_list_first, attention_list_second], wifi=wifi_list, label=label_list)




       
        ''' save model '''
        # The selected metric controls the epoch; all saved metric files are
        # synchronized snapshots from that same epoch.
        if checkpoint_selection == 'mpjpe':
            improved = avg_val_mpjpe < best_val_mpjpe
            selection_value = avg_val_mpjpe
        else:
            improved = avg_val_pampjpe < best_val_pampjpe
            selection_value = avg_val_pampjpe

        if improved:
            best_epoch = epoch
            best_val_mpjpe = avg_val_mpjpe
            best_val_pampjpe = avg_val_pampjpe
            print(
                f'saving best model with {checkpoint_selection} '
                f'{selection_value} at {epoch} epoch!'
            )
            torch.save(model, '{}/pose_pampjpe.pt'.format(weights_path))
            torch.save(model, '{}/pose_mpjpe.pt'.format(weights_path))

            for idx, pck_value in enumerate(pck_overall):
                best_val_pck[idx] = pck_value
                torch.save(model, '{}/pose_pck{}.pt'.format(weights_path, pck_order[idx]))

            print(f'---> synchronized all metrics at Epoch {epoch} <---')

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=epoch)

    print('*'*100)
    print(f'Best mpjpe: {best_val_mpjpe}') 
    print(f'Best pa-mpjpe: {best_val_pampjpe}')  
    for idx, pck_value in enumerate(best_val_pck):
        print(f'Best pck{pck_order[idx]}: {pck_value}')   
    print('*'*100)
    with open(os.path.join(weights_path, 'selection.json'), 'w', encoding='utf-8') as fd:
        json.dump(
            {
                'selection_metric': checkpoint_selection,
                'selected_epoch': best_epoch,
                'mpjpe': float(best_val_mpjpe),
                'pa_mpjpe': float(best_val_pampjpe),
                'pck50': float(best_val_pck[0]),
                'pck40': float(best_val_pck[1]),
                'pck30': float(best_val_pck[2]),
                'pck20': float(best_val_pck[3]),
                'pck10': float(best_val_pck[4]),
            },
            fd,
            indent=2,
        )
    
    
