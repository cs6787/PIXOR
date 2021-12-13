from pandas.core.frame import DataFrame
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import random
from torch.multiprocessing import Pool

from loss import CustomLoss
from dist_loss import CustomDistLoss
from datagen import get_data_loader
from model import PIXOR
from dist_model import PIXOR_DIST
from dist_model2 import PIXOR_DIST2
from utils import get_model_name, load_config, get_logger, plot_bev, plot_label_map, plot_pr_curve, get_bev
from postprocess import filter_pred, compute_matches, compute_ap
import torch.nn.utils.prune as prune
import pandas as pd


def build_model(config, device, train=True):
    net = PIXOR(config['geometry'], config['use_bn'])
    loss_fn = CustomLoss(device, config, num_classes=1)

    if torch.cuda.device_count() <= 1:
        config['mGPUs'] = False
    if config['mGPUs']:
        print("using multi gpu")
        net = nn.DataParallel(net)

    net = net.to(device)
    loss_fn = loss_fn.to(device)
    if not train:
        return net, loss_fn

    optimizer = torch.optim.SGD(net.parameters(
    ), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['lr_decay_at'], gamma=0.1)

    return net, loss_fn, optimizer, scheduler


def build_dist_model(config, device, train=True, distill="distill_layer"):

    if distill == "distill_layer":
        net = PIXOR_DIST(config['geometry'], config['use_bn'])

    if distill == "distill_conv":
        net = PIXOR_DIST2(config['geometry'], config['use_bn'])

    loss_fn = CustomDistLoss(device, config, num_classes=1)

    if torch.cuda.device_count() <= 1:
        config['mGPUs'] = False
    if config['mGPUs']:
        print("using multi gpu")
        net = nn.DataParallel(net)  # parallelizing net

    net = net.to(device)
    loss_fn = loss_fn.to(device)

    if not train:
        return net, loss_fn

    optimizer = torch.optim.SGD(net.parameters(
    ), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['lr_decay_at'], gamma=0.1)

    return net, loss_fn, optimizer, scheduler


def eval_batch_distill(config, net, teacher_net, loss_fn, loader, device, eval_range='all'):

    net.eval()
    teacher_net.eval()  # even though I'm pretty sure it's already in eval mode

    if config['mGPUs']:
        net.module.set_decode(True)
        teacher_net.module.set_decode(True)
    else:
        net.set_decode(True)
        teacher_net.set_decode(True)
    

    cls_loss = 0
    loc_loss = 0
    cls_teacher_loss = 0
    loc_teacher_loss = 0
    all_scores = []
    all_matches = []
    log_images = []
    gts = 0
    preds = 0
    t_fwd = 0
    t_nms = 0

    log_img_list = random.sample(range(len(loader.dataset)), 10)

    with torch.no_grad():
        for i, data in enumerate(loader):
            tic = time.time()
            input, label_map, image_id = data
            input = input.to(device)
            label_map = label_map.to(device)
            tac = time.time()
            predictions = net(input)
            t_fwd += time.time() - tac

            # getting the teacher predictions, no need to get forward times for this set
            teacher_predictions = teacher_net(input)

            loss, cls, loc, cls_teacher, loc_teacher = loss_fn(
                predictions, teacher_predictions, label_map)
            cls_loss += cls
            loc_loss += loc
            cls_teacher_loss += cls_teacher
            loc_teacher_loss += loc_teacher
            t_fwd += (time.time() - tic)

            toc = time.time()
            # Parallel post-processing
            predictions = list(torch.split(predictions.cpu(), 1, dim=0))
            batch_size = len(predictions)
            # with Pool(processes=3) as pool:
            #    preds_filtered = pool.starmap(
            #        filter_pred, [(config, pred) for pred in predictions])

            preds_filtered = []
            for pred in predictions:
                preds_filtered.append(filter_pred(config, pred))

            t_nms += (time.time() - toc)
            args = []

            for j in range(batch_size):
                _, label_list = loader.dataset.get_label(image_id[j].item())
                corners, scores = preds_filtered[j]
                gts += len(label_list)
                preds += len(scores)
                all_scores.extend(list(scores))
                if image_id[j] in log_img_list:
                    input_np = input[j].cpu().permute(1, 2, 0).numpy()
                    pred_image = get_bev(input_np, corners)
                    log_images.append(pred_image)

                arg = (np.array(label_list), corners, scores)
                args.append(arg)

            # Parallel compute matchesi

            with Pool(processes=3) as pool:
                matches = pool.starmap(compute_matches, args)

            for j in range(batch_size):
                all_matches.extend(list(matches[j][1]))

            # print(time.time() -tic)

    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    try:
        AP, precisions, recalls, precision, recall = compute_ap(
            all_matches, gts, preds)
    except:
        AP, precisions, recalls, precision, recall = [-1]*5

    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['Forward Pass Time'] = t_fwd/len(loader.dataset)
    metrics['Postprocess Time'] = t_nms/len(loader.dataset)

    cls_loss = cls_loss / len(loader)
    loc_loss = loc_loss / len(loader)
    cls_teacher_loss = cls_teacher_loss / len(loader)
    loc_teacher_loss = loc_teacher_loss / len(loader)
    metrics['loss'] = cls_loss + loc_loss + cls_teacher_loss + loc_teacher_loss

    return metrics, precisions, recalls, log_images


def eval_batch(config, net, loss_fn, loader, device, eval_range='all'):
    net.eval()
    if config['mGPUs']:
        net.module.set_decode(True)
    else:
        net.set_decode(True)

    cls_loss = 0
    loc_loss = 0
    all_scores = []
    all_matches = []
    log_images = []
    gts = 0
    preds = 0
    t_fwd = 0
    t_nms = 0

    log_img_list = random.sample(range(len(loader.dataset)), 10)

    with torch.no_grad():
        for i, data in enumerate(loader):
            tic = time.time()
            input, label_map, image_id = data
            input = input.to(device)
            label_map = label_map.to(device)
            tac = time.time()
            predictions = net(input)
            t_fwd += time.time() - tac
            loss, cls, loc = loss_fn(predictions, label_map)
            cls_loss += cls
            loc_loss += loc
            t_fwd += (time.time() - tic)

            toc = time.time()
            # Parallel post-processing
            predictions = list(torch.split(predictions.cpu(), 1, dim=0))
            batch_size = len(predictions)
            # with Pool(processes=3) as pool:
            #    preds_filtered = pool.starmap(
            #        filter_pred, [(config, pred) for pred in predictions])

            preds_filtered = []
            for pred in predictions:
                preds_filtered.append(filter_pred(config, pred))

            t_nms += (time.time() - toc)
            args = []

            for j in range(batch_size):
                _, label_list = loader.dataset.get_label(image_id[j].item())
                corners, scores = preds_filtered[j]
                gts += len(label_list)
                preds += len(scores)
                all_scores.extend(list(scores))
                if image_id[j] in log_img_list:
                    input_np = input[j].cpu().permute(1, 2, 0).numpy()
                    pred_image = get_bev(input_np, corners)
                    log_images.append(pred_image)

                arg = (np.array(label_list), corners, scores)
                args.append(arg)

            # Parallel compute matchesi

            with Pool(processes=3) as pool:
                matches = pool.starmap(compute_matches, args)

            for j in range(batch_size):
                all_matches.extend(list(matches[j][1]))

            # print(time.time() -tic)

    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    try:
        AP, precisions, recalls, precision, recall = compute_ap(
            all_matches, gts, preds)
    except:
        AP, precisions, recalls, precision, recall = [-1]*5

    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['Forward Pass Time'] = t_fwd/len(loader.dataset)
    metrics['Postprocess Time'] = t_nms/len(loader.dataset)

    cls_loss = cls_loss / len(loader)
    loc_loss = loc_loss / len(loader)
    metrics['loss'] = cls_loss + loc_loss

    return metrics, precisions, recalls, log_images


def eval_dataset(config, net, loss_fn, loader, device, e_range='all'):
    net.eval()
    if config['mGPUs']:
        net.module.set_decode(True)
    else:
        net.set_decode(True)

    t_fwds = 0
    t_post = 0
    loss_sum = 0

    img_list = range(len(loader.dataset))
    if e_range != 'all':
        e_range = min(e_range, len(loader.dataset))
        img_list = random.sample(img_list, e_range)

    log_img_list = random.sample(img_list, 10)

    gts = 0
    preds = 0
    all_scores = []
    all_matches = []
    log_images = []

    with torch.no_grad():
        for image_id in img_list:
            # tic = time.time()
            num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = eval_one(net, loss_fn, config, loader,
                                                                                                image_id, device, plot=False)
            gts += num_gt
            preds += num_pred
            loss_sum += loss
            all_scores.extend(list(scores))
            all_matches.extend(list(pred_match))

            t_fwds += t_forward
            t_post += t_nms

            if image_id in log_img_list:
                log_images.append(pred_image)
            # print(time.time() - tic)

    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(
        all_matches, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['loss'] = loss_sum / len(img_list)
    metrics['Forward Pass Time'] = t_fwds / len(img_list)
    metrics['Postprocess Time'] = t_post / len(img_list)

    return metrics, precisions, recalls, log_images


def train_orig(exp_name, device):
    # Load Hyperparameters
    config, learning_rate, batch_size, max_epochs = load_config(exp_name)

    # Dataset and DataLoader
    train_data_loader, test_data_loader = get_data_loader(batch_size, config['use_npy'],
                                                          geometry=config['geometry'], frame_range=config['frame_range'])
    # Model
    net, loss_fn, optimizer, scheduler = build_model(
        config, device, train=True)

    # Tensorboard Logger
    train_logger = get_logger(config, 'train')
    val_logger = get_logger(config, 'val')

    if config['resume_training']:
        saved_ckpt_path = get_model_name(config)
        if config['mGPUs']:
            net.module.load_state_dict(torch.load(
                saved_ckpt_path, map_location=device))
        else:
            net.load_state_dict(torch.load(
                saved_ckpt_path, map_location=device))
        print("Successfully loaded trained ckpt at {}".format(saved_ckpt_path))
        st_epoch = config['resume_from']
    else:
        # writefile(config, 'train_loss.csv', 'iteration, cls_loss, loc_loss\n')
        # writefile(config, 'val_loss.csv', 'epoch, cls_loss, loc_loss\n')
        st_epoch = 0

    step = 1 + st_epoch * len(train_data_loader)
    cls_loss = 0
    loc_loss = 0
    for epoch in range(st_epoch, max_epochs):
        start_time = time.time()

        train_loss = 0

        net.train()
        if config['mGPUs']:
            net.module.set_decode(False)
        else:
            net.set_decode(False)
        scheduler.step()

        for input, label_map, image_id in train_data_loader:

            tic = time.time()  # print('step', step)
            input = input.to(device)
            label_map = label_map.to(device)
            optimizer.zero_grad()

            # Forward
            predictions = net(input)
            loss, cls, loc = loss_fn(predictions, label_map)
            loss.backward()
            optimizer.step()
            cls_loss += cls
            loc_loss += loc
            train_loss += loss.item()

            if step % config['log_every'] == 0:
                cls_loss = cls_loss / config['log_every']
                loc_loss = loc_loss / config['log_every']
                train_logger.scalar_summary('cls_loss', cls_loss, step)
                train_logger.scalar_summary('loc_loss', loc_loss, step)
                cls_loss = 0
                loc_loss = 0

                # for tag, value in net.named_parameters():
                #    tag = tag.replace('.', '/')
                #    train_logger.histo_summary(tag, value.data.cpu().numpy(), step)
                #    train_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

            step += 1
            # print(time.time() - tic)

        # Record Training Loss
        train_loss = train_loss / len(train_data_loader)
        train_logger.scalar_summary('loss', train_loss, epoch + 1)
        print("Epoch {}|Time {:.3f}|Training Loss: {:.5f}".format(
            epoch + 1, time.time() - start_time, train_loss))

        # Run Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            tic = time.time()
            val_metrics, _, _, log_images = eval_batch(
                config, net, loss_fn, test_data_loader, device)
            for tag, value in val_metrics.items():
                val_logger.scalar_summary(tag, value, epoch + 1)
            val_logger.image_summary('Predictions', log_images, epoch + 1)
            print("Epoch {}|Time {:.3f}|Validation Loss: {:.5f}".format(
                epoch + 1, time.time() - tic, val_metrics['loss']))

        # Save Checkpoint
        if (epoch + 1) == max_epochs or (epoch + 1) % config['save_every'] == 0:
            model_path = get_model_name(config, epoch + 1)
            if config['mGPUs']:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print("Checkpoint saved at {}".format(model_path))

    print('Finished Training')


def train_distilled(net, loss_fn, optimizer, scheduler, teacher_net, device, config, learning_rate, batch_size, max_epochs, exp_name=None):

    df_logs = pd.DataFrame()

    # Dataset and DataLoader - using same data for distilled model (re training on same data)
    train_data_loader, test_data_loader = get_data_loader(
        batch_size, config['use_npy'], geometry=config['geometry'], frame_range=config['frame_range'])

    # Tensorboard Logger
    train_logger = get_logger(config, 'train')
    val_logger = get_logger(config, 'val')

    # setting the starting epoch
    if config['resume_training']:
        st_epoch = config['resume_from']
    else:
        st_epoch = 0

    # setting the initial step
    step = 1 + st_epoch * len(train_data_loader)
    # cross entropy loss - calculated adding the cross entropy loss wrt data and teacher model
    cls_loss = 0
    # regression loss - calculated adding regression loss wrt data and teacher model
    loc_loss = 0
    # cross entropy loss wrt to the teacher model
    cls_teacher_loss = 0
    # regression loss wrt to the teacher model
    loc_teacher_loss = 0

    best_loss = 100000000000
    num_val_not_decreasing = 0

    # training for a certain number of epochs
    for epoch in range(st_epoch, st_epoch + max_epochs):

        print("Epoch number: " + str(epoch + 1))

        start_time = time.time()  # starting timer

        train_loss = 0  # iteratively calculating training loss

        net.train()  # setting net to training mode (vs net.eval())

        # whether to set decode to false if using mGPUs
        if config['mGPUs']:
            net.module.set_decode(False)
            teacher_net.module.set_decode(False)
        else:
            net.set_decode(False)
            teacher_net.set_decode(False)

        for input, label_map, image_id, in train_data_loader:

            tic = time.time()  # print('step', step)
            input = input.to(device)
            label_map = label_map.to(device)
            optimizer.zero_grad()

            # Forward
            predictions = net(input)
            teacher_predictions = teacher_net(input)

            loss, cls, loc, cls_teacher, loc_teacher = loss_fn(
                predictions, teacher_predictions, label_map)

            loss.backward()
            optimizer.step()

            # keeping sum total of all the losses - will average later
            cls_loss += cls
            loc_loss += loc
            cls_teacher_loss += cls_teacher
            loc_teacher_loss += loc_teacher
            train_loss += loss.item()

            # logging cls_loss and loc_loss every certain number of steps
            if step % config['log_every'] == 0:
                cls_loss = cls_loss / config['log_every']
                loc_loss = loc_loss / config['log_every']
                cls_teacher_loss = cls_teacher_loss / config['log_every']
                loc_teacher_loss = loc_teacher_loss / config['log_every']
                train_logger.scalar_summary('cls_loss', cls_loss, step)
                train_logger.scalar_summary('loc_loss', loc_loss, step)
                train_logger.scalar_summary(
                    'cls_teacher_loss', cls_teacher_loss, step)
                train_logger.scalar_summary(
                    'loc_teacher_loss', loc_teacher_loss, step)
                cls_loss = 0
                loc_loss = 0
                cls_teacher_loss = 0
                loc_teacher_loss = 0

            # increasing # of steps
            step += 1

        scheduler.step()  # stepping the learning rate (after the first optimizer step)

        # Record Training Loss
        train_loss = train_loss / len(train_data_loader)
        train_logger.scalar_summary('loss', train_loss, epoch + 1)
        print("Epoch {}|Time {:.3f}|Training Loss: {:.5f}".format(
            epoch + 1, time.time() - start_time, train_loss))

        # Run Validation
        # if (epoch + 1) % 2 == 0:
        tic = time.time()
        val_metrics, _, _, log_images = eval_batch_distill(
            config, net, teacher_net, loss_fn, test_data_loader, device)
        for tag, value in val_metrics.items():
            val_logger.scalar_summary(tag, value, epoch + 1)
        val_logger.image_summary('Predictions', log_images, epoch + 1)
        print("Epoch {}|Time {:.3f}|Validation Loss: {:.5f}".format(
            epoch + 1, time.time() - tic, val_metrics['loss']))
        print(val_metrics)


        # Save Checkpoint, implemented with early stopping if validation loss is too high
        # if (epoch + 1) == (st_epoch + max_epochs) or (epoch + 1) % config['save_every'] == 0:
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            model_path = get_model_name(
                config, exp_name=exp_name, epoch=epoch + 1)
            if config['mGPUs']:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print("Checkpoint saved at {}".format(model_path))
        else:
            num_val_not_decreasing += 1
            # early stopping
            if num_val_not_decreasing == 4:
                print(
                    f"early stopping on epoch {str(epoch)} for experiement {exp_name}")
                break

    df_logs = pd.concat(
        [
            df_logs,
            pd.DataFrame(
                {
                    "exp_name": exp_name,
                    "AP": val_metrics["AP"],
                    "Precision": val_metrics["Precision"],
                    "Recall": val_metrics["Recall"],
                    "Forward Pass Time": val_metrics["Forward Pass Time"],
                    "Postprocess Time": val_metrics["Postprocess Time"],
                    "loss": val_metrics["loss"],
                    "train_loss": train_loss,
                }, index=[1]
            ),
        ]
    )

    print('Finished Training')

    return df_logs


def train(net, device, config, learning_rate, batch_size, max_epochs, exp_name=None):

    df_logs = pd.DataFrame()

    # Dataset and DataLoader
    train_data_loader, test_data_loader = get_data_loader(batch_size, config['use_npy'],
                                                          geometry=config['geometry'], frame_range=config['frame_range'])

    # Tensorboard Logger
    train_logger = get_logger(config, exp_name=exp_name, mode='train')
    val_logger = get_logger(config, exp_name=exp_name, mode='val')

    if config['resume_training']:
        st_epoch = config['resume_from']
    else:
        # writefile(config, 'train_loss.csv', 'iteration, cls_loss, loc_loss\n')
        # writefile(config, 'val_loss.csv', 'epoch, cls_loss, loc_loss\n')
        st_epoch = 0

    step = 1 + st_epoch * len(train_data_loader)
    cls_loss = 0
    loc_loss = 0

    best_loss = 100000000000
    num_val_not_decreasing = 0

    for epoch in range(st_epoch, st_epoch + max_epochs):

        start_time = time.time()

        train_loss = 0

        net.train()
        if config['mGPUs']:
            net.module.set_decode(False)
        else:
            net.set_decode(False)
        scheduler.step()

        for input, label_map, image_id in train_data_loader:

            tic = time.time()  # print('step', step)
            input = input.to(device)
            label_map = label_map.to(device)
            optimizer.zero_grad()

            # Forward
            predictions = net(input)
            loss, cls, loc = loss_fn(predictions, label_map)
            loss.backward()
            optimizer.step()
            cls_loss += cls
            loc_loss += loc
            train_loss += loss.item()

            if step % config['log_every'] == 0:
                cls_loss = cls_loss / config['log_every']
                loc_loss = loc_loss / config['log_every']
                train_logger.scalar_summary('cls_loss', cls_loss, step)
                train_logger.scalar_summary('loc_loss', loc_loss, step)
                cls_loss = 0
                loc_loss = 0

                # for tag, value in net.named_parameters():
                #    tag = tag.replace('.', '/')
                #    train_logger.histo_summary(tag, value.data.cpu().numpy(), step)
                #    train_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

            step += batch_size
            # print(time.time() - tic)

        # Record Training Loss
        train_loss = train_loss / len(train_data_loader)
        train_logger.scalar_summary('loss', train_loss, epoch + 1)
        print("Epoch {}|Time {:.3f}|Training Loss: {:.5f}".format(
            epoch + 1, time.time() - start_time, train_loss))

        # Run Validation
        # if (epoch + 1) % 2 == 0:
        tic = time.time()
        val_metrics, _, _, log_images = eval_batch(
            config, net, loss_fn, test_data_loader, device)
        for tag, value in val_metrics.items():
            val_logger.scalar_summary(tag, value, epoch + 1)
        val_logger.image_summary('Predictions', log_images, epoch + 1)
        print("Epoch {}|Time {:.3f}|Validation Loss: {:.5f}".format(
            epoch + 1, time.time() - tic, val_metrics['loss']))
        print(val_metrics)

        # Save Checkpoint
        # if (epoch + 1) == (st_epoch + max_epochs) or (epoch + 1) % config['save_every'] == 0:
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            model_path = get_model_name(
                config, exp_name=exp_name, epoch=epoch + 1)
            if config['mGPUs']:
                torch.save(net.module.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print("Checkpoint saved at {}".format(model_path))
        else:
            num_val_not_decreasing += 1
            # early stopping
            if num_val_not_decreasing == 4:
                print(
                    f"early stopping on epoch {str(epoch)} for experiement {exp_name}")
                break

    df_logs = pd.concat(
        [
            df_logs,
            pd.DataFrame(
                {
                    "exp_name": exp_name,
                    "AP": val_metrics["AP"],
                    "Precision": val_metrics["Precision"],
                    "Recall": val_metrics["Recall"],
                    "Forward Pass Time": val_metrics["Forward Pass Time"],
                    "Postprocess Time": val_metrics["Postprocess Time"],
                    "loss": val_metrics["loss"],
                    "train_loss": train_loss,
                }, index=[1]
            ),
        ]
    )

    print('Finished Training')

    return df_logs


def eval_one(net, loss_fn, config, loader, image_id, device, plot=False, verbose=False):
    input, label_map, image_id = loader.dataset[image_id]
    input = input.to(device)
    label_map, label_list = loader.dataset.get_label(image_id)
    loader.dataset.reg_target_transform(label_map)
    label_map = torch.from_numpy(label_map).permute(
        2, 0, 1).unsqueeze_(0).to(device)

    # Forward Pass
    t_start = time.time()
    pred = net(input.unsqueeze(0))
    t_forward = time.time() - t_start

    loss, cls_loss, loc_loss = loss_fn(pred, label_map)
    pred.squeeze_(0)
    cls_pred = pred[0, ...]

    if verbose:
        print("Forward pass time", t_forward)

    # Filter Predictions
    t_start = time.time()
    corners, scores = filter_pred(config, pred)
    t_post = time.time() - t_start

    if verbose:
        print("Non max suppression time:", t_post)

    gt_boxes = np.array(label_list)
    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                                     corners, scores, iou_threshold=0.5)

    num_gt = len(label_list)
    num_pred = len(scores)
    input_np = input.cpu().permute(1, 2, 0).numpy()
    pred_image = get_bev(input_np, corners)

    if plot == True:
        # Visualization
        plot_bev(input_np, label_list, window_name='GT')
        plot_bev(input_np, corners, window_name='Prediction')
        plot_label_map(cls_pred.numpy())

    return num_gt, num_pred, scores, pred_image, pred_match, loss.item(), t_forward, t_post


def experiment(exp_name, device, eval_range='all', plot=True):
    config, _, _, _ = load_config(exp_name)
    net, loss_fn = build_model(config, device, train=False)
    state_dict = torch.load(get_model_name(config), map_location=device)
    if config['mGPUs']:
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
    train_loader, val_loader = get_data_loader(config['batch_size'], config['use_npy'], geometry=config['geometry'],
                                               frame_range=config['frame_range'])

    # Train Set
    train_metrics, train_precisions, train_recalls, _ = eval_batch(
        config, net, loss_fn, train_loader, device, eval_range)
    print("Training mAP", train_metrics['AP'])
    fig_name = "PRCurve_train_" + config['name']
    legend = "AP={:.1%} @IOU=0.5".format(train_metrics['AP'])
    plot_pr_curve(train_precisions, train_recalls, legend, name=fig_name)

    # Val Set
    val_metrics, val_precisions, val_recalls, _ = eval_batch(
        config, net, loss_fn, val_loader, device, eval_range)

    print("Validation mAP", val_metrics['AP'])
    print("Net Fwd Pass Time on average {:.4f}s".format(
        val_metrics['Forward Pass Time']))
    print("Nms Time on average {:.4f}s".format(
        val_metrics['Postprocess Time']))

    fig_name = "PRCurve_val_" + config['name']
    legend = "AP={:.1%} @IOU=0.5".format(val_metrics['AP'])
    plot_pr_curve(val_precisions, val_recalls, legend, name=fig_name)


def test(exp_name, device, image_id):
    config, _, _, _ = load_config(exp_name)
    net, loss_fn = build_model(config, device, train=False)
    net.load_state_dict(torch.load(
        get_model_name(config), map_location=device))
    net.set_decode(True)
    train_loader, val_loader = get_data_loader(1, config['use_npy'], geometry=config['geometry'],
                                               frame_range=config['frame_range'])
    net.eval()

    with torch.no_grad():
        num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
            eval_one(net, loss_fn, config, train_loader,
                     image_id, device, plot=True)

        TP = (pred_match != -1).sum()
        print("Loss: {:.4f}".format(loss))
        print("Precision: {:.2f}".format(TP/num_pred))
        print("Recall: {:.2f}".format(TP/num_gt))
        print("forward pass time {:.3f}s".format(t_forward))
        print("nms time {:.3f}s".format(t_nms))


def prune_model(model, method="L1_instructured", prune_amt=0.2):

    params_to_prune = []

    for mod_name, nn_mod in model.named_modules():

        if isinstance(nn_mod, torch.nn.Conv2d):
            if nn_mod.bias is not None:
                params_to_prune.append((nn_mod, "bias"))

    if method == "L1_unstructured":
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_amt,
        )
    elif method == "random_unstructured":
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=prune_amt,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PIXOR custom implementation')
    parser.add_argument(
        '--mode', choices=['train', 'val', 'test', 'prune-fine-tune', 'distill'], help='name of the experiment')
    parser.add_argument('--name', required=True, help="name of the experiment")
    parser.add_argument('--device', default='cpu', help='device to train on')
    parser.add_argument('--eval_range', type=int, help="range of evaluation")
    parser.add_argument('--test_id', type=int, default=25,
                        help="id of the image to test")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    print("Using device", device)

    if args.mode == 'train':
        train_orig(args.name, device)

        # Load Hyperparameters
        # config, learning_rate, batch_size, max_epochs = load_config(args.name)
        # train(device, config, learning_rate, batch_size, max_epochs)
    if args.mode == 'val':
        if args.eval_range is None:
            args.eval_range = 'all'
        experiment(args.name, device, eval_range=args.eval_range, plot=False)
    if args.mode == 'test':
        test(args.name, device, image_id=args.test_id)

    if args.mode == "distill":
        # loading model config
        config, learning_rate, batch_size, max_epochs = load_config(args.name)
        # loading teacher config
        default_config, _, _, _ = load_config('default')

        # building model
        net, loss_fn, optimizer, scheduler = build_dist_model(
            config, device, train=True, distill=args.name)

        # loading the teacher model ('34epoch')
        teacher_net, _ = build_model(default_config, device, train=False)
        teacher_net.load_state_dict(torch.load(
            "experiments/default/34epoch", map_location=device))

        teacher_net.eval()  # setting to evaluate mode

        df_logs = train_distilled(net, loss_fn, optimizer, scheduler, teacher_net, device, config,
                                  learning_rate, batch_size, max_epochs, exp_name=args.name)

        df_logs.to_csv(args.name + "_experiment_logs.csv")

    if args.mode == 'prune-fine-tune':

        # Load Hyperparameters
        config, learning_rate, batch_size, _ = load_config(args.name)
        config["resume_training"] = True
        config["max_epochs"] = 5

        df_logs = pd.DataFrame()
        for prune_method in ["random_unstructured", "L1_unstructured"]:
            # 2,4,8,16, and 32 pruning reductation as recommended in the state of pruning paper
            for prune_amt in [0.5, 1-0.25, 1-0.125, 1-0.0625, 1-0.03125]:

                # Model
                net, loss_fn, optimizer, scheduler = build_model(
                    config, device, train=True)

                if config['resume_training']:
                    saved_ckpt_path = get_model_name(config)
                    net.load_state_dict(torch.load(
                        saved_ckpt_path, map_location=device))
                    print("Successfully loaded trained ckpt at {}".format(
                        saved_ckpt_path))

                prune_model(net, method=prune_method, prune_amt=prune_amt)

                df_logs_temp = train(net, device, config, learning_rate, batch_size,
                                     config["max_epochs"], exp_name=prune_method + "_" + str(prune_amt))

                df_logs = pd.concat([df_logs, df_logs_temp])

        df_logs.to_csv("prune_experiment_logs.csv")

    # before launching the program! CUDA_VISIBLE_DEVICES=0, 1 python main.py .......
