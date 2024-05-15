import datetime

import os
import time
import gc
from loguru import logger

from trm.data import make_data_loader
from trm.utils.metric_logger import MetricLogger
from trm.engine.inference import inference

import mindspore
from loguru import logger
from mindspore import ops




def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    output_dir,
    checkpoint_period,
    test_period,
    arguments,
    group_params,
    max_norm=5
):
    
    
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    
    contr_weight = cfg.MODEL.TRM.LOSS.CONTRASTIVE_WEIGHT
    consis_weight = cfg.MODEL.TRM.LOSS.CONSIS_WEIGHT
    exc_weight = cfg.MODEL.TRM.LOSS.EXC_WEIGHT
    def forward_fn(batches,epoch):
        contrastive_scores, iou_scores, loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc = model(batches,cur_epoch=epoch)
        # logger.info(f'forward finished')
        # print(output,loss)
        # loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc = loss0
        loss_vid, loss_sent = loss_vid * contr_weight, loss_sent * contr_weight
        scoremap_loss_pos, scoremap_loss_neg = scoremap_loss_pos * consis_weight, scoremap_loss_neg * consis_weight,
        scoremap_loss_exc = scoremap_loss_exc * exc_weight
        meters.update(loss_vid=loss_vid, loss_sent=loss_sent, loss_iou_stnc=loss_iou_stnc, loss_iou_phrase=loss_iou_phrase, scoremap_loss_pos=scoremap_loss_pos, scoremap_loss_neg=scoremap_loss_neg, scoremap_loss_exc=scoremap_loss_exc)
        loss=0
        # loss = loss_iou_stnc
        if epoch <= cfg.SOLVER.ONLY_IOU:
                loss += loss_iou_phrase + loss_iou_stnc + (scoremap_loss_pos + scoremap_loss_neg)*0.5 + scoremap_loss_exc
                loss += loss_sent + loss_vid
        else:
            loss += loss_iou_phrase + loss_iou_stnc + (scoremap_loss_pos + scoremap_loss_neg)*0.5 + scoremap_loss_exc
            loss += (loss_sent + loss_vid) * 0.01
        # logger.info(f'loss: {loss} loss_vid: {loss_vid} loss_sent: {loss_sent} loss_iou_stnc: {loss_iou_stnc} loss_iou_phrase: {loss_iou_phrase} scoremap_loss_pos: {scoremap_loss_pos} scoremap_loss_neg: {scoremap_loss_neg} scoremap_loss_exc: {scoremap_loss_exc}')
        return loss,contrastive_scores, iou_scores
    
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    start_training_time = time.time()
    end = time.time()
    max_iteration = len(data_loader)
    writer_count = 0
    def train_step(data):
        # logger.info(f'grad_fn start')
        (loss, contrastive_scores, iou_scores), grads = grad_fn(data,epoch)
        # logger.info(f'grad_fn finished')
        optimizer(grads)
        return loss

    for epoch in range(arguments["epoch"], max_epoch + 1):
        model.set_train()
        rest_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch
        # if epoch <= cfg.SOLVER.FREEZE_BERT:
        #     for param in group_params[0]['params']:
        #         param.requires_grad(False)
        # else:
        #     for param in group_params[0]['params']:
        #         param.requires_grad(True)
        # logger.info("Start epoch {}. base_lr={:.1e}, bert_lr={:.1e}, bert.requires_grad={}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], str(param_dict['bert'][0].requires_grad)))
        if epoch <= cfg.SOLVER.ONLY_IOU:
            logger.info("Using all losses")
        else:
            logger.info("Using only bce loss")
        for iteration, batches in enumerate(data_loader.create_dict_iterator(num_epochs=1)):
            writer_count += 1
            iteration += 1
            # loss,output=forward_fn(model,batches,epoch)
            loss = train_step(batches)
            # loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc = model(batches, cur_epoch=epoch)
            # model(batches,cur_epoch=epoch)
            # optimizer(grads)
            logger.info(f'Epoch:{epoch} it:{iteration} optimizer finished')
            # if max_norm > 0:
            #     ops.clip_by_norm(grads, max_norm)
            # optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (max_iteration - iteration + rest_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                    )
                )
            gc.collect()

        if checkpoint_period != -1 and epoch % checkpoint_period == 0:
            logger.info('saving model')
            mindspore.save_checkpoint(model, os.path.join(output_dir, f"{cfg.MODEL.TRM.FEAT2D.NAME}_model_{epoch}e.ckpt"))
            # checkpointer.save(f"{cfg.MODEL.TRM.FEAT2D.NAME}_model_{epoch}e", **arguments)
            logger.info('model saved')

        if data_loader_val is not None and test_period > 0 and epoch % test_period == 0 and epoch >= cfg.SOLVER.SKIP_TEST:
            # synchronize()
            # torch.cuda.empty_cache()
            model.set_train(False)
            result_dict = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            # synchronize()
            
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
