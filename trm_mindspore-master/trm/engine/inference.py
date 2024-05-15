import logging
from trm.data.datasets.evaluation import evaluate
from trm.utils.metric_logger import MetricLogger
from ..utils.timer import Timer, get_time_str
from loguru import logger

def compute_on_dataset(model, data_loader, device, timer=None):
    model.set_train(False)
    results_dict = {}

    meters = MetricLogger(delimiter="  ")
    for batch in data_loader.create_dict_iterator(num_epochs=1):  # use tqdm(data_loader) for showing progress bar
        batches = batch
        idxs = batches['index'].asnumpy().tolist()
        if timer:
            timer.tic()
        contrastive_scores, phrase_iou_scores, loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc = model(batches)
        meters.update(loss_vid=loss_vid, loss_sent=loss_sent, loss_iou_stnc=loss_iou_stnc, loss_iou_phrase=loss_iou_phrase, scoremap_loss_pos=scoremap_loss_pos, scoremap_loss_neg=scoremap_loss_neg)
        if timer:
            timer.toc()
        contrastive_output, iou_output = [o.asnumpy() for o in contrastive_scores], [o.asnumpy() for o in phrase_iou_scores]
        results_dict.update(
            {video_id: {'contrastive': result1, 'iou': result2} for video_id, result1, result2 in zip(idxs, contrastive_output, iou_output)}
        )
        logger.info(str(meters))
    logger.info(str(meters))
    return results_dict




def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    results = []
    for d_loader in data_loader:
        dataset = d_loader
        # logger.info("Start evaluation on {} (Size: {}).".format(dataset.ann_name, len(dataset)))
        inference_timer = Timer()
        predictions = compute_on_dataset(model, d_loader, device, inference_timer)
        # wait for all processes to complete before measuring the time
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({:.03f} s / inference per device)".format(
                total_infer_time,
                inference_timer.total_time / len(dataset),
            )
        )
        results.append(evaluate(cfg, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh))
    return results
