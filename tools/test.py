import argparse
import gorilla, os
import os.path as osp
import torch
from tqdm import tqdm
import numpy as np
from cfnet.dataset import build_dataloader, build_dataset
from cfnet import CFNet
from cfnet.utils.mask_encoder import rle_decode, rle_encode
from cfnet.utils import get_root_logger_val, save_pred_instances
import json
# torch.cuda.set_device(1)
def get_mask(spmask, superpoint):
    mask = spmask[superpoint]
    return mask

def _print_results_acc(iou_25, iou_50, mious, logger):
    logger.info(f"{'=' * 100}")
    logger.info("{0:<12}{1:<12}{2:<12}{3:<12}{4:<12}{5:<12}{6:<12}"
               .format("IoU", "zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"))
    logger.info(f"{'-' * 100}")
    line_1_str = '{:<12}'.format("0.25")
    for sub_group_type, score in iou_25.items():
        line_1_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_1_str)
    line_2_str = '{:<12}'.format("0.50")
    for sub_group_type, score in iou_50.items():
        line_2_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_2_str)
    line_3_str = '{:<12}'.format("miou")
    for sub_group_type, score in mious.items():
        line_3_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_3_str)
    logger.info(f"{'=' * 100}\n")
    
def decode_stimulus_string(s):
    """
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    :param s: the stimulus string
    """
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)

    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractors_ids


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--filename', type=str, default='best.pth', help='path to checkpoint')
    parser.add_argument('--out', default=None, type=str, help='directory for output results')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpu_ids', type=int, default=[0,1], nargs='+', help='ids of gpus to use')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)

    cfg = gorilla.Config.fromfile(args.config)
    # args.checkpoint=os.path.join(cfg.work_dir, osp.splitext(osp.basename(args.config))[0], args.filename)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger_val(log_file=args.checkpoint.replace('.pth', '_scanrefer.log'))

    model = CFNet(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    meta = gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    res = meta.get('meta')
   
    logger.info(f'Test pth epoch:{res.get("epoch",0)}, best val miou:{res.get("best_miou",0.0)}')
    print(f'Test pth epoch:{res.get("epoch",0)}, best val miou:{res.get("best_miou",0.0)}')


    dataset = build_dataset(cfg.data.val, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.val)

    scan_ids, object_ids, ann_ids, pious, spious, gt_pmasks, pred_pmasks = [], [], [], [], [], [], []
    meta_datas, view_dependents= [], []
    nt_labels = []
    progress_bar = tqdm(total=len(dataloader))
    seed_ind, sample_ind, indi = [], [], []
    with torch.no_grad():
        model.eval()
        # i=1
        for batch in dataloader:
            
            res = model(batch, mode='predict')  
    
            seed_ind.extend(res['seed_ind'])
            sample_ind.extend(res['sample_ind'])
            indi.extend(res['indi'])
            
            scan_ids.extend(res['scan_id'])
            object_ids.extend(res['object_ids'])
            ann_ids.extend(res['ann_id'])
            pious.extend(res['piou'])
            spious.extend(res['spiou'])
            nt_labels.extend(res['nt_label'])
            pred_pmasks.extend(
                [
                    rle_encode((pred_pmask>0.5).int().numpy())
                    for pred_pmask in res['pred_pmask']
                ]
            )
            gt_pmasks.extend(
                [
                    rle_encode((gt_pmask>0.5).int().numpy())
                    for gt_pmask in res['gt_pmask']
                ]
            )
            if 'meta_datas' in res:
                meta_datas.extend(res['meta_datas'])
                #view_dependents.extend(res['view_dependents'])
            # else:
            #     print('No meta_datas')
            progress_bar.update()
            # if i==1:break
        progress_bar.close()
    
    '''
    if len(meta_datas)>0:
        hardness = [decode_stimulus_string(meta_data)[2] for meta_data in meta_datas]
        ious_vd, ious_vind = [], []
        ious_easy, ious_hard = [], []
        for idx, scan_id in enumerate(scan_ids):
            piou = pious[idx]      
            if len(meta_datas)>0:
                if hardness[idx] > 2:
                    ious_hard.append(piou.item())
                else:
                    ious_easy.append(piou.item())
                
                if view_dependents[idx]:
                    ious_vd.append(piou.item())
                else:
                    ious_vind.append(piou.item())   
    '''   
    if len(nt_labels) > 0:
        eval_dict = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}
        eval_type_mask = np.empty(len(scan_ids))
        for idx, scan_id in enumerate(scan_ids):
            eval_type_mask[idx] = eval_dict[meta_datas[idx]['eval_type']]
            if nt_labels[idx]:
                pious[idx] = torch.tensor(0.0)
            if meta_datas[idx]['eval_type'] in ("zt_wo_d", "zt_w_d"):
                if nt_labels[idx]:
                    pious[idx] = torch.tensor(1.0)
                    spious[idx] = torch.tensor(1.0)
                else:
                    pious[idx] = torch.tensor(0.0)
                    spious[idx] = torch.tensor(0.0)
        pious = torch.stack(pious, dim=0).cpu().numpy()
        acc_half_results = {}
        acc_quarter_results = {}
        meam_ious = {}
        for sub_group in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt"):
            selected_indices = eval_type_mask == eval_dict[sub_group]
            selected = pious[selected_indices]    
            meam_ious[sub_group] = selected.mean()  
            acc_half_results[sub_group] = (selected > 0.5).sum().astype(float) / selected.size
            acc_quarter_results[sub_group] = (selected > 0.25).sum().astype(float) / selected.size
    else: 
        pious = torch.stack(pious, dim=0).cpu().numpy()
    precision_half = (pious > 0.5).sum().astype(float) / pious.size
    precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
    pmiou = pious.mean()
    # superpoint-level metrics
    spious = torch.stack(spious, dim=0).cpu().numpy()
    spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
    spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
    spmiou = spious.mean()
    logger.info(f'mIoU : {pmiou}')
    if len(nt_labels) > 0:
        acc_half_results["overall"] = precision_half
        acc_quarter_results["overall"] = precision_quarter
        meam_ious["overall"] = pmiou
        _print_results_acc(acc_quarter_results, acc_half_results, meam_ious, logger)
    else:
        logger.info('mIOU: {:.3f}. Acc_50: {:.3f}. Acc_25: {:.3f}'.format(pmiou, precision_half,
                                                                    precision_quarter))
    logger.info('spmIOU: {:.3f}. spAcc_50: {:.3f}. spAcc_25: {:.3f}'.format(spmiou, spprecision_half,
                                                                      spprecision_quarter))        
    
    if len(nt_labels) > 0:
        pass
    else:
        with open(os.path.join(cfg.data.val.data_root,"lookup.json"),'r') as load_f:

            # unique为1, multi为0
            unique_multi_lookup = json.load(load_f)
        unique, multi = [], []
        for idx, scan_id in enumerate(scan_ids):
            if unique_multi_lookup[scan_id][str(object_ids[idx][0])][str(ann_ids[idx])] == 0:
                unique.append(pious[idx])
            else:
                multi.append(pious[idx])
        unique = np.array(unique)
        multi = np.array(multi)
        for u in [0.25, 0.5]:
            logger.info(f'Acc@{u}: \tunique: '+str(round((unique>u).mean(), 4))+' \tmulti: '+str(round((multi>u).mean(), 4))+' \tall: '+str(round((pious>u).mean(), 4)))
        logger.info('mIoU:\t \tunique: '+str(round(unique.mean(), 4))+' \tmulti: '+str(round(multi.mean(), 4))+' \tall: '+str(round(pious.mean(), 4)))
        
    # save output
    if args.out is None:
        output = input('If you want to save the results? (y/n)')
        if output == 'y':
            # args.out = os.path.join(os.path.dirname(args.checkpoint), 'results')
            task = args.checkpoint.split('/')[-1].split('.')[0]
            args.out = os.path.join(os.path.dirname(args.checkpoint), f'results_{task}')
        else:
            logger.info('Not saving results.')
            exit()
        
    if args.out:
        logger.info('Saving pred_pmasks...')
        save_pred_instances(args.out, 'pred_pmasks', scan_ids, object_ids, ann_ids, pred_pmasks)
        logger.info('Done.')
        logger.info('Saving gt_pmasks...')
        save_pred_instances(args.out, 'gt_pmasks', scan_ids, object_ids, ann_ids, gt_pmasks)
        logger.info('Done.')

if __name__ == '__main__':
    main()
