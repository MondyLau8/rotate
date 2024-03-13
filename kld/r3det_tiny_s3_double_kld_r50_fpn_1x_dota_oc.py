_base_ = ['../r3det/r3det_r50_fpn_1x_dota_oc.py']

angle_version = 'oc'
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0110',
                    kv_stride=2),
                stages=(False, False, True, True),
                position='after_conv2')
        ]),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        reg_decoded_bbox=True,
        stacked_convs=2,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[4, 8, 16, 32, 64]),
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1.0,
            loss_weight=1.0)),
    frm_cfgs=[
        dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128]),
        dict(in_channels=256, featmap_strides=[4, 8, 16, 32, 64])
              ],
    num_refine_stages=2,
    refine_heads=[
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=3,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            reg_decoded_bbox=True,
            loss_bbox=dict(
                type='GDLoss_v1',
                loss_type='kld',
                fun='log1p',
                tau=1.0,
                loss_weight=1.0)),
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            reg_decoded_bbox=True,
            loss_bbox=dict(
                type='GDLoss_v1',
                loss_type='kld',
                fun='log1p',
                tau=1.0,
                loss_weight=1.0))
    ],
    train_cfg=dict(
        s0=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        sr=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.5,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.6,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0, 1.0])
    )
