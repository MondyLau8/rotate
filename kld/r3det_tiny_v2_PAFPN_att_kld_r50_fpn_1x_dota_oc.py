_base_ = ['../r3det/r3det_v2_r50_fpn_1x_dota_oc.py']

angle_version = 'oc'
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='1110',
                    kv_stride=4),
                stages=(False, True, True, True),
                position='after_conv2')
        ]),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1.0,
            loss_weight=1.0)),
    refine_heads=[
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=18,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
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
    ])
